from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from datetime import datetime
import operator
import re
import os
import hashlib
import base64
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# ========== Checkpointer imports for persistence ==========
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


# ========== Load env ===========
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ============ 1. Schema ================

# Define the structure of a single task using Pydantic
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(...,
        description="One sentence describing what the reader should be able to do/understand after this section."
    )
    bullets: List[str] = Field(..., 
        min_length=2, 
        max_length=6, 
        description="2-6 concrete, non-overlapping subpoints to cover in this section."
    )
    target_words: int = Field(..., 
        description="Target word count for this section (100-500)."
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design", "lifestyle", "guide"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None     # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="Unique placeholder, e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Image filename, e.g. energy_flow.png")
    alt: str
    caption: str
    prompt: str
    aspect_ratio: Literal["1:1","16:9","9:16","4:3","3:4"] = "16:9"
    image_size: Literal["1K","2K","4K"] = "1K"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)



# =============== 2. Define State ===============

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    
    final: str


# ============== LLM ==============
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# ============= 3. Router (decide upfront) ================

ROUTER_SYSTEM = """
You are a routing module for a comprehensive blog planner.
You can handle ANY topic — technology, science, finance, health, lifestyle, travel, history, or business.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals, timeless advice).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples, real-world tools, or latest trends to be useful.
- open_book (needs_research=true):
  Mostly volatile: latest news, weekly roundups, current rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–8 high-signal queries.
- Queries should be scoped and specific to the given topic.
- If user asked for "latest/recent", reflect that constraint IN THE QUERIES.

For time-sensitive research:
- Respect the provided as_of date.
- Use recency_days to scope recent research.
- Include relevant date constraints in search queries.
- Never treat undated sources as confirmed recent developments.
"""

def router_node(state: State) -> dict:
    topic = state.get("topic", "General Topic")
    as_of = state.get("as_of") or datetime.now().strftime("%Y-%m-%d")
    recency_days = state.get("recency_days") or 30

    decider = llm.with_structured_output(RouterDecision)
    try:
        decision = decider.invoke(
            [
              SystemMessage(content=ROUTER_SYSTEM),
              HumanMessage(
                  content=(
                      f"Topic: {topic}\n"
                      f"As of date: {as_of}\n"
                      f"Research window: last {recency_days} days"
                    )
              )
            ]
        )
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries,
            "as_of": as_of,
            "recency_days": recency_days
        }
    except Exception as e:
        print(f"Router error: {e}")
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
            "as_of": as_of,
            "recency_days": recency_days
        }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research", False) else "orchestrator"



# ================ 4. Research (Tavily) ================

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return []
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})

        normalized: List[dict] = []
        for r in results or []:
            normalized.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return normalized
    except Exception as e:
        print(f"Tavily search warning: {e}")
        return []


# ================= Research node =================

RESEARCH_SYSTEM = """
You are a research synthesizer for blog writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short and factual.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])
    max_results = 5

    raw_results: List[dict] = []
    for q in queries[:6]:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    try:
        extractor = llm.with_structured_output(EvidencePack)
        pack = extractor.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=f"Raw results:\n{raw_results[:12]}"),
            ]
        )

        allowed_urls = {
            r["url"] for r in raw_results if r.get("url")
        }

        dedup = {}
        for e in pack.evidence:
            if e.url in allowed_urls:
                dedup[e.url] = e

        return {"evidence": list(dedup.values())}
    except Exception as e:
        print(f"Evidence synthesis error: {e}")
        simple_list = [
            EvidenceItem(title=r.get("title", "Source"), url=r.get("url", ""), snippet=r.get("snippet", ""))
            for r in raw_results if r.get("url")
        ]
        return {"evidence": simple_list[:8]}



# ================ 5. Orchestrator (Plan) ==================

ORCH_SYSTEM = """
You are an expert blog strategist and writer.
Your job is to produce a well-structured, engaging, and actionable outline for a blog post on ANY topic.

Hard requirements:
- Create 4–7 sections (tasks) suitable for the topic and target audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–5 bullets that are concrete, specific, and non-overlapping
  3) target word count (150–450)

Quality bar:
- Match the tone to the audience (e.g. conversational, informative, or technical).
- For technical topics, include at least one minimal code sketch / practical steps if helpful (requires_code=True).
- For non-technical topics (lifestyle, health, business, guides), provide practical examples, action steps, or real-life tips.

Grounding rules:
- Mode closed_book: keep it evergreen; rely on general knowledge.
- Mode hybrid: use evidence for up-to-date examples and tools.
- Mode open_book: summarize latest facts, events, and implications.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")
    topic = state.get("topic", "General Topic")

    try:
        plan = planner.invoke(
            [
                SystemMessage(content=ORCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n"
                        f"Mode: {mode}\n\n"
                        f"Evidence (may be empty):\n"
                        f"{[e.model_dump() for e in evidence][:12]}"
                    )
                ),
            ]
        )
        return {"plan": plan}

    except Exception as e:
        print(f"Orchestrator error: {e}")
        
        fallback_plan = Plan(
            blog_title=f"The Complete Guide to {topic}",
            audience="General Readers & Enthusiasts",
            tone="Informative, engaging, practical",
            blog_kind="explainer",
            tasks=[
                Task(id=1, title="Introduction & Why It Matters", goal="Introduce the core topic and why it is important.", bullets=["Overview and background", "Why it matters today", "What this guide will cover"], target_words=200),
                Task(id=2, title="Core Concepts & Fundamentals", goal="Explain the foundational concepts clearly.", bullets=["Core principles", "How it works in practice", "Key examples"], target_words=300),
                Task(id=3, title="Actionable Tips & Best Practices", goal="Provide practical tips and common mistakes to avoid.", bullets=["Step-by-step best practices", "Common pitfalls to avoid", "Expert recommendations"], target_words=300),
                Task(id=4, title="Conclusion & Key Takeaways", goal="Summarize main points and next steps.", bullets=["Key takeaways", "Action items", "Final thoughts"], target_words=200)
            ]
        )
        return {"plan": fallback_plan}



# ================ 6. Fanout ===================

def fanout(state: State):
    plan = state.get("plan")
    if not plan or not plan.tasks:
        return []

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": plan.model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in plan.tasks
    ]



# ================ 7. Worker ==================

WORKER_SYSTEM = """
You are an expert blog writer.
Write ONE section of a blog post in clean, beautiful Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order.
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no conversational chatter).
- Start with a '## <Section Title>' heading.

Style:
- Engaging paragraphs, bullet points where helpful.
- If requires_code is true, include a clean, helpful code snippet.
- For outside claims, cite evidence URLs as Markdown links ([Source](URL)) if available.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- Title: {e.title}\n"
            f"  URL: {e.url}\n"
            f"  Snippet: {e.snippet or ''}"
            for e in evidence[:10]
        )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Topic: {topic}\n"
                        f"Mode: {mode}\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence:\n{evidence_text}\n"
                    )
                ),
            ]
        )

        content = response.content
        if isinstance(content, list):
            section_md = "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            ).strip()
        else:
            section_md = content.strip()

    except Exception as e:
        section_md = f"## {task.title}\n\n{task.goal}\n\n*(Section generation note: {e})*"

    return {"sections": [(task.id, section_md)]}



# ===============  8. Reducer With Images (subgraph) =====================

def merge_content(state: State) -> dict:
    plan = state.get("plan")
    title = plan.blog_title if plan else state.get("topic", "Blog Post")

    ordered_sections = [md for _, md in sorted(state.get("sections", []), key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {title}\n\n{body}\n"
    
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """
You are a professional technical blog visual director
and image prompt engineer.

Your job is to analyze the complete blog content and
create highly relevant image specifications.

IMAGE RELEVANCE RULES:

1. Every image must directly represent a specific concept,
   technology, process, or real-world subject discussed
   in the blog.

2. Never generate generic technology images just to
   make the blog visually attractive.

3. Read the surrounding section carefully before
   deciding what image is required.

4. The image must visually explain the actual subject
   rather than simply represent its broad category.

5. Avoid repeating the same visual concept across images.

PROMPT QUALITY RULES:

Every image prompt must describe:

- Exact main subject.
- Specific objects and components visible.
- Environment and background.
- Composition and camera angle.
- Lighting and visual style.
- Important visual details.
- Relationship between objects, if relevant.

ACCURACY RULES:

- Never invent real-world facts, statistics or data.
- Never invent company logos, product interfaces or
  real-world infrastructure details.
- Do not represent fictional objects as real products.
- Do not create misleading technical diagrams.
- Do not add unrelated objects or visual elements.

IMAGE GENERATION RULES:

- Create professional editorial-quality images.
- Use realistic materials, natural lighting and
  detailed textures where appropriate.
- Prefer photorealistic editorial photography
  for real-world subjects.
- Prefer clean 3D illustrations for abstract
  infrastructure and system concepts.
- Use a consistent visual style throughout the blog.
- Use landscape composition for blog illustrations.
- Do not generate text, captions, labels, watermarks,
  logos or random typography inside images.
- Do not generate collages or split-screen compositions.

IMAGE LIMIT:

- Maximum 3 images.
- Generate images only when they add real value.
- Every placeholder must appear exactly once.
- Every image specification must have a matching
  placeholder.
- If no useful image is required, return images=[].

IMPORTANT:

A technically accurate diagram must not be fabricated
by an image generation model.

If a concept requires exact labels, data, architecture,
steps or relationships, do not create a misleading
illustration. Prefer a programmatically generated diagram.

Return strictly according to GlobalImagePlan schema.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state.get("merged_md", "")
    plan = state.get("plan")
    title = plan.blog_title if plan else state.get("topic", "")

    try:
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"

                    "Analyze the complete blog carefully.\n"
                    "Identify the exact concepts that need visual "
                    "representation.\n\n"

                    "For each image:\n"
                    "- Identify the section it belongs to.\n"
                    "- Understand the section's actual meaning.\n"
                    "- Create a specific visual representation.\n"
                    "- Avoid generic or decorative images.\n"
                    "- Ensure the image matches the surrounding content.\n\n"

                    f"Complete blog content:\n{merged_md}"
                    )
                )
            ]
        )
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images],
        }
    except Exception as e:
        print(f"Decide images error: {e}")
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
        }


def _cloudflare_generate_image(prompt: str, spec: dict):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError("Cloudflare credentials not configured.")

    model = "@cf/black-forest-labs/flux-1-schnell"
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    enhanced_prompt = f"""
    Create a professional editorial-quality image.

    Subject:
    {prompt}

    Visual requirements:
    - Highly detailed and visually coherent.
    - Accurate representation of the specified subject.
    - Professional composition.
    - Natural lighting and realistic textures.
    - Clear visual hierarchy.
    - No unnecessary objects.
    - No text, labels, logos or watermarks.
    - No blurry details.
    - No random visual elements.

    The image must focus entirely on the specified subject.
    """

    payload = {
        "prompt": enhanced_prompt,
        "steps": 4,
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    data = response.json()

    if not response.ok or not data.get("success", False):
        raise RuntimeError(f"Cloudflare failed: {data.get('errors', response.text)}")

    image_base64 = data["result"]["image"]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image



# ================= Generate and Place Image in Markdown =================

def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")
    title = plan.blog_title if plan else state.get("topic", "blog")
    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    image_specs = state.get("image_specs", []) or []

    def safe_slug(value: str) -> str:
        slug = value.lower()
        slug = re.sub(r'[<>:"/\\|?*]', "_", slug)
        slug = re.sub(r"[\s_]+", "_", slug)
        return slug.strip(" ._")[:80] or "blog"

    blog_slug = safe_slug(title)
    blog_dir = IMAGES_DIR / blog_slug
    blog_dir.mkdir(parents=True, exist_ok=True)

    markdown_filename = BASE_DIR / f"{blog_slug}.md"

    if not image_specs:
        clean_md = re.sub(r'\[\[IMAGE_\d+\]\]', '', md)
        markdown_filename.write_text(clean_md, encoding="utf-8")
        return {"final": clean_md}

    image_specs = image_specs[:3]
    seen_placeholders = set()

    for spec in image_specs:
        placeholder = spec.get("placeholder", "")
        prompt = spec.get("prompt", "")

        if not placeholder or placeholder in seen_placeholders or placeholder not in md:
            continue
        seen_placeholders.add(placeholder)

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        image_filename = f"image_{prompt_hash}.png"
        out_path = blog_dir / image_filename

        if not out_path.exists():
            try:
                img = _cloudflare_generate_image(prompt, spec)
                img.save(out_path, format="PNG")
            except Exception as e:
                # If image generation fails, add clean placeholder note
                prompt_block = (
                    f"\n> *[Image: {spec.get('alt', 'Illustration')} - {spec.get('caption', '')}]*\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        image_rel_path = f"images/{blog_slug}/{image_filename}"
        img_md = f"\n![{spec.get('alt', 'Blog Image')}]({image_rel_path})\n\n*{spec.get('caption', '')}*\n"
        md = md.replace(placeholder, img_md)

    clean_md = re.sub(r'\[\[IMAGE_\d+\]\]', '', md)
    markdown_filename.write_text(clean_md, encoding="utf-8")

    return {"final": clean_md}


# ============== build reducer subgraph =====================

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


# =============== 9. Build main graph ==================

graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("reducer", reducer_subgraph)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge("research", "orchestrator")

graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)


# =============== 10. PostgreSQL Connection & Checkpointer ================

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5442/blog_generator")
db_status_info = {"status": "connected", "type": "postgres"}

try:
    pool = ConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True})
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
except Exception as e:
    print(f"Warning: PostgreSQL connection failed: {e}. Falling back to SQLite...")

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(str(BASE_DIR / "blog_history.db"), check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    db_status_info = {"status": "fallback", "type": "sqlite"}


# =========== Compile graph with persistence ==============
app = graph.compile(checkpointer=checkpointer)


# ========== 11. Helper Functions for Frontend ============

def retrieve_all_threads():
    """Returns list of all saved thread IDs and topics from database."""
    threads = []
    seen = set()
    try:
        for cp in checkpointer.list(None):
            cfg = cp.config or {}
            tid = cfg.get("configurable", {}).get("thread_id")
            if tid and tid not in seen:
                seen.add(tid)
                channel_values = cp.checkpoint.get("channel_values", {})
                topic = channel_values.get("topic")
                plan = channel_values.get("plan")
                title = getattr(plan, "blog_title", None) if plan else topic or f"Chat {tid[:8]}"
                threads.append({"thread_id": tid, "title": title})
    except Exception as e:
        print(f"Error reading threads: {e}")
    return threads


def delete_thread(thread_id: str):
    """Deletes thread checkpoints from database."""
    try:
        import psycopg
        conn = psycopg.connect(DB_URL, autocommit=True)
        cur = conn.cursor()
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (thread_id,))
        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (thread_id,))
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (thread_id,))
        conn.close()
        return True
    except Exception:
        return False


def get_thread_state(thread_id: str):
    """Returns the state dictionary for a given thread."""
    try:
        state = app.get_state({"configurable": {"thread_id": thread_id}})
        return state.values or {}
    except Exception:
        return {}
