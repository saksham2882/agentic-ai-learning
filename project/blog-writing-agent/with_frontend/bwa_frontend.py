import streamlit as st
import uuid
import sys
import re
from pathlib import Path
from datetime import datetime


# ============ Path setup ==============
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# ========== Import from backend =========
from bwa_backend import (
    app,
    retrieve_all_threads,
    delete_thread,
    get_thread_state,
    db_status_info,
    BASE_DIR,
    IMAGES_DIR
)

# ========== Page configuration ==========
st.set_page_config(
    page_title="AI Blog Writing Chatbot",
    layout="wide"
)

# ========== Custom Styling for clean chatbot experience ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.95rem;
    }
    .db-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============== Session State Setup ====================

def generate_thread_id() -> str:
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "**Hello! I am your AI Blog Writing Assistant.**\n\nI can write a high-quality blog on **any topic** you like — technology, business, health, travel, finance, productivity, or lifestyle.\n\nJust tell me what you'd like to write about!"
        }
    ]

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


# =============== Helper functions ===================

def refresh_threads():
    st.session_state["chat_threads"] = retrieve_all_threads()


def switch_thread(tid: str):
    st.session_state["thread_id"] = tid
    # Load past state
    state = get_thread_state(tid)
    final_blog = state.get("final")
    topic = state.get("topic", "Blog")

    st.session_state["messages"] = [
        {"role": "user", "content": f"Write a blog on: {topic}"}
    ]
    if final_blog:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": final_blog,
            "is_blog": True
        })
    else:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"Loaded chat for topic: **{topic}**."
        })


def new_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "**Started a new chat!** What topic would you like to write about today?"
        }
    ]
    refresh_threads()


# ============ Sidebar: Saved Chats & Database Status ==============

with st.sidebar:
    st.markdown("### AI Blog Writer")
    
    # DB Status badge
    if db_status_info.get("type") == "postgres":
        st.markdown('<div class="db-status">PostgreSQL Connected (:5442)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="db-status" style="color: #F59E0B; background: rgba(245, 158, 11, 0.12);">SQLite Saved</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("New Chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("#### Past Chats & Blogs")

    refresh_threads()
    threads = st.session_state["chat_threads"]

    if not threads:
        st.caption("No saved chats yet. Start writing a blog!")
    else:
        for t in threads:
            tid = t["thread_id"]
            title = t.get("title") or t.get("topic") or tid[:8]
            is_active = tid == st.session_state["thread_id"]

            col_btn, col_del = st.columns([0.82, 0.18])
            with col_btn:
                btn_name = f"{'👉 ' if is_active else ''} {title[:20]}..."
                if st.button(btn_name, key=f"th_{tid}", use_container_width=True):
                    switch_thread(tid)
                    st.rerun()
            with col_del:
                if st.button("Delete Chat", key=f"del_{tid}", help="Delete chat"):
                    delete_thread(tid)
                    if st.session_state["thread_id"] == tid:
                        new_chat()
                    else:
                        refresh_threads()
                    st.rerun()


# =============== Main Chat UI =====================

st.markdown("""
<div class="main-header">
    <h1>AI Blog Writing Assistant</h1>
    <p>Powered by LangGraph, Tavily Search & PostgreSQL Persistence</p>
</div>
""", unsafe_allow_html=True)


# ========== Helper to render images in markdown if path exists ==========
def render_content_with_images(text: str):
    parts = re.split(r'(!\[.*?\]\(.*?\))', text)

    for part in parts:
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', part)

        if img_match:
            alt = img_match.group(1)
            img_path = img_match.group(2)
            resolved = BASE_DIR / img_path

            if resolved.exists():
                st.image(str(resolved), caption=alt, use_container_width=True)
            else:
                st.markdown(part)
        else:
            if part.strip():
                st.markdown(part)


# =========== Display chat history ==============
for idx, msg in enumerate(st.session_state["messages"]):

    with st.chat_message(msg["role"]):
        render_content_with_images(msg["content"])

        # If it's a blog post, add a convenient download button
        if msg.get("is_blog"):
            st.download_button(
                label="Download Blog (.md)",
                data=msg["content"],
                file_name="blog_post.md",
                mime="text/markdown",
                key=f"dl_{idx}"
            )


# ============ Chat Input & Agent Stream Execution ==============

user_input = st.chat_input("Enter a topic....")

if user_input:

    # 1. Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant turn with thinking and tool steps
    with st.chat_message("assistant"):

        # Real-time thinking and step-by-step indicator
        status_box = st.status("Thinking & planning...", expanded=True)

        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        payload = {
            "topic": user_input,
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "recency_days": 30,
            "sections": []
        }

        final_blog_content = ""

        try:
            # Stream events as the graph executes
            for event in app.stream(payload, config=config, stream_mode="updates"):
                node_name = list(event.keys())[0] if event else ""
                node_data = event.get(node_name, {})

                if node_name == "router":
                    mode = node_data.get("mode", "hybrid")
                    queries = node_data.get("queries", [])
                    status_box.write(f"**Topic Analysis**: Selected `{mode}` mode.")
                    if queries:
                        status_box.write(f"**Search Queries**: {', '.join([f'`{q}`' for q in queries[:3]])}")

                elif node_name == "research":
                    evidence = node_data.get("evidence", [])
                    status_box.write(f"**Web Search (Tavily)**: Found {len(evidence)} verified sources.")

                elif node_name == "orchestrator":
                    plan = node_data.get("plan")
                    if plan:
                        status_box.write(f"**Outline Created**: *{plan.blog_title}* ({len(plan.tasks)} sections planned)")

                elif node_name == "worker":
                    status_box.write("**Writing Sections**: Writing comprehensive content...")

                elif node_name == "reducer":
                    final_blog_content = node_data.get("final", "")
                    status_box.write("**Polishing & Visuals**: Merging sections & placing illustrations...")

            status_box.update(label="Blog written successfully!", state="complete", expanded=False)

            # If not captured in reducer update, fetch from state
            if not final_blog_content:
                state_now = get_thread_state(st.session_state["thread_id"])
                final_blog_content = state_now.get("final", "Blog generated successfully!")

            # Render generated blog
            render_content_with_images(final_blog_content)

            # Download button
            st.download_button(
                label="Download Blog (.md)",
                data=final_blog_content,
                file_name=f"{user_input[:20].lower().replace(' ', '_')}.md",
                mime="text/markdown"
            )

            # Save in session history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_blog_content,
                "is_blog": True
            })

            refresh_threads()

        except Exception as e:
            status_box.update(label="Generation issue occurred", state="error", expanded=True)
            st.error(f"Error: {e}")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"An error occurred while generating the blog: {e}"
            })
