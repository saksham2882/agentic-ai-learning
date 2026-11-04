<h1 align="center">🤖 Agentic AI Learning</h1>

<p align="center">
  A comprehensive, hands-on master curriculum progressing from core <b>LangGraph</b> state machines to production-grade agentic architectures:<br/>
  stateful chatbots, tool execution, Model Context Protocol (MCP), RAG evolutions (CRAG & Self-RAG), short/long-term memory, human-in-the-loop, and a multi-agent blog writer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker"/>
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Study Notes (Concepts & Terminology)](#-study-notes-concepts--terminology)
- [Repository Roadmap & Curriculum](#-repository-roadmap--curriculum)
- [Capstone Project: Multi-Agent Blog Writer](#-capstone-project-multi-agent-blog-writer)
- [Architecture & Workflow Progression](#-architecture--workflow-progression)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#1-installation)
  - [Environment Configuration](#2-environment-configuration)
  - [Database Setup (Docker)](#3-database-setup-docker)
- [Running the Applications](#-running-the-applications)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Streamlit Web Apps](#streamlit-web-apps)
  - [MCP Server & Client](#mcp-server--client)
- [Repository Directory Structure](#-repository-directory-structure)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Contact & Support](#-contact--support)

---

## 🔭 Overview

This repository is an engineered, step-by-step practical guide to **Agentic AI**. Rather than isolated snippets, each numbered directory represents a self-contained module that builds upon the previous concept—transitioning from basic state graphs to resilient multi-agent collaboration with persistent memory and self-correcting RAG.

> **Learning Repository:** Code favors transparency, clear state inspection, and pedagogical clarity over opaque abstractions.

---

## 📖 Study Notes (Concepts & Terminology)

To accompany the code, a comprehensive study guide is available in **[`NOTES.md`](./NOTES.md)**. It breaks down complex ideas into intuitive, plain-language explanations without confusing academic jargon:
- Differences between traditional chatbots, linear chains, and autonomous agents
- LangGraph mechanics (`StateGraph`, `MessagesState`, `Send`, `Command`, `interrupt`)
- Memory paradigms (Short-Term trimming & summarization vs. Long-Term persistent profiles)
- Advanced RAG evaluation (Corrective RAG vs. Self-Reflective RAG)
- Model Context Protocol (MCP) as an open standard for AI tools
- Complete LangGraph API cheat sheet and glossary

👉 **Read the full guide: [`NOTES.md`](./NOTES.md)**

---

## 📚 Repository Roadmap & Curriculum

| # | Module Folder | Topic | Core Implementations & Notebooks |
|---|---|---|---|
| **01** | [`01_langgraph_tutorials`](./01_langgraph_tutorials) | **LangGraph Fundamentals** | 12 notebooks covering sequential, parallel, conditional, and iterative loops (BMI calculator, batsman stats, essay reviewer, quadratic solver, tweet generator, checkpointer persistence, fault tolerance). |
| **02** | [`02_langgraph_streamlit_chatbot`](./02_langgraph_streamlit_chatbot) | **Streaming Chatbot** | Streamlit UI with streaming token output (`graph.stream`) and session state multi-turn conversation. |
| **03** | [`03_langgraph_streamlit_SQLite_DB_chatbot`](./03_langgraph_streamlit_SQLite_DB_chatbot) | **Persistent Storage** | Persistent chat history backed by SQLite checkpointer (`SqliteSaver`) across page reloads and threads. |
| **04** | [`04_langSmith_tutorials`](./04_langSmith_tutorials) | **Observability with LangSmith** | End-to-end tracing of individual LLM calls, sequential chains, RAG indexing pipelines, `@traceable` decorators, and agents. |
| **05** | [`05_langgraph_chatbot_observability`](./05_langgraph_chatbot_observability) | **Observable Chatbot** | Production-style observability integrating LangSmith project tracing with SQLite checkpointer state. |
| **06** | [`06_langgraph_chatbot_with_tools`](./06_langgraph_chatbot_with_tools) | **Tool Calling** | LangGraph `ToolNode` and `tools_condition` with Alpha Vantage stock prices and DuckDuckGo search integration. |
| **07** | [`07_langgraph_mcp_server`](./07_langgraph_mcp_server) | **FastMCP Server** | Creating a standards-compliant Model Context Protocol server exposing arithmetic and utility tools. |
| **08** | [`08_langgraph_mcp_client`](./08_langgraph_mcp_client) | **MCP Client** | Connecting LangGraph agents to local (stdio) and remote (SSE/HTTP) MCP tool servers (sync & async clients). |
| **09** | [`09_langgraph_streamlit_mcp_chatbot`](./09_langgraph_streamlit_mcp_chatbot) | **MCP Streamlit Chatbot** | Interactive web UI powered by dynamic MCP tool discovery and execution. |
| **10** | [`10_rag_with_langgraph`](./10_rag_with_langgraph) | **Foundational Graph RAG** | Document loading (PDF), chunking, FAISS vector indexing, retrieval nodes, and grounded generation. |
| **11** | [`11_rag_chatbot`](./11_rag_chatbot) | **Interactive RAG App** | Upload custom PDFs directly via Streamlit, build per-thread FAISS indexes, and chat with documents. |
| **12** | [`12_human_in_the_loop_using_langgraph`](./12_human_in_the_loop_using_langgraph) | **Human-in-the-Loop (HITL)** | Pausing execution using breakpoints/interrupts, state inspection, and manual state modification (`update_state`). |
| **13** | [`13_subgraph_in_langgraph`](./13_subgraph_in_langgraph) | **Modular Subgraphs** | Nesting graphs inside parent graphs; managing isolated state vs. shared global state. |
| **14** | [`14_llm_memory`](./14_llm_memory) | **Memory Architecture** | Foundations of memory in LLM systems: stateful vs. stateless designs. |
| **15** | [`15_llm_short_term_memory`](./15_llm_short_term_memory) | **Short-Term Memory (STM)** | 5 notebooks: thread persistence, token window trimming (`trim_messages`), message deletion (`RemoveMessage`), and dynamic summarization. |
| **16** | [`16_llm_long_term_memory`](./16_llm_long_term_memory) | **Long-Term Memory (LTM)** | 5 notebooks: cross-session profile storage, namespaces (`InMemoryStore`, `PostgresStore`), semantic facts reading & writing. |
| **17** | [`17_corrective_rag`](./17_corrective_rag) | **Corrective RAG (CRAG)** | 6 notebooks: automated retrieval grading (relevant/irrelevant), web-search fallback (Tavily/DDG), and query rewriting. |
| **18** | [`18_self_rag`](./18_self_rag) | **Self-Reflective RAG (Self-RAG)** | 7 notebooks: retrieve on-demand, document relevance grading, answer faithfulness (hallucination check), answer usefulness, and self-correction loops. |
| **Project** | [`project/blog-writing-agent`](./project/blog-writing-agent) | **Capstone: Blog Writing Agent** | Full multi-agent application with topic routing, web research, parallel section writing (Map-Reduce), reducer subgraph, AI image generation, and Streamlit UI. |

---

## ✍️ Capstone Project: Multi-Agent Blog Writer

Located in [`project/blog-writing-agent`](./project/blog-writing-agent), this production-style application demonstrates the **Orchestrator-Worker (Map-Reduce)** pattern combined with subgraphs and persistence.

### Architectural Flow

```mermaid
flowchart TD
    Start([User Topic]) --> Router{Router Node}
    Router -->|Needs Research| Research[Research Agent: Tavily Search]
    Router -->|No Research Needed| Orchestrator
    Research --> Orchestrator[Orchestrator: Generates Structured Section Plan]
    
    Orchestrator -->|Dynamic Fan-Out via Send API| W1[Worker Agent 1: Section 1]
    Orchestrator -->|Dynamic Fan-Out via Send API| W2[Worker Agent 2: Section 2]
    Orchestrator -->|Dynamic Fan-Out via Send API| WN[Worker Agent N: Section N]
    
    subgraph ReducerSubgraph["Reducer Subgraph"]
        M[Merge & Format Sections] --> D[Decide Image Prompts & Locations]
        D --> G[Generate & Download Images: Cloudflare Flux]
    end
    
    W1 --> ReducerSubgraph
    W2 --> ReducerSubgraph
    WN --> ReducerSubgraph
    
    ReducerSubgraph --> FinalOutput([Complete Markdown Blog Post with Images])
```

### Key Engineering Features
- **Structured Planning:** Pydantic schemas enforce word count, bullet points, citation requirements, and audience tone for each section.
- **Dynamic Parallel Workers:** LangGraph's `Send()` API spawns concurrent workers per planned section, cutting generation time significantly.
- **Automated Research:** Evaluates topic complexity and gathers fresh web sources via Tavily Search API.
- **Image Generation Pipeline:** Generates tailored image prompts for sections, requests images from Cloudflare Workers AI (`flux-1-schnell`), downloads and stores them locally, and inserts proper Markdown references.
- **Dual Persistence:** Automatically connects to PostgreSQL 16 checkpointer (`PostgresSaver`) via connection pool, with seamless automatic fallback to local SQLite (`SqliteSaver`) if the database is offline.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Graph Orchestration** | LangGraph, LangChain Core |
| **Language Models** | Google Gemini (`gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-2.5-flash`) via `langchain-google-genai` |
| **Tool Protocol** | Model Context Protocol (MCP) via `fastmcp` and `langchain-mcp-adapters` |
| **Vector Search & RAG** | FAISS (`faiss-cpu`), HuggingFace Embeddings (`all-MiniLM-L6-v2`), PyPDF |
| **State Persistence** | SQLite (`langgraph-checkpoint-sqlite`), PostgreSQL 16 (`langgraph-checkpoint-postgres`, `psycopg-pool`) |
| **Observability** | LangSmith (Traces, Spans, Metrics, Token Tracking) |
| **External APIs** | Tavily Search, Alpha Vantage, Cloudflare Workers AI (`flux-1-schnell`), DuckDuckGo |
| **UI & Presentation** | Streamlit, Jupyter Notebooks |
| **Infrastructure** | Docker, Docker Compose |

---

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.10 or higher
- **Docker & Docker Compose:** Required only for running the PostgreSQL checkpointer container.
- **API Keys:** Google Gemini API key (free tier available), plus optional keys for Tavily, Alpha Vantage, and Cloudflare.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/saksham2882/agentic-ai-learning.git
cd agentic-ai-learning

# Create and activate virtual environment
python -m venv myenv
# On Windows (PowerShell):
.\myenv\Scripts\Activate.ps1
# On macOS / Linux:
source myenv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

#### Extended Dependencies for Advanced Modules:
Some modules (RAG, MCP, PostgreSQL persistence, and Blog Agent) require additional specialized libraries. Install them with:

```bash
pip install langchain-community langchain-huggingface langchain-text-splitters \
            langchain-mcp-adapters fastmcp faiss-cpu pypdf \
            langgraph-checkpoint-sqlite langgraph-checkpoint-postgres \
            "psycopg[binary,pool]" pydantic pillow requests duckduckgo-search \
            tavily-python
```

### 2. Environment Configuration

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

| Variable | Required In | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | **All Modules** | Powers Gemini LLM calls across all graphs and notebooks. |
| `LANGCHAIN_TRACING_V2` | Folders `04`, `05` | Set to `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | Folders `04`, `05` | API key from [smith.langchain.com](https://smith.langchain.com). |
| `LANGCHAIN_PROJECT` | Folders `04`, `05` | Name of your LangSmith project dashboard. |
| `ALPHA_API_KEY` | Folder `06` | Alpha Vantage stock market data tool. |
| `TAVILY_API_KEY` | Folder `17`, `Capstone` | Web search agent tool for research. |
| `CLOUDFLARE_ACCOUNT_ID` | `Capstone` | Cloudflare Workers AI account ID for image generation. |
| `CLOUDFLARE_API_TOKEN` | `Capstone` | Cloudflare API token with Workers AI access. |
| `DATABASE_URL` | Folder `16`, `Capstone` | Postgres URI (e.g. `postgresql://postgres:postgres@localhost:5442/blog_generator`). |

### 3. Database Setup (Docker)

To run the PostgreSQL persistence layer used in Module 16 and the Capstone Blog Agent:

```bash
docker compose up -d
```
*This starts a PostgreSQL 16 instance mapped to host port `5442` (matching `.env.example`).*

---

## ▶️ Running the Applications

### Jupyter Notebooks

Launch Jupyter and open any notebook in the numbered folders:

```bash
jupyter notebook
```

### Streamlit Web Apps

Run any of the interactive frontend applications using `streamlit run`:

```bash
# 1. Basic Streaming Chatbot (Module 02)
streamlit run 02_langgraph_streamlit_chatbot/streamlit_frontend.py

# 2. SQLite Persistent Chatbot (Module 03)
streamlit run 03_langgraph_streamlit_SQLite_DB_chatbot/streamlit_database_frontend.py

# 3. Tool-Calling Chatbot (Module 06)
streamlit run 06_langgraph_chatbot_with_tools/with_streamlit_frontend/streamlit_frontend.py

# 4. MCP-Enabled Chatbot (Module 09)
streamlit run 09_langgraph_streamlit_mcp_chatbot/streamlit_frontend_mcp.py

# 5. Interactive PDF RAG Chatbot (Module 11)
streamlit run 11_rag_chatbot/streamlit_rag_frontend.py

# 6. Capstone: Blog Writing Agent with Images (Project)
streamlit run project/blog-writing-agent/with_frontend/bwa_frontend.py
```

### MCP Server & Client

To run the Model Context Protocol terminal client:

```bash
# Terminal Client (Module 08)
python 08_langgraph_mcp_client/chatbot_mcp.py
```
*The client automatically manages the FastMCP arithmetic server (`07_langgraph_mcp_server/arithmetic_server.py`) over `stdio`.*

---

## 🗂️ Repository Directory Structure

```plaintext
agentic-ai-learning/
├── 01_langgraph_tutorials/                 # 12 foundational LangGraph notebooks
├── 02_langgraph_streamlit_chatbot/         # Streaming chatbot with Streamlit
├── 03_langgraph_streamlit_SQLite_DB_chatbot/# Checkpointer persistence with SQLite
├── 04_langSmith_tutorials/                 # Observability, tracing, and tags
├── 05_langgraph_chatbot_observability/     # SQLite chatbot instrumented with LangSmith
├── 06_langgraph_chatbot_with_tools/        # ToolNode, Alpha Vantage & DuckDuckGo
├── 07_langgraph_mcp_server/                # FastMCP arithmetic server
├── 08_langgraph_mcp_client/                # Local stdio & remote SSE MCP client
├── 09_langgraph_streamlit_mcp_chatbot/     # Streamlit UI powered by MCP tools
├── 10_rag_with_langgraph/                  # LangGraph RAG pipeline with FAISS
├── 11_rag_chatbot/                         # Streamlit PDF RAG chatbot
├── 12_human_in_the_loop_using_langgraph/   # HITL breakpoints and state inspection
├── 13_subgraph_in_langgraph/               # Nested graphs and shared state
├── 14_llm_memory/                          # Memory fundamentals overview
├── 15_llm_short_term_memory/               # STM: checkpointers, trimming, summarization
├── 16_llm_long_term_memory/               # LTM: namespaces, profiles, PostgresStore
├── 17_corrective_rag/                      # CRAG: retrieval grader, web fallback, query rewrite
├── 18_self_rag/                            # Self-RAG: reflection tokens, hallucination grading
├── project/
│   └── blog-writing-agent/                 # Capstone: Multi-agent blog generator
│       ├── 01_basic_blog_writing_agent.ipynb
│       ├── 02_bwa_improved_prompting.ipynb
│       ├── 03_bwa_including_research.ipynb
│       ├── 04_bwa_including_images.ipynb
│       ├── with_frontend/                  # Streamlit app (bwa_frontend.py, bwa_backend.py)
│       └── images/                         # Generated article images
├── NOTES.md                                # Comprehensive concepts & terminology guide
├── docker-compose.yml                      # PostgreSQL 16 container definition
├── requirements.txt                        # Base Python dependencies
└── .env.example                            # Template for API keys and configuration
```

---

## 🩺 Troubleshooting & FAQs

| Issue | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'fastmcp'` or `'faiss'` | Optional packages not installed in virtual environment. | Run the extended `pip install` command listed in [Installation](#extended-dependencies-for-advanced-modules). |
| `GoogleGenerativeAI: ResourceExhausted` | Reached Gemini API free-tier rate limits. | Wait 60 seconds before retrying or implement exponential backoff. |
| `PostgreSQL connection refused on localhost:5442` | Docker container is stopped or port is blocked. | Run `docker compose up -d` and check status with `docker ps`. Note: The blog agent will gracefully fallback to SQLite if PostgreSQL is unavailable. |
| `Remote MCP server timeout` | External third-party MCP endpoints are offline or slow. | Test with the local arithmetic server in `07_langgraph_mcp_server`. |
| State not saving between turns | Missing `thread_id` in configuration. | Ensure you pass `config={"configurable": {"thread_id": "<unique_id>"}}` to `app.invoke()` or `app.stream()`. |

---

## 🔗 Contact & Support

- **Developer:** Saksham Agrahari
- **Email:** [agrahari0899@gmail.com](mailto:agrahari0899@gmail.com)
- **GitHub:** [@saksham2882](https://github.com/saksham2882)
- **LinkedIn:** [@saksham-agrahari](https://www.linkedin.com/in/saksham-agrahari/)
- **Portfolio:** [saksham-agrahari.vercel.app](https://saksham-agrahari.vercel.app)

---

<p align="center">
  Made with ❤️ by <b>Saksham Agrahari</b> • Star ⭐ this repository if you find it helpful!
</p>
