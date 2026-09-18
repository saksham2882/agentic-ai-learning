from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os
import sys
from pathlib import Path


load_dotenv()


# =========== Dedicated async loop for backend tasks ============
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)



# ========= define google gemini model ==========
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# ========= Tools =========

# ====== Pre-build Tools =======
search_tool = DuckDuckGoSearchRun(region="us-en")

# Handle DuckDuckGo search timeout safely
@tool
def safe_search(query: str) -> str:
    """Search the web. If search fails, report the error without retrying."""
    try:
        return search_tool.invoke(query)
    except Exception as e:
        return (
            "Web search is temporarily unavailable due to a timeout. "
            "Do not retry the search. Inform the user that search is unavailable."
        )


# ========= Custom Tool ===========
@tool
def get_stock_price(symbol: str, date: str = "") -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA'). 
    If date is provided (DD-MM-YYYY), fetch historical stock price on that day OR for that entire month.
    Otherwise, fetch the latest stock price.
    """

    ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")

    if not ALPHA_API_KEY:
        raise ValueError("ALPHA_API_KEY is missing from .env")
    
    if date:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY"
            f"&symbol={symbol}"
            f"&outputsize=full"
            f"&apikey={ALPHA_API_KEY}"
        )

        r = requests.get(url, timeout=15)
        data = r.json()

        stock_data = data.get("Time Series (Daily)", {})
        return stock_data.get(date, {"error": f"No data found for {date}"})

    else:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={ALPHA_API_KEY}"
        )
        r = requests.get(url, timeout=15)
        return r.json()



# ============= Path Config =============
SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "langgraph_mcp_server"
    / "arithmetic_server.py"
)

# =========== MCP Client  ============
client = MultiServerMCPClient(
    {
        # Local Arithmetic Server
        "arithmetic": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER_PATH)]
        },

        # Remote Expense Tracker Server
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        },

        # Remote weather server
        "weather": {
            "transport": "streamable_http",
            "url": "https://weatherapi.projecteol.ru/mcp/",
        },

        # Remote time and timezone server
        "world_clock": {
            "transport": "streamable_http",
            "url": "https://worldclock.pro/mcp",
        },
    }
)


# ========= Load MCP Tools =========
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []
    

mcp_tools = load_mcp_tools()

tools = [safe_search, get_stock_price, *mcp_tools]

# ========== Make the LLM tool-aware ============
llm_with_tools = llm.bind_tools(tools) if tools else llm 



# ========== define state for chatbot ==============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ========== define chat_node function =============
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    # take user query from state
    message = state['messages']

    # send query to llm
    response = await llm_with_tools.ainvoke(message)

    # store response in state
    return { 'messages': [response] }

tool_node = ToolNode(tools) if tools else None



# ========== create Checkpointer =============
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())


# ============ Graph ==========
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node: 
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)



# =========== Helper =============
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())
