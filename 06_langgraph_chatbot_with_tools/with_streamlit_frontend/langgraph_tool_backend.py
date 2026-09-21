from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver            # A checkpoint saver that stores checkpoints in a SQLite database.
import sqlite3                                                 # help to create SQLite Database
import requests

from dotenv import load_dotenv
import os 


# ====== load env ======
load_dotenv()


# ====== define google gemini model =======
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")



# ====== Tools =======
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


# ====== Custom Tools ======
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return { "error": "Division by zero is not allowed" }
            result = first_num / second_num
        else:
            return { "error": f"Unsupported operation '{operation}'" }
        
        return { "first_num": first_num, "second_num": second_num, "operation": operation, "result": result }
    except Exception as e:
        return { "error": str(e) }
    

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
            f"https://www.alphavantage.co/query"
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
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={ALPHA_API_KEY}"
        )

        r = requests.get(url, timeout=15)
        return r.json()



# ========== Make tool list ===========
tools = [get_stock_price, safe_search, calculator]

# ========== Make the LLM tool-aware ============
llm_with_tools = llm.bind_tools(tools)



# ========== define state for chatbot ==============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]        # reducer (add_messages in the list instead of update)


# ========== define chat_node function =============
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    # take user query from state
    message = state['messages']

    # send query to llm
    response = llm_with_tools.invoke(message)

    # store response in state
    return { 'messages': [response] }

tool_node = ToolNode(tools)     # Executes tool calls



# ========== create SQLite Database =============
# NOTE: SQLite Database work on single thread (means in which thread it created, use on same one), so we explicitly make the check_same_thread=False to not check for same thread and same DB.
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# ========== define checkPointer =============
checkpointer = SqliteSaver(conn=connection)



# ========== create a graph =============
graph = StateGraph(ChatState)

# add nodes to the graph
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# add edges to the graph
graph.add_edge(START, 'chat_node')

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

# Again call the LLM after tool execute
graph.add_edge("tools", "chat_node")



# =========== compile the graph ==============
chatbot = graph.compile(checkpointer=checkpointer)


# ========== Helper =============
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads) 