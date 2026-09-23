from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ======= Tools =========
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


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    Human-In-The-Loop:
    Before confirming the purchase, this tool will interrupt and wait for a human decision ("yes" / anything else).
    """

    # This pause the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} share of {symbol}? (Yes/No)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }
    

tools = [get_stock_price, purchase_stock]

llm_with_tools = llm.bind_tools(tools)


# ============ state ===========
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ========== chat_node function ==========
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ============ checkpointer ===========
memory = MemorySaver()


# ============ Graph =============
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=memory)


# ============ Simple usage example (CLI with HITL) ==============
if __name__ == "__main__":
    
    # Use a fixed thread_id so the conversation is persisted in memory
    thread_id = "demo-thread"

    while True:
        user_input = input("user: ")

        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})

        # Check for HITL interrupt from purchase_stock
        interrupts = result.get("__interrupt__", [])


        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")