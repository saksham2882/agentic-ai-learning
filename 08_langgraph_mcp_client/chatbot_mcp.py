from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


# ====== load env ======
load_dotenv()


# ====== define google gemini model =======
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# ============= Path Config =============
SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "langgraph_mcp_server"
    / "arithmetic_server.py"
)

# ====== MCP Client for local FastMCP server ======
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



# ========== define state for chatbot ==============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# ========== build graph ===========
async def build_graph():

    tools = await client.get_tools()
    
    print("\nAvailable MCP tools:")
    for t in tools:
        print(f"- {t.name}: {t.description}")

    llm_with_tools = llm.bind_tools(tools)

    # ========== define chat_node function =============
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""

        # take user query from state
        message = state['messages']

        # send query to llm
        response = await llm_with_tools.ainvoke(message)

        # store response in state
        return { 'messages': [response] }

    tool_node = ToolNode(tools)

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
    chatbot = graph.compile()
    
    return chatbot


# ========== main ===========
async def main():

    chatbot = await build_graph()

    # human_message = "Find the modulus of 268969 and 5785 and give answer like a cricket commentator."
    # human_message = "Add an expense - Rs.450 for Linkedin Learning Education Course on 16th Sep 2026"
    # human_message = "What is the current weather in New Delhi, India?"
    human_message = "What is the current time in New Delhi, India?"

    result = await chatbot.ainvoke({ "messages": [HumanMessage(content=human_message)]})

    # last_msg = result['messages'][-1]
    # if isinstance(last_msg.content, list):
    #     for part in last_msg.content:
    #         if isinstance(part, dict) and 'text' in part:
    #             print(part['text'])
    #         elif hasattr(part, 'text'):
    #             print(part.text)
    #         else:
    #             print(part)
    # else:
    #     print(last_msg.content)

    print(result['messages'][-1].content)



if __name__ == '__main__':

    asyncio.run(main())


# NOTE: This code is asynchronous code because the library we are using to create the MCP
# client supports only asynchronous operations.