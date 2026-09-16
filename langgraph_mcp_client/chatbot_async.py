from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import asyncio


# ====== load env ======
load_dotenv()


# ====== define google gemini model =======
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


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
    


# ========== Make tool list ===========
tools = [calculator]

# ========== Make the LLM tool-aware ============
llm_with_tools = llm.bind_tools(tools)


# ========== define state for chatbot ==============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# ========== build graph ===========
def build_graph():

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

    chatbot = build_graph()

    result = await chatbot.ainvoke({ "messages": [HumanMessage(content="Find the modulus of 268969 and 5785 and give answer like a cricket commentator.")]})

    print(result['messages'][-1].content[0].text)



if __name__ == '__main__':

    asyncio.run(main())


# NOTE: This code is asynchronous code because the library we are using to create the MCP
# client supports only asynchronous operations.