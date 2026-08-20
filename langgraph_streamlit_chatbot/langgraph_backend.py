from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver       # used to store in RAM


# load env
load_dotenv()

# define google gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# define state for chatbot
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]    # reducer (add_messages in the list instead of update)


# define chat_node function
def chat_node(state: ChatState):

    # take user query from state
    message = state['messages']

    # send query to llm
    response = llm.invoke(message)

    # store response in state
    return { 'messages': [response] }


# define check pointer for memorySaver
checkpointer = InMemorySaver()


# create a graph
graph = StateGraph(ChatState)

# add nodes to the graph
graph.add_node('chat_node', chat_node)

# add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile the graph
chatbot = graph.compile(checkpointer=checkpointer)



# ======= streaming ===========
# stream = chatbot.stream(                                                            # stream gives a Generator Object
#     {'messages': [HumanMessage(content="What is the recipe to make pizza")]}, 
#     config={ 'configurable': { 'thread_id': 'thread_1'}},
#     stream_mode='messages'
# )

# print(type(stream))


# for message_chunk, metadata in chatbot.stream(                                      # here we iterate over the Generator
#     {'messages': [HumanMessage(content="What is the recipe to make pizza")]}, 
#     config={ 'configurable': { 'thread_id': 'thread_1'}},
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)