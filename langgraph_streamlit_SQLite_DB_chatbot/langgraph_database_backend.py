from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver        # A checkpoint saver that stores checkpoints in a SQLite database.
import sqlite3                                             # help to create SQLite Database


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


# create SQLite Database
# NOTE: SQLite Database work on single thread (means in which thread it created, use on same one), so we explicitly make the check_same_thread=False to not check for same thread and same DB.
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)


# define checkPointer
checkpointer = SqliteSaver(conn=connection)


# create a graph
graph = StateGraph(ChatState)

# add nodes to the graph
graph.add_node('chat_node', chat_node)

# add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile the graph
chatbot = graph.compile(checkpointer=checkpointer)


# return all checkpointer in db
# checkpointer.list(None)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads) 




# ========= Test =============
# CONFIG={ 'configurable': { 'thread_id': 'thread-1'}}

# response = chatbot.invoke(
#     {'messages': [HumanMessage(content='what is my name')]},
#     config=CONFIG
# )

# print(response)