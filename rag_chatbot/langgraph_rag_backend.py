import os
import sqlite3
import requests
import tempfile

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv


load_dotenv()


# ========= define google gemini model ==========
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", streaming=True)


# ========= embeddings =========
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ========= PDF retriever store (per thread) =========
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Return a summary dict that can be surfaced in the UI.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass



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


    
# ======== RAG as Tool =========
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }



tools = [safe_search, get_stock_price, calculator, rag_tool]

# ========== Make the LLM tool-aware ============
llm_with_tools = llm.bind_tools(tools) if tools else llm 



# ========== define state for chatbot ==============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ========== define chat_node function =============
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call"
            "the `rag_tool` and include the thread_id"
            f"`{thread_id}`. You can also use the web search, stock price, and calculator tools when helpful."
            "If no document is available, ask the user to upload a PDF."
        )
    )    

    # take user query from state
    messages = [system_message, *state['messages']]

    # send query to llm
    response = llm_with_tools.invoke(messages, config=config)

    # store response in state
    return { 'messages': [response] }


tool_node = ToolNode(tools) if tools else None



# ========== create Checkpointer =============
connection = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


# ============ Graph ==========
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpointer)



# =========== Helper =============
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
