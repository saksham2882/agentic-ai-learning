import streamlit as st
from langgraph_rag_backend import chatbot, ingest_pdf, retrieve_all_threads, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
import uuid


# ========================== UTILITY FUNCTIONS ========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            'configurable': { 'thread_id': thread_id }
        }
    )
    # Check whether the thread has any saved state
    if not state or not state.values:
        return []

    # Safely return messages; empty list if key is missing
    return state.values.get('messages', [])


def get_clean_conversation(messages):
    """Return only user messages and final AI responses."""
    clean_messages = []

    for msg in messages:
        # Keep user messages
        if isinstance(msg, HumanMessage):
            clean_messages.append({
                "role": "user",
                "content": msg.content,
            })

        # Keep only final AI responses, not tool calls
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                continue

            # Ignore empty or non-text AI messages
            if not isinstance(msg.content, str) or not msg.content.strip():
                continue

            clean_messages.append({
                "role": "assistant",
                "content": msg.content,
            })

    return clean_messages



# =========================== SESSION SETUP ============================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'ingested_docs' not in st.session_state:
    st.session_state["ingested_docs"] = {}


add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None



# =========================== CONFIGURATION ============================
# CONFIG={ 'configurable': { 'thread_id': st.session_state['thread_id']}}

# change to organized traces in thread for better observability
CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn",
}


# =========================== SIDEBAR UI ===============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}`"
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")


uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="PDF indexed", state="complete", expanded=False)


st.sidebar.subheader("Past Conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id


#  ========================== MAIN UI ==================================
st.title("Multi Utility Chatbot")

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Ask about your document....')


if user_input:

    # Save and display the user's message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # ======= use streaming to show AI response as writing ========
    with st.chat_message('assistant'):

        # Display the current AI activity
        status = st.status("Thinking...", expanded=True)

        activity = {
            "last_status": "Thinking...",
            "current_tool": None,
            "tool_used": False,
            "tool_result_received": False,
        }

        # Friendly names for the available tools
        TOOL_LABELS = {
            "safe_search": "Web Search",
            "calculator": "Calculator",
            "get_stock_price": "Stock Price Lookup",
            "rag_tool": "PDF Retrieval",
        }

        def update_status(label, state="running", expanded=True):
            """Update status only when the label changes."""
            if activity["last_status"] != label:
                status.update(
                    label=label,
                    state=state,
                    expanded=expanded,
                )
                activity["last_status"] = label

        def ai_only_stream():
            """Stream AI text and track LangGraph tool activity."""

            for message_chunk, metadata in chatbot.stream(
                { "messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Identify the active LangGraph node
                node = metadata.get("langgraph_node", "")

                # 1. Detect tool calls before their results arrive
                if isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                    detected_tools = []

                    # Tool calls arriving as streaming chunks
                    tool_call_chunks = getattr(message_chunk, "tool_call_chunks", []) or []

                    for call in tool_call_chunks:
                        if isinstance(call, dict):
                            tool_name = call.get("name")
                        else:
                            tool_name = getattr(call, "name", None)

                        if tool_name and tool_name not in detected_tools:
                            detected_tools.append(tool_name)

                    # Also handle complete tool calls
                    complete_tool_calls = getattr(message_chunk, "tool_calls", []) or []
              
                    for call in complete_tool_calls:
                        if isinstance(call, dict):
                            tool_name = call.get("name")
                        else:
                            tool_name = getattr(call, "name", None)

                        if tool_name and tool_name not in detected_tools:
                            detected_tools.append(tool_name)

                    for tool_name in detected_tools:
                        activity["current_tool"] = tool_name
                        activity["tool_used"] = True
                        activity["tool_result_received"] = False

                        friendly_name = TOOL_LABELS.get(tool_name, tool_name)

                        update_status(
                            f"Calling tool: {friendly_name}...",
                            state="running",
                            expanded=True,
                        )

                # 2. Detect tool results
                if isinstance(message_chunk, ToolMessage):
                    tool_name = (getattr(message_chunk, "name", None) or activity["current_tool"] or "Tool")

                    friendly_name = TOOL_LABELS.get(tool_name, tool_name)

                    activity["tool_result_received"] = True

                    update_status(
                        f"Received result from: {friendly_name}",
                        state="running",
                        expanded=True,
                    )

                # 3. Extract and stream only readable AI text
                if not isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                    continue

                content = message_chunk.content
                text_parts = []

                if isinstance(content, str):
                    if content:
                        text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            text = block.get("text", "")
                            if text:
                                text_parts.append(text)

                for text_part in text_parts:
                    # Show that the model is generating its answer
                    if activity["tool_result_received"]:
                        update_status(
                            "Writing answer using tool results...",
                            state="running",
                            expanded=True,
                        )
                    elif not activity["tool_used"]:
                        update_status(
                            "Generating response...",
                            state="running",
                            expanded=True,
                        )

                    yield text_part  

        # Stream the answer and handle errors
        try:
            ai_message = st.write_stream(ai_only_stream())

        except Exception as exc:
            status.update(
                label="Something went wrong",
                state="error",
                expanded=True,
            )
            st.error(f"Error: {exc}")
            ai_message = ""
        else:
            status.update(
                label="Response completed",
                state="complete",
                expanded=False,
            ) 

        # Save only a successfully generated, non-empty response
        if ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message,
            }) 

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)

    # Load only clean, user-facing conversation messages
    st.session_state["message_history"] = get_clean_conversation(messages)

    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})

    st.rerun()