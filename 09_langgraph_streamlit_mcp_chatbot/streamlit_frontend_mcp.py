import streamlit as st
from langgraph_mcp_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import queue


# ========================== UTILITY FUNCTIONS ========================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
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




# =========================== SESSION SETUP ============================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])



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
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        
        # convert message format to simple user, assistant
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({ 'role': role, 'content': msg.content })
        
        st.session_state['message_history'] = temp_messages



#  ========================== MAIN UI ==================================

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type here...')


if user_input:

    # Save and display the user's message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })
    with st.chat_message('user'):
        st.text(user_input)



    # ======= use streaming to show AI response as writing ========
    with st.chat_message('assistant'):

        # Status container to show current activity
        status = st.status("Thinking...", expanded=False)

        # Bridge between the backend async loop and Streamlit UI
        event_queue = queue.Queue()

        async def produce_stream():
            """Consume LangGraph's async stream on the backend event loop."""

            try:
                async for message_chunk, metadata in chatbot.astream(
                    { "messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    # Identify the current LangGraph node
                    node = metadata.get("langgraph_node", "")

                    # Detect tool calls from the AI message
                    tool_call_chunks = getattr(message_chunk, "tool_call_chunks", [])

                    for tool_call in tool_call_chunks:
                        tool_name = tool_call.get("name")

                        if tool_name:
                            event_queue.put(("tool_call", tool_name))

                    
                    # Show when tool execution returns a result
                    if node == "tools":
                        tool_name = getattr(message_chunk, "name", None)

                        if tool_name:
                            event_queue.put(("tool_result", tool_name))

                    # Extract assistant text
                    if isinstance(message_chunk, AIMessage):

                        # Handle text content blocks
                        content = message_chunk.content

                        if isinstance(content, str):
                            if content:
                                event_queue.put(("text", content))

                        elif isinstance(content, list):
                            for block in content:
                                if (isinstance(block, dict) and block.get("type") == "text"):
                                    text = block.get("text", "")

                                    if text:
                                        event_queue.put(("text", text))

            except Exception as exc:
                event_queue.put(("error", exc))
            
            finally:
                event_queue.put(("done", None))


        def stream_to_ui():
            """Read backend events and update the Streamlit UI."""

            while True:
                event, payload = event_queue.get()

                if event == "tool_call":
                    status.update(
                        label=f"Calling tool: {payload}",
                        state="running",
                    )
                elif event == "tool_result":
                    status.update(
                        label=f"Tool result received: {payload}",
                        state="running",
                    )
                elif event == "text":
                    yield payload
                elif event == "error":
                    raise payload
                elif event == "done":
                    break

        # Run the async producer on the backend's dedicated event loop
        task = submit_async_task(produce_stream())

        try:
            # Stream text into the chat UI
            ai_message = st.write_stream(stream_to_ui())

            # Wait for the producer coroutine to finish
            task.result()

            status.update(
                label="Response completed",
                state="complete",
                expanded=False,
            )

            # Save the completed assistant response
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message,
            })

        except Exception as exc:
            status.update(
                label="Something went wrong",
                state="error",
                expanded=True,
            )
            st.error(f"Error: {exc}")