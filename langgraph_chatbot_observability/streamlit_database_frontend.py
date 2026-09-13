import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


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

    # first add the user message to message_history
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })
    with st.chat_message('user'):
        st.text(user_input)



    # ======= use streaming to show AI response as writing ========
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({ 'role': 'assistant', 'content': ai_message })