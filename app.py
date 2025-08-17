import uuid
import streamlit as st
from agent_helper import Agent


# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = Agent()

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history_summary" not in st.session_state:
    st.session_state.chat_history_summary = "None"


st.title("Chatbot")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

# Handle new user input
if message := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    agent_response, agent_state = st.session_state.agent.invoke(message, thread_id=st.session_state.chat_id)

    st.session_state.chat_history_summary = agent_state["summary"]

    with st.chat_message("assistant"):
        st.markdown(agent_response)
    st.session_state.messages.append({"role": "assistant", "content": agent_response})

# Sidebar: system message input
with st.sidebar:

    st.header("Agent")

    st.subheader("Last Chat History Summary")
    with st.container(height=300): # Set the desired height in pixels
        st.markdown(st.session_state.chat_history_summary)

    st.subheader("Agent Graph")
    st.image("graph.png",caption="Chatbot built with LangGraph, featuring Tavily search integration and chat history summarization for memory", use_container_width =True)
