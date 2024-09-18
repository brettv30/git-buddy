import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import random
import os
from agents.llm_compiler.agent import LLMCompilerAgent  # Import the new agent class

# Start Streamlit app
st.set_page_config(page_title="Git Buddy")

st.title("Git Buddy")

# Initialize chatbot components
@st.cache_resource
def set_up_components():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = "git-buddy"

    # Create an instance of LLMCompilerAgent
    git_buddy = LLMCompilerAgent()

    return git_buddy

git_buddy = set_up_components()
ctx = get_script_run_ctx()

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Git, GitHub, or TortoiseGit!",
        }
    ]

# Prompt for user input and save to chat history
if prompt := st.chat_input(
    "What is the difference between Git and GitHub?", key="prompt"
):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, then respond
if st.session_state.messages[-1]["role"] != "assistant":
    if len(st.session_state.messages[-1]["content"]) > 1000:
        st.write(
            "Your question is too long. Please ask your question again with fewer words."
        )
        message = {
            "role": "assistant",
            "content": "Your question is too long. Please reword it with fewer words.",
        }
        st.session_state.messages.append(message)
    elif len(st.session_state.messages[-1]["content"]) < 10:
        st.write("Please ask a question with more words.")
        message = {
            "role": "assistant",
            "content": "Please ask a question with more words.",
        }
        st.session_state.messages.append(message)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the stream method of LLMCompilerAgent
                response_stream = git_buddy.stream(
                    st.session_state.messages[-1]["content"], ctx.session_id
                )
                
                # Display the streaming response
                response_container = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if "messages" in chunk and chunk["messages"]:
                        content = chunk["messages"][-1].content
                        full_response += content
                        response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)

                # Add the full response to the chat history
                message = {
                    "role": "assistant",
                    "content": full_response,
                }
                st.session_state.messages.append(message)

    # st.rerun()