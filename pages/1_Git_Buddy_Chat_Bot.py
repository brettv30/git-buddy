import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import random
import os
from utilities.utils import (
    Config,
    ComponentInitializer,
    APIHandler,
)

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
    all_components = ComponentInitializer(Config())

    rag_chain, retriever_chain = all_components.initialize_components()

    api_handler = APIHandler(rag_chain, retriever_chain)

    return api_handler


api_handler = set_up_components()
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
    st.chat_input("", disabled=True)
    # Stop someone from being silly
    if len(st.session_state.messages[-1]["content"]) > 1000:
        st.write(
            "Your question is too long. Please ask your question again with less words."
        )
        message = {
            "role": "assistant",
            "content": "Your question is too long. Please reword it with less words.",
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
                with st.status("Accessing Tools...", expanded=True) as status:
                    additional_sources = api_handler.find_additional_sources(
                        st.session_state.messages[-1]["content"]
                    )
                    st.write("Making OpenAI API Request...")
                    chat_response = api_handler.make_request_with_retry(
                        st.session_state.messages[-1]["content"],
                        additional_sources, 
                        ctx.session_id
                    )
                    status.update(
                        label="Look here for a play-by-play...",
                        state="complete",
                        expanded=False,
                    )

                    if type(chat_response) is not str:
                        st.error(chat_response)
                    else:
                        st.write(chat_response)

                message = {
                    "role": "assistant",
                    "content": chat_response,
                }
                st.session_state.messages.append(message)

    st.rerun()
