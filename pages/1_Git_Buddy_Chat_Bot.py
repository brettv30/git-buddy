import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import random
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
    all_components = ComponentInitializer(Config())

    rag_chain, retriever_chain = all_components.initialize_components()

    api_handler = APIHandler(rag_chain, retriever_chain)

    # Generate a random 4-digit number with leading zeros
    api_handler.set_session_id(f"{random.randint(0, 9999):04}")

    return api_handler


api_handler = set_up_components()
ctx = get_script_run_ctx()

# Set unique session ID to store individual histories for each session
api_handler.set_session_id(ctx.session_id)

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
                    api_handler.set_user_query(st.session_state.messages[-1]["content"])
                    additional_sources = api_handler.find_additional_sources(
                        st.session_state.messages[-1]["content"]
                    )
                    st.write("Making OpenAI API Request...")
                    chat_response = api_handler.make_request_with_retry(
                        additional_sources
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
