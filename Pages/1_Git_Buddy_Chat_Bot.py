import streamlit as st
import langchain
from langchain.callbacks import StreamlitCallbackHandler
from utils_agent import *

# Start Streamlit app
st.set_page_config(page_title="Git Buddy")

st.title("Git Buddy: The chatbot that answers all things Git, GitHub, and TortoiseGit")

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Git, GitHub, or TortoiseGit!",
        }
    ]

# Prompt for user input and save to chat history
if prompt := st.chat_input("What is the difference between Git and GitHub?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant and we have a key, then respond
if (
    st.session_state.messages[-1]["role"]
    != "assistant"
    # and st.session_state.openai_api_key
):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            langchain.debug = True
            chat_response = run_agent(
                st.session_state.messages[-1]["content"], st_callback
            )

            if chat_response.startswith("I now know the answer to the question."):
                chat_response = chat_response.removeprefix(
                    "I now know the answer to the question."
                )

            # Write the agent's response to the chat
            st.write(chat_response)

            # Set the message dictionary that will append to the messages list
            message = {
                "role": "assistant",
                "content": chat_response,
            }
            st.session_state.messages.append(message)

    langchain.debug = False
