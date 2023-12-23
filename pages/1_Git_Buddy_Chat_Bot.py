import streamlit as st
import re
import time
import tiktoken
import pinecone
import streamlit as st
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from utilities.utils import get_answer

# Start Streamlit app
st.set_page_config(page_title="Git Buddy")

st.title("Git Buddy")

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

# If last message is not from assistant, then respond
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # From Utils
            chat_response = get_answer(st.session_state.messages[-1]["content"])

            # Write the agent's response to the chat
            st.write(chat_response)

            # Set the message dictionary that will append to the messages list
            message = {
                "role": "assistant",
                "content": chat_response,
            }
            st.session_state.messages.append(message)
