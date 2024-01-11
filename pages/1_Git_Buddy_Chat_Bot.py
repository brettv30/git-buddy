import time
import streamlit as st
from utilities.utils import get_improved_answer, set_chat_messages, clear_memory

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
            with st.status("Accessing Tools...", expanded=True) as status:
                chat_response = get_improved_answer(
                    st.session_state.messages[-1]["content"]
                )
                status.update(
                    label="Look here for a play-by-play...",
                    state="complete",
                    expanded=False,
                )

            if "Error occurred" in chat_response:
                st.error(chat_response)
                set_chat_messages(chat_response)
                with st.status("Looking for issues...", expanded=True) as status:
                    st.write("Clearing Git Buddy's memory to free up token space...")
                    clear_memory()
                    st.write("Memory cleared...")
                    status.update(
                        label="Ready for more questions!",
                        state="complete",
                        expanded=False,
                    )
            else:
                st.write(chat_response)
                set_chat_messages(chat_response)
