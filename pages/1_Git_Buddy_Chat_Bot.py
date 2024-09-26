import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
import os
import sys

# Append the repo directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from agents.basic.agent import GitBuddyAgentManager  # noqa: E402
from utilities.utils import stream_data # noqa: E402

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

    agent = GitBuddyAgentManager.set_graph_agent()
    return agent


ctx = get_script_run_ctx()
git_buddy = set_up_components()

# Initialize the chat messages history
if "messages" not in st.session_state:
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

# If the last message is not from assistant, then respond
if st.session_state.messages[-1]["role"] != "assistant":
    user_message = st.session_state.messages[-1]["content"]
    if len(user_message) > 1000:
        assistant_message = (
            "Your question is too long. Please reword it with fewer words."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        with st.chat_message("assistant"):
            st.write(assistant_message)
    elif len(user_message) < 10:
        assistant_message = "Please ask a question with more words."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        with st.chat_message("assistant"):
            st.write(assistant_message)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_messages = git_buddy.invoke(
                    {"messages": [("user", user_message)]},
                    config={"configurable": {"thread_id": ctx.session_id}},
                )

                st.write(stream_data(response_messages["messages"][-1].content))

                # Add the full response to the chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_messages["messages"][-1].content,
                    }
                )
