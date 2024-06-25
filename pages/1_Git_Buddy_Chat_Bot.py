import streamlit as st
from utilities.utils import (
    Config,
    ComponentInitializer,
    APIHandler,
    DocumentParser,
    GitBuddyChatBot,
    PromptParser,
)

# Start Streamlit app
st.set_page_config(page_title="Git Buddy")

st.title("Git Buddy")


# Initialize chatbot components
@st.cache_resource
def set_up_components():
    config = Config()
    all_components = ComponentInitializer(config)

    (prompt, qa_llm, search, memory, retriever) = all_components.initialize_components()
    prompt_parser = PromptParser(config, memory, prompt)

    api_handler = APIHandler(config, prompt_parser, prompt, memory, qa_llm)
    doc_parser = DocumentParser(search)
    git_buddy = GitBuddyChatBot(config, api_handler, retriever, doc_parser)

    return prompt_parser, git_buddy


prompt_parser, git_buddy = set_up_components()

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
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with st.status("Accessing Tools...", expanded=True) as status:
                chat_response = git_buddy.get_improved_answer(
                    st.session_state.messages[-1]["content"]
                )
                status.update(
                    label="Look here for a play-by-play...",
                    state="complete",
                    expanded=False,
                )

            if type(chat_response) is not str:
                st.error(chat_response)
                git_buddy.set_chat_messages(chat_response)
                with st.status("Looking for issues...", expanded=True) as status:
                    st.write("Clearing Git Buddy's memory to free up token space...")
                    prompt_parser.clear_memory()
                    st.write("Memory cleared...")
                    status.update(
                        label="Ready for more questions!",
                        state="complete",
                        expanded=False,
                    )
            else:
                st.write(chat_response)
                git_buddy.set_chat_messages(chat_response)
    st.rerun()
