import streamlit as st

# Start Streamlit App
st.set_page_config(page_title="Welcome")

st.title("Welcome to Git Buddy!")

col1, col2, col3 = st.columns([1.25, 0.75, 3.5])

with col1:
    st.caption("Author: Brett Vogelsang")

with col2:
    st.link_button("LinkedIn", "https://www.linkedin.com/in/brettvogelsang/")

with col3:
    st.link_button("GitHub", "https://github.com/brettv30")

st.markdown(
    """
[Git Buddy](https://git-buddy.streamlit.app/Git_Buddy_Chat_Bot) originated as an idea to utilize a Large Language Model (LLM) as a chatbot tool to assist with teaching others, specifically beginners, about Git, GitHub, and TortoiseGit. Git Buddy's main purpose is to provide a comfortable environment where people starting to learn version control are safe to ask any question that comes to mind. If you're not sure where to start or what to ask, please see `Additional Resources - Starter Questions` for some exmaples.

Git Buddy utilizes the following tools:

- [Langchain](https://www.langchain.com) to orchestrate
- [Pinecone](https://www.pinecone.io) to store document embeddings
- [OpenAI](https://openai.com) to supply the LLM (GPT-3.5-Turbo)
- [Streamlit](https://streamlit.io) to deliver the application

Git Buddy follows a Retrieval Augmented Generation (RAG) framework which allows for additional context to be provided to the LLM before passing the entire prompt to the model. The documents stored for retrieval can be found in `Additional Resources - Documentation`. RAG is essential for enhancing LLMs in app development. It supplements LLMs with external data sources, helping arrive at more relevant responses by reducing errors or hallucinations. RAG determines what info is relevant to the user's query through semantic search, which searches data by meaning (rather than just looking for literal matches of search terms). RAG is particularly effective for LLM apps that need to access domain-specific or proprietary data. 

Additionally, once similar documents are retrieved from Pinecone they are passed through a web search via [DuckDuckGo](https://python.langchain.com/docs/integrations/tools/ddg) to provide additional website sources in the LLM response. Empowering users not only with new knowledge via the LLM response, but also with actual website links that contain valuable information related to the user's query. This step further reduces errors and hallucinations related to incorrect and inaccurate website links in the LLM response.
"""
)
