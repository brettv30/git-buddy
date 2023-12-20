import streamlit as st

st.set_page_config(page_title="Documentation")

st.header(
    "Git Buddy has accesss to the following documentation which are stored as vector embeddings in a Pinecone Vector databse:"
)
st.markdown(
    """
    [Pro Git](https://git-scm.com/book/en/v2) by Scott Chacon and Ben Straub

    [GitHub Docs](https://docs.github.com/en) The Official GitHub Documentation

    [TortoiseGit Manual](https://tortoisegit.org/docs/tortoisegit/) by Lubbe Onken, Simon Large, Frank LI, and Sven Strickroth

    [TortoiseGitMerge Manual](https://tortoisegit.org/docs/tortoisegitmerge/) by Stefan Kung, Lubbe Onken, Simon Large, and Sven Strickroth
"""
)
