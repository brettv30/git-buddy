import streamlit as st

st.set_page_config(page_title="Additional Resources")

tab1, tab2, tab3 = st.tabs(["Documentation", "Model Architecture", "Starter Questions"])

with tab1:
    st.header(
        "Git Buddy's Pinecone vector database was populated with information from the following documentation:"
    )
    st.markdown(
        """
        [Pro Git](https://git-scm.com/book/en/v2) by Scott Chacon and Ben Straub

        [GitHub Docs](https://docs.github.com/en) The Official GitHub Documentation

        [TortoiseGit Manual](https://tortoisegit.org/docs/tortoisegit/) by Lubbe Onken, Simon Large, Frank LI, and Sven Strickroth

        [TortoiseGitMerge Manual](https://tortoisegit.org/docs/tortoisegitmerge/) by Stefan Kung, Lubbe Onken, Simon Large, and Sven Strickroth
    """
    )


with tab2:
    st.header("Git Buddy uses the following architecture:")
    st.image(
        "images\High-Level-RAG-Diagram.png", caption="High Level Git-Buddy Architecture"
    )

with tab3:
    st.header("Git, GitHub, and TortoiseGit Starter Questions:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        - What is Git?
        - What is GitHub?
        - What is TortoiseGit?
        - Why is GitHub useful?
        - What is the difference between Git and GitHub?
        - How can I set up a GitHub repository? 
        - What are common Git commands and what are they used for?         
        - What is an Issue? 
        - What is a Tag? 
        - What is a release? 
        - How do I get a remote repository onto my local computer?
        """
        )
    with col2:
        st.markdown(
            """
        - Can you explain branching to me please?
        - What are GitHub Actions used for? 
        - What is a Pull Request? 
        - How are projects and milestones tracked in GitHub?
        - Can you explain the popular branching strategies to me? 
        - What is the difference between a Commit and a Push?
        - Why should you always pull before you push to a remote repository?
        - What are commit messages and how should I write one?
        - How should I structure my issue description?
        - How do I merge two branches together?
        - How do I commit updates using TortoiseGit?
        """
        )
