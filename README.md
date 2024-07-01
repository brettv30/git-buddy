# Git Buddy 🤖

[![Pinecone](https://img.shields.io/badge/-Pinecone-0000ff?style=flat-square&logo=pinecone&logoColor=white)](https://www.pinecone.io)
[![Cohere](https://img.shields.io/badge/Cohere-80006b?style=flat-square&logo=cohere&logoColor=white)](https://cohere.com/)
[![OpenAI](https://img.shields.io/badge/-OpenAI-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
[![Langchain](https://img.shields.io/badge/-Langchain-gray?style=flat-square)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://git-buddy.streamlit.app/)

## About Git Buddy 📖

Think of Git Buddy as a friendly, understanding, and helpful version control expert who never runs out of patience. Git Buddy offers streamlined assistance with learning about Git, GitHub, and TortoiseGit, designed for users at all levels of expertise. It combines the ease of a chatbot interface with advanced Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) technology to enhance your version control system learning experience. By coupling GPT-3.5-Turbo with few-shot prompting and RAG inside a streamlit application Git Buddy is prepared to answer any and all questions related to Git, GitHub, and TortoiseGit. 

https://github.com/brettv30/git-buddy/assets/50777864/cbf060dd-9c65-42f6-ae48-6bcf37824440

## Key Features 🔑

- **TortoiseGit Advisor**: Helps you utilize TortoiseGit in your projects with ease.
- **GitHub Navigator**: Guides you through GitHub, from repository management to pull requests.
- **Git Teacher**: Offers detailed explanations, examples, and tips on Git commands and best practices.

## Technical Specifications 💡

- **Vector Database Population**: [GitHub Docs](https://docs.github.com/en) & [Progit](https://git-scm.com/book/en/v2) were used as the starting places for a recursive URL search that extracted all important information from each webpage's HTML. [TortoiseGit](https://tortoisegit.org/docs/tortoisegit/) & [TortoiseGitMerge](https://tortoisegit.org/docs/tortoisegitmerge/) were copied as PDFs and split apart during document chunking. 
- **Document Chunking in RAG**: For the RAG system, document chunks are set to 400 tokens, with a 50-token overlap across documents, optimizing the balance between context and detail.
- **Document Reranking**: The Cohere `Rerank` model is used to rerank the top 100 documents by finding the top five documents most semantically relevant to the query.
- **Embeddings Model**: Utilizes the `text-embedding-ada-002` model from OpenAI for generating document and query embeddings while processing text data.
- **Prompting Methodology**: Adopts few-shot prompting, conversational memory, and additional document context for robust, efficient, and accurate responses.
- **Conversational Memory**: Git Buddy retains knowledge of the seven most recent messages, ensuring relevant and contextual interactions while excluding older chat history.
- **Temperature Setting**: `GPT-3.5-Turbo` operates at a temperature setting of 0.5, balancing creativity and coherence in responses.

## Additional Resources 🔍

Explore [Additional Resources](https://git-buddy.streamlit.app/Additional_Resources) in the app for details about the documentation Git Buddy accesses, its underlying model architecture, and helpful starter questions.

## Contributing ✏️

All contributions are welcome! Simply open up an issue and create a pull request. If you are introducing new features, please provide a detailed description of the specific use case you are addressing and set up instructions to test.

## License 📝

This project is under the MIT License. See [LICENSE](https://github.com/brettv30/git-buddy/blob/main/LICENSE) for more details.
