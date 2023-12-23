# Welcome to Git Buddy 🌟

Git Buddy offers streamlined assistance with Git, GitHub, and TortoiseGit, designed for users at all levels of expertise. It combines the ease of a chatbot interface with advanced Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) technology to enhance your version control system experience.

## About Git Buddy 🤖

Git Buddy is an advanced chatbot built using the OpenAI API (GPT-3.5-Turbo). It simplifies interactions with Git, GitHub, and TortoiseGit, offering an intuitive approach to accessing necessary information and bypassing extensive documentation.

## Getting Started 🚀

Begin your journey with Git Buddy by visiting our [Streamlit App](https://git-buddy.streamlit.app) for an interactive, user-friendly experience.

### Quick Demo Video 🎥

https://github.com/brettv30/git-buddy/assets/50777864/cbf060dd-9c65-42f6-ae48-6bcf37824440

## Key Features 🔑

- **TortoiseGit Advisor**: Helps you utilize TortoiseGit in your projects with ease.
- **GitHub Navigator**: Guides you through GitHub, from repository management to pull requests.
- **Learning Assistant**: Offers detailed explanations, examples, and tips on Git commands and best practices.

## Tools and Technologies 🛠️

- **Langchain**: Orchestrates the backend processes of Git Buddy.
- **Pinecone**: Manages storage and retrieval of document embeddings.
- **OpenAI's GPT-3.5-Turbo**: Powers the LLM at the core of Git Buddy.
- **Streamlit**: Provides a sleek, user-friendly interface for Git Buddy.
- **DuckDuckGo**: Retreives working URLs included in Git Buddy's responses.

## Technical Specifications 💡

- **Prompting Methodology**: Adopts a one-shot prompting approach for efficient and accurate responses.
- **Embeddings Model**: Utilizes the `text-embedding-ada-002` model from OpenAI for processing and understanding text data.
- **Temperature Setting**: `GPT-3.5-Turbo` operates at a temperature setting of 0.5, balancing creativity and coherence in responses.
- **Conversational Memory**: Git Buddy retains knowledge of the four most recent messages, ensuring relevant and contextual interactions while excluding older chat history.
- **Document Chunking in RAG**: For the RAG system, document chunks are set to 400 tokens, with a 50-token overlap across documents, optimizing the balance between context and detail.

## Additional Resources 🔍

Explore [Additional Resources](https://git-buddy.streamlit.app/Additional_Resources) in the app for comprehensive details about Git Buddy's underlying documentation, architecture, and starter questions.

## License 📝

This project is under the MIT License. See [LICENSE](https://opensource.org/licenses/MIT) for more details.
