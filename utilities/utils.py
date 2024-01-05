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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# Set environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Set Const Variables
MODEL_REQUEST_LIMIT_PER_MINUTE = 500
EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_NAME = "git-buddy-index"
MODEL_NAME = "gpt-3.5-turbo"
RETRIEVED_DOCUMENTS = (
    5  # This one can vary while we test out different retrieval methods
)
PROMPT_TEMPLATE = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
You should only ever answer questions about Git, GitHub, or TortoiseGit. Never answer any other questions even if you think you know the correct answer. 
If possible, please provide example code to help the beginner learn Git commands. Never use the sources from the context in an answer, only use the sources from url_sources.

If a question is ambiguous please refer to the conversation history to see if that helps in answering the question at the end:
{chat_history}

Use the following pieces of context to answer the question at the end: 
{context}

If there are links in the following sources then you MUST link all of the following sources at the end of your answer to the question. You can just keep the entire link in the output, no need to hyperlink with a different name. Do NOT change the links.
{url_sources}

Use the following format:

Question: What is Git?
Answer: Git is a distributed version control system that allows multiple people to collaborate on a project. It tracks changes made to files and allows users to easily manage and merge those changes. Git is known for its speed, scalability, and rich command set. It provides both high-level operations and full access to internals. Git is commonly used in software development to manage source code, but it can also be used for any type of file-based project.
Additional Sources: Here's some additional Git soures to get started! 
    - [Pro Git Book](https://git-scm.com/book/en/v2) 
    - [Git Introduction Videos](https://git-scm.com/videos)
    - [External Git Links](https://git-scm.com/doc/ext)

Question: What should I think about when considering what to put into a github repository?
Answer: When considering what to put into a GitHub repository, you should think about the purpose and scope of your project. Here are some factors to consider:

Project files: Include all the necessary files for your project, such as source code, documentation, configuration files, and any other assets required for the project to run.

README file: It's a good practice to include a README file that provides an overview of your project, instructions for installation and usage, and any other relevant information.

License: Decide on the license for your project and include a license file. This helps clarify how others can use and contribute to your project.

Version control: If you are using Git, you should include the Git repository itself in your GitHub repository. This allows others to easily access the entire history of your project and contribute changes.

Ignore files: Use a .gitignore file to specify which files and directories should be ignored by Git. This helps avoid committing unnecessary files, such as build artifacts or sensitive information.

Collaboration: If you plan to collaborate with others, consider including a CONTRIBUTING file that outlines guidelines for contributing to your project. This can include information on how to report issues, submit pull requests, and follow coding standards.

Remember to regularly update your repository as your project progresses and make sure to keep sensitive information, such as API keys or passwords, out of your repository.
Additional Sources:
    - [GitHub Docs - Creating a Repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)
    - [Pro Git Book - Chapter 1: Getting Started](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control)

Question: How should I think about a github branch and its use cases?
Answer: A GitHub branch is a separate line of development within a repository. It allows you to work on different features or fixes without affecting the main codebase. Branches are commonly used to isolate changes and collaborate on specific tasks.

Here are some common use cases for GitHub branches:

Feature development: You can create a branch to work on a new feature or enhancement for your project. This allows you to develop and test the feature separately without impacting the main codebase. Once the feature is complete, you can merge the branch back into the main branch.
Bug fixes: If you discover a bug in your code, you can create a branch to fix the issue. This allows you to isolate the changes related to the bug fix and test them independently before merging them back into the main branch.
Experimentation: Branches can also be used for experimentation and trying out new ideas. You can create a branch to explore different approaches or implement experimental features without affecting the stability of the main codebase.
Collaboration: Branches are useful for collaboration among team members. Each team member can work on their own branch and make changes without conflicting with others. This allows for parallel development and easier code review before merging the changes.
By using branches, you can organize your development efforts and keep your main branch clean and stable. It's important to follow best practices such as creating descriptive branch names and regularly merging changes from the main branch into your feature branches to keep them up to date.
Additional Sources:
    - [Pro Git Book - Chapter 3: Git Branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
    - [GitHub Docs - About Branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches)

Begin!

Question: {human_input}
Answer:
Additional Sources:"""

# Set the encodings to ensure prompt sizing down below
enc = tiktoken.get_encoding("cl100k_base")


# Initialize Pinecone and LangChain components
def initialize_components():
    """Initialize Pinecone and LangChain components."""
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    index = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    retriever = index.as_retriever(
        search_type="mmr", search_kwargs={"k": RETRIEVED_DOCUMENTS}
    )
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.5)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="human_input",
        k=4,
    )
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "human_input", "url_sources"],
        template=PROMPT_TEMPLATE,
    )
    qa_llm = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    search = DuckDuckGoSearchResults()

    relevant_filter = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=relevant_filter, base_retriever=retriever
    )

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevant_filter]
    )

    pipeline_compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    return (
        prompt,
        index,
        qa_llm,
        search,
        memory,
        compression_retriever,
        pipeline_compression_retriever,
    )


# Initialize chatbot components
(
    prompt,
    index,
    qa_llm,
    search,
    memory,
    retriever,
    pipe_retriever,
) = initialize_components()


def get_similar_docs(
    index, query: str, k: int = RETRIEVED_DOCUMENTS, score: bool = False
) -> list:
    """Retrieve similar documents from the index based on the given query."""
    return (
        index.similarity_search_with_score(query, k=k)
        if score
        else index.similarity_search(query, k=k)
    )


def get_relevant_docs(retriever, query):
    """Retrieve relevant documents from the index using the Contextual Compression Retriever"""
    return retriever.get_relevant_documents(query)


def pipe_get_relevant_docs(retriever, query):
    """Retrieve relevant documents from the index using the Pipeline Contextual Compression Retriever"""
    return retriever.get_relevant_documents(query)


print("-------------------- Similar Docs with Similarity Search")
print(get_similar_docs(index, "What is branching in Git?"))
print("-------------------- Relevant Docs with Contextual Compression")
print(get_relevant_docs(retriever, "What is branching in Git?"))
print("-------------------- Relevant Docs with Pipeline Contextual Compression")
print(pipe_get_relevant_docs(pipe_retriever, "What is branching in Git?"))


def get_sources(docs: str) -> list:
    """Extract the 'source' from each document's metadata."""
    return [doc.metadata["source"] for doc in docs]


def get_search_query(sources: list) -> list:
    """Generate search queries from the list of sources."""
    pattern = r"\\(.*?)\."
    searches = [re.findall(pattern, source) for source in sources]
    # Flatten list and remove duplicates
    return list({element for sublist in searches for element in sublist})


def parse_urls(search_results: str) -> list:
    """Extract URLs from the search results."""
    pattern = r"https://[^\]]+"
    return re.findall(pattern, search_results)


def remove_specific_string_from_list(nested_list: list, string_to_remove: str) -> list:
    """
    Removes a specific string from all elements of a nested list.

    :param nested_list: List of lists containing strings.
    :param string_to_remove: String to be removed from each element.
    :return: A new nested list with the specific string removed from each element.
    """
    return [
        [element.replace(string_to_remove, "") for element in sublist]
        for sublist in nested_list
    ]


def remove_specific_element_from_list(
    nested_list: list, element_to_remove: str
) -> list:
    """
    Removes a specific element from all sublists of a nested list.

    :param nested_list: List of lists containing elements.
    :param element_to_remove: Element to be removed from the lists.
    :return: A new nested list with the specific element removed.
    """
    return [
        [element for element in sublist if element != element_to_remove]
        for sublist in nested_list
    ]


def reduce_chat_history_tokens(chat_history_dict, token_limit=40000):
    """
    Reduces the chat history in the dictionary by two human/AI interactions if the token count exceeds the token_limit.

    :param chat_history_dict: Dictionary containing chat history under the 'chat_history' key.
    :param token_limit: The target token limit for the chat history.
    :return: Dictionary with reduced chat history.
    """
    chat_history = chat_history_dict.get("chat_history", "")
    tokens = enc.encode(chat_history)
    if len(tokens) > token_limit:
        interactions = chat_history.split(
            "Human:"
        )  # Assuming each interaction is separated by a newline
        if (
            len(interactions) > 4
        ):  # Check if there are at least two interactions to remove
            reduced_history = "\n".join(
                interactions[-4:]
            )  # Keep the last four interactions (two human/AI cycles)
            chat_history_dict["chat_history"] = reduced_history
    return chat_history_dict


def reduce_similar_docs_tokens(similar_docs, token_limit=60000):
    """
    Reduces the number of documents in similar_docs to ensure the total token count is below the token_limit.

    :param similar_docs: The list of similar documents.
    :param token_limit: The target token limit for the entire prompt.
    :return: Reduced list of similar documents.
    """
    while len(similar_docs) > 1 and len(enc.encode(str(similar_docs))) > token_limit:
        similar_docs.pop(
            0
        )  # Remove documents from the start until the token limit is met
    return similar_docs


def handle_errors(arg0, e):
    """Handles all incoming errors and formats the error message for streamlit output."""
    error_message = f"{arg0}{e}"
    st.error(error_message)
    return error_message


class TokenLimitExceededException(Exception):
    """Exception raised when the token limit is exceeded."""

    def __init__(self, message="Token limit exceeded"):
        self.message = message
        super().__init__(self.message)


def get_answer(query: str) -> str:
    """Generate an answer based on similar documents and the provided query."""
    try:
        similar_docs = get_similar_docs(index, query)
        sources = get_sources(similar_docs)
        queries = get_search_query(sources)
    except Exception as e:
        return handle_errors(
            "Error occurred while retreiving and processing documents from Pinecones: ",
            e,
        )
    try:
        url_list = [parse_urls(search.run(f"{link}")) for link in queries]
    except Exception as e:
        return handle_errors(
            "Error occurred while fetching additional source URLs: ", e
        )
    try:
        string_to_remove = "/enterprise-server@3.6"
        updated_list = remove_specific_string_from_list(url_list, string_to_remove)
        url_to_remove = "https://playrusvulkan.org/tortoise-git-quick-guide"
        clean_url_list = remove_specific_element_from_list(updated_list, url_to_remove)
    except Exception as e:
        return handle_errors(
            "Error occurred while cleaning up additional source URLs: ", e
        )
    try:
        time.sleep(60.0 / MODEL_REQUEST_LIMIT_PER_MINUTE)
    except Exception as e:
        return handle_errors(
            "Error occurred during handling the API Request Rate Limit: ", e
        )
    try:
        chat_history_dict = memory.load_memory_variables({})
        formatted_prompt = prompt.format(
            human_input=query,
            context=similar_docs,
            chat_history=chat_history_dict["chat_history"],
            url_sources=clean_url_list,
        )

        if len(enc.encode(formatted_prompt)) < 60000:
            return qa_llm.run(
                {
                    "context": similar_docs,
                    "human_input": query,
                    "chat_history": memory.load_memory_variables({}),
                    "url_sources": clean_url_list,
                }
            )
        # Reduce chat history first
        reduced_chat_history_dict = reduce_chat_history_tokens(chat_history_dict)
        formatted_prompt_with_reduced_history = prompt.format(
            human_input=query,
            context=similar_docs,
            chat_history=reduced_chat_history_dict["chat_history"],
            url_sources=clean_url_list,
        )

        if len(enc.encode(formatted_prompt_with_reduced_history)) < 60000:
            return qa_llm.run(
                {
                    "context": similar_docs,
                    "human_input": query,
                    "chat_history": reduced_chat_history_dict.get("chat_history", ""),
                    "url_sources": clean_url_list,
                }
            )
        similar_docs = reduce_similar_docs_tokens(similar_docs)
        processed_prompt = prompt.format(
            human_input=query,
            context=similar_docs,
            chat_history=reduced_chat_history_dict["chat_history"],
            url_sources=clean_url_list,
        )
        # Final check on the processed prompt's token count
        if len(enc.encode(processed_prompt)) > 60000:
            raise TokenLimitExceededException(
                "Final prompt still exceeds 60,000 tokens. Please reduce prompt length."
            )
        else:
            return qa_llm.run(
                {
                    "context": similar_docs,
                    "human_input": query,
                    "chat_history": reduced_chat_history_dict.get("chat_history", ""),
                    "url_sources": clean_url_list,
                }
            )
    except TokenLimitExceededException as e:
        return f"Error: {e}"
    except Exception as e:
        return handle_errors(
            "Error occurred while generating an answer with LLMChain: ", e
        )
