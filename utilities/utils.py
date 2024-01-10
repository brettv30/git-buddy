import re
import time
import tiktoken
import pinecone
import streamlit as st
from langchain.chains import LLMChain
from bs4 import BeautifulSoup as Soup
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.tools import DuckDuckGoSearchResults
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Set environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Set Const Variables
MODEL_REQUEST_LIMIT_PER_MINUTE = 500
MODEL_TOKEN_LIMIT_PER_QUERY = 4097
EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_NAME = "git-buddy-index"
MODEL_NAME = "gpt-3.5-turbo"
RETRIEVED_DOCUMENTS = (
    50  # This one can vary while we test out different retrieval methods
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


# Create functions used to scrape & extract data
def remove_extra_whitespace(my_str):
    # Remove useless page content from GitHub docs
    string_to_remove1 = "This book is available in  English.   Full translation available in  azərbaycan dili,български език,Deutsch,Español,Français,Ελληνικά,日本語,한국어,Nederlands,Русский,Slovenščina,Tagalog,Українська简体中文,   Partial translations available in  Čeština,Македонски,Polski,Српски,Ўзбекча,繁體中文,   Translations started for  Беларуская,فارسی,Indonesian,Italiano,Bahasa Melayu,Português (Brasil),Português (Portugal),Svenska,Türkçe. The source of this book is hosted on GitHub.Patches, suggestions and comments are welcome."
    string_to_remove2 = "Help and supportDid this doc help you?YesNoPrivacy policyHelp us make these docs great!All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.Make a contributionLearn how to contributeStill need help?Ask the GitHub communityContact supportLegal© 2023 GitHub, Inc.TermsPrivacyStatusPricingExpert servicesBlog"

    interim_string = my_str.replace(string_to_remove1, "").replace(
        string_to_remove2, ""
    )

    # remove all useless whitespace in the string
    clean_string = (re.sub(" +", " ", (interim_string.replace("\n", " ")))).strip()

    # Pattern to match a period followed by a capital letter
    pattern = r"\.([A-Z])"
    # Replacement pattern - a period, a space, and the matched capital letter
    replacement = r". \1"

    # Substitute the pattern in the text with the replacement pattern
    return re.sub(pattern, replacement, clean_string)


def load_docs(url, max_depth=6):
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=lambda x: " ".join(
            [p.get_text() for p in Soup(x, "html.parser").find_all("p")]
        ),
        exclude_dirs=[
            "https://docs.github.com/en/enterprise-cloud@latest",
            "https://docs.github.com/en/enterprise-server@3.11",
            "https://docs.github.com/en/enterprise-server@3.10",
            "https://docs.github.com/en/enterprise-server@3.9",
            "https://docs.github.com/en/enterprise-server@3.8",
            "https://docs.github.com/en/enterprise-server@3.7",
        ],
    )
    return loader.load()


def load_pdfs(directory):
    loader = DirectoryLoader(directory)
    return loader.load()


def flatten_list_of_lists(list_of_lists):
    # This function takes a list of lists (nested list) as input.
    # It returns a single flattened list containing all the elements
    # from the sublists, maintaining their order.

    # Using a list comprehension, iterate through each sublist in the list of lists.
    # For each sublist, iterate through each element.
    # The element is then added to the resulting list.
    return [element for sublist in list_of_lists for element in sublist]


def clean_docs(url_docs):
    cleaned_docs = [
        remove_extra_whitespace(element.page_content.replace("\n", ""))
        for element in url_docs
    ]
    metadata = [document.metadata for document in url_docs]

    return [
        Document(page_content=cleaned_docs[i], metadata=metadata[i])
        for i in range(len(url_docs))
    ]


def split_docs(documents, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)


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

    compressor = CohereRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return (
        prompt,
        index,
        qa_llm,
        search,
        memory,
        compression_retriever,
    )


# Initialize chatbot components
(
    prompt,
    index,
    qa_llm,
    search,
    memory,
    retriever,
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


def get_sources(docs: str) -> list:
    """Extract the 'source' from each document's metadata."""
    return [doc.metadata["source"] for doc in docs]


def get_search_query(sources: list) -> list:
    """Generate search queries from the list of sources."""
    search_list = []
    for source in sources:
        if source == "data\\TortoiseGit-Manual.pdf":
            sources.remove(source)
            if source not in search_list:
                search_list.append("TortoiseGit-Manual")
        elif source == "data\\TortoiseGitMerge-Manual.pdf":
            sources.remove(source)
            if source not in search_list:
                search_list.append("TortoiseGitMerge-Manual")

    url_list = [parse_urls(search.run(f"{link}")) for link in search_list]

    for url in url_list:
        sources.extend(url)

    return sources


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


def remove_specific_element_from_list(url_list: list, element_to_remove: str) -> list:
    """
    Removes a specific element from all sublists of a nested list.

    :param nested_list: List of lists containing elements.
    :param element_to_remove: Element to be removed from the lists.
    :return: A new nested list with the specific element removed.
    """
    return [element for element in url_list if element != element_to_remove]


def reduce_chat_history_tokens(chat_history_dict):
    """Extract and return Human-AI combos from the text, dropping the first two interactions."""
    chat_history = chat_history_dict.get("chat_history", "")

    pattern = r"(Human:.*?)(?=Human:|$)|(AI:.*?)(?=AI:|$)"
    matches = re.findall(pattern, chat_history, re.DOTALL)

    # Each match is a tuple, where one of the elements is empty. We join the tuple to get the full text.
    combos = ["".join(match) for match in matches]

    while len(combos) > 2:
        combos.pop(0)

    return {"chat_history": "\n".join(combos)}


def reduce_doc_tokens(
    docs,
    incoming_prompt,
    user_query,
    reduced_chat_history,
    url_list,
    token_limit=MODEL_TOKEN_LIMIT_PER_QUERY,
):
    """
    Reduces the number of documents in similar_docs to ensure the total token count is below the token_limit.

    :param similar_docs: The list of similar documents.
    :param token_limit: The target token limit for the entire prompt.
    :return: Reduced list of similar documents.
    """
    while len(docs) > 1 and len(enc.encode(str(incoming_prompt))) > token_limit:
        docs.pop(0)  # Remove documents from the start until the token limit is met
        incoming_prompt = prompt.format(
            human_input=user_query,
            context=docs,
            chat_history=reduced_chat_history["chat_history"],
            url_sources=url_list,
        )
    return docs


def handle_errors(arg0, e):
    """Handles all incoming errors and formats the error message for streamlit output."""
    return f"{arg0}{e}"


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
        similar_docs = reduce_docs_tokens(similar_docs)
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


def get_improved_answer(query):
    try:
        relevant_docs = get_relevant_docs(retriever, query)
    except Exception as e:
        return handle_errors(
            "Error occurred in Contextual Compression Pipeline. Please try query again. If error persists create an issue on GitHub. Additional Error Information: ",
            e,
        )
    try:
        sources = get_sources(relevant_docs)
        links = get_search_query(sources)
    except Exception as e:
        return handle_errors(
            "Error occurred while retrieving document sources. Please try query again. Additional Error Information: ",
            e,
        )
    try:
        url_to_remove = "https://playrusvulkan.org/tortoise-git-quick-guide"
        clean_url_list = remove_specific_element_from_list(links, url_to_remove)
    except Exception as e:
        return handle_errors(
            "Error occurred while cleaning up source URLs. Please try query again. Additional Error Information: ",
            e,
        )

    time.sleep(60.0 / MODEL_REQUEST_LIMIT_PER_MINUTE)

    try:
        return verify_api_limits(query, relevant_docs, clean_url_list)
    except Exception as e:
        return handle_errors(
            "Error occurred while generating an answer with LLMChain: ", e
        )


def verify_api_limits(query, relevant_docs, clean_url_list):
    chat_history_dict = memory.load_memory_variables({})
    formatted_prompt = prompt.format(
        human_input=query,
        context=relevant_docs,
        chat_history=chat_history_dict["chat_history"],
        url_sources=clean_url_list,
    )

    if len(enc.encode(formatted_prompt)) < MODEL_TOKEN_LIMIT_PER_QUERY:
        return qa_llm.run(
            {
                "context": relevant_docs,
                "human_input": query,
                "chat_history": memory.load_memory_variables({}),
                "url_sources": clean_url_list,
            }
        )
    # Reduce chat history first
    reduced_chat_history_dict = reduce_chat_history_tokens(chat_history_dict)
    formatted_prompt_with_reduced_history = prompt.format(
        human_input=query,
        context=relevant_docs,
        chat_history=reduced_chat_history_dict["chat_history"],
        url_sources=clean_url_list,
    )

    if (
        len(enc.encode(formatted_prompt_with_reduced_history))
        < MODEL_TOKEN_LIMIT_PER_QUERY
    ):
        return qa_llm.run(
            {
                "context": relevant_docs,
                "human_input": query,
                "chat_history": reduced_chat_history_dict["chat_history"],
                "url_sources": clean_url_list,
            }
        )
    shorter_relevant_docs = reduce_doc_tokens(
        relevant_docs,
        formatted_prompt_with_reduced_history,
        query,
        reduced_chat_history_dict,
        clean_url_list,
    )
    processed_prompt = prompt.format(
        human_input=query,
        context=shorter_relevant_docs,
        chat_history=reduced_chat_history_dict["chat_history"],
        url_sources=clean_url_list,
    )
    if len(enc.encode(processed_prompt)) > MODEL_TOKEN_LIMIT_PER_QUERY:
        raise TokenLimitExceededException(
            "Final prompt still exceeds 4097 tokens. Please reduce prompt length."
        )
    return qa_llm.run(
        {
            "context": relevant_docs,
            "human_input": query,
            "chat_history": reduced_chat_history_dict["chat_history"],
            "url_sources": clean_url_list,
        }
    )
