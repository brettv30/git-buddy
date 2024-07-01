import re
import time
import random
import tiktoken
import streamlit as st
from openai import RateLimitError
from langchain.chains import LLMChain
from bs4 import BeautifulSoup as Soup
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import (
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages.utils import convert_to_messages
from langchain.chains.combine_documents.base import DOCUMENTS_KEY
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import RecursiveUrlLoader, DirectoryLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompt_values import PromptValue


class Config:
    """
    Config class to hold configuration settings for the application.

    Attributes:
        openai_api_key (str): API key for OpenAI.
        pinecone_api_key (str): API key for Pinecone.
        cohere_api_key (str): API key for Cohere.
        model_request_limit_per_minute (int): Limit for model requests per minute.
        model_token_limit_per_query (int): Token limit for each query to the model.
        directory (str): Directory path for storing data.
        embeddings_model (str): Model used for text embeddings.
        index_name (str): Name of the Pinecone index.
        model_name (str): Name of the language model.
        retrieved_documents (int): Number of documents to retrieve for each query.
        prompt_template (str): Template for the chatbot prompt.
    """

    def __init__(self):
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        self.cohere_api_key = st.secrets["COHERE_API_KEY"]
        self.model_request_limit_per_minute = 500
        self.model_token_limit_per_query = 10000
        self.directory = "data"
        self.embeddings_model = "text-embedding-ada-002"
        self.index_name = "git-buddy-index"
        self.model_name = "gpt-3.5-turbo"
        self.retrieved_documents = 100  # Can vary for different retrieval methods
        self.prompt_template = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. 
Your responses are geared towards beginners. 
You should only ever answer questions if either the question or the context relates to Git, GitHub, or TortoiseGit. 
Never include the example questions and answers from this prompt in any response.
If possible, please provide example code to help the beginner learn Git commands. 
Never use the sources from the context in an answer, only use the sources from url_sources.
Use the additional sources as recommendaations to the user at the end of the response.

Use the following pieces of context to answer the question at the end:
<context 
{context}
context>

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
"""
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_template),
                MessagesPlaceholder("chat_history"),
                MessagesPlaceholder("url_sources"),
                ("human", "{input}"),
            ]
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.user_query = ""
        self.total_tokens = 0


class DocumentManager:
    """
    Manages the loading and processing of documents.

    Args:
        config (Config): Configuration settings for the document manager.
        max_depth (int): Maximum depth for recursive loading of documents.
        chunk_size (int): Size of each chunk in number of characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
    """

    def __init__(self, config, max_depth=6, chunk_size=400, chunk_overlap=50):
        self.config = config
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_docs(self, url):
        """
        Load documents from a specified URL recursively up to a specified depth.

        Args:
            url (str): The base URL from which to start loading documents.

        Returns:
            list: A list of loaded documents.
        """
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=self.max_depth,
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

    def load_pdfs(self):
        """
        Load PDF documents from a specified directory.

        Args:
            directory (str): The directory path where the PDFs are stored.

        Returns:
            list: A list of loaded PDF documents.
        """
        loader = DirectoryLoader(self.config.directory)
        return loader.load()

    @staticmethod
    def clean_docs(url_docs):
        """
        Clean a list of URL documents by removing extra whitespace and unnecessary content.

        Args:
            url_docs (list): The list of URL documents to clean.

        Returns:
            list: A list of cleaned documents.
        """
        cleaned_docs = [
            DocumentManager.remove_extra_whitespace(
                element.page_content.replace("\n", "")
            )
            for element in url_docs
        ]
        metadata = [document.metadata for document in url_docs]

        return [
            Document(page_content=cleaned_docs[i], metadata=metadata[i])
            for i in range(len(url_docs))
        ]

    @staticmethod
    def remove_extra_whitespace(my_str):
        """
        Remove unnecessary whitespace and specific strings from GitHub documentation content.

        Args:
            my_str (str): The string from which to remove whitespace and specific content.

        Returns:
            str: Cleaned string with unnecessary content and extra whitespace removed.
        """
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

    def split_docs(self, documents):
        """
        Split documents into smaller chunks based on character count, considering overlap.

        Args:
            documents (list): The list of documents to split.
            chunk_size (int): The size of each chunk in number of characters.
            chunk_overlap (int): The number of characters to overlap between chunks.

        Returns:
            list: A list of split and chunked documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        return text_splitter.split_documents(documents)


class ComponentInitializer(Config):
    """
    Initializer class used to initialize all Pinecone, Cohere, Langchain, and OpenAI Components

    Attributes:
        config (Config): Configuration settings for the component initializer.
        memory (int): Maximum number of Human/AI interactions retained in model memory
        top_docs (int): Maximum number of documents retained after document reranking
        temperature (float): Number indicating the level of predictability as it pertains to model output
    """

    def __init__(self, config):
        self.config = config
        self.top_docs = 5
        self.temperature = 0.5
        self.store = {}

    def initialize_components(self):
        """
        Initialize components for Pinecone and LangChain including embeddings, index, retriever, etc.

        Returns:
            tuple: A tuple containing initialized components such as prompt, index, QA LLM, search, memory, and compression retriever.
        """
        embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
        docsearch = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name=self.config.index_name
        )
        retriever = docsearch.as_retriever(
            search_type="mmr", search_kwargs={"k": self.config.retrieved_documents}
        )
        compressor = CohereRerank(top_n=self.top_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        llm = ChatOpenAI(
            model_name=self.config.model_name, temperature=self.temperature
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, compression_retriever, self.config.contextualize_q_prompt
        )

        qa_chain = create_stuff_documents_chain(llm, self.config.rag_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain, history_aware_retriever

    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def clear_history(self):
        session_id = "[001]"
        self.store[session_id] = ChatMessageHistory()


class APIHandler(Config):
    """
    API Handler class used to handle all API Rate Limit Errors

    Attributes:
        config (Config): Configuration settings for the API Handler
        prompt_parser (PromptParser): Prompt Parser methods used to handle token limitations
        llm_prompt (str): Prompt passed into the Large Language Model
        chat_memory (dict): Dictionary containing the last 4 Human/AI interactions in chat_history
        qa_llm (LLMChain): OpenAI LLM model instance
        max_retries (int): Maximum number of retries before the application returns an error
    """

    def __init__(self, chain, retriever_chain, max_retries=5):
        self.conf_obj = Config()
        self.comp_obj = ComponentInitializer(self.conf_obj)
        self.max_retries = max_retries
        self.chain = chain
        self.retriever_chain = retriever_chain
        self.total_tokens = 0

    @staticmethod
    def set_user_query(query):
        Config.user_query = query

    def find_additional_sources(self, query) -> list:
        """
        Generate search queries based on a list of sources.

        Args:
            sources (list): A list of sources to generate search queries for.
            query (string): the query sent through the chatbot from the user

        Returns:
            list: A list of generated search queries.
        """
        st.write("Retrieving Contextual Documents...")
        doc_list = self.retriever_chain.invoke({"input": query})

        sources = [doc.metadata["source"] for doc in doc_list]
        search_list = []

        search = DuckDuckGoSearchRun()

        try:
            if sources:
                for source in sources:
                    if source == "data\\TortoiseGit-Manual.pdf":
                        sources.remove(source)
                        source = "TortoiseGit-Manual"
                        if source not in search_list:
                            search_list.append(source)
                    elif source == "data\\TortoiseGitMerge-Manual.pdf":
                        sources.remove(source)
                        source = "TortoiseGitMerge-Manual"
                        if source not in search_list:
                            search_list.append(source)
                    elif source not in search_list:
                        search_list.append(source)

                st.write("Searching for Additional Sources...")

                url_list = [
                    self.parse_urls(search.run(f"{query} {link}"))
                    for link in search_list
                ]

            else:
                url_list = [self.parse_urls(search.run(query))]

            for url in url_list:
                sources.extend(url)

        except Exception:
            return "Error occurred while extracting document sources. Please try query again. If error persists create an issue on GitHub."

        try:
            return self.clean_url_list(sources)
        except Exception:
            return "Error occurred while cleaning source URLs. Please try query again. If error persists create an issue on GitHub."

    @staticmethod
    def clean_url_list(dup_url_list) -> list:
        urls_to_remove = [
            "https://playrusvulkan.org/tortoise-git-quick-guide",
            "data\\TortoiseGit-Manual.pdf",
            "data\\TortoiseGitMerge-Manual.pdf",
            "https://debfaq.com/using-tortoisemerge-as-your-git-merge-tool-on-windows/",
        ]  # URLs with known issues
        interim_url_list = [
            element for element in dup_url_list if element not in urls_to_remove
        ]
        clean_url_list = list(set(interim_url_list))

        return clean_url_list

    @staticmethod
    def parse_urls(search_results: str) -> list:
        """
        Extract URLs from search results.

        Args:
            search_results (str): The string containing search results with URLs.

        Returns:
            list: A list of extracted URLs.
        """
        pattern = r"https://[^\]]+"
        return re.findall(pattern, search_results)

    def make_request_with_retry(self, additional_sources):
        """
        Attempt to make an API call with a retry mechanism on RateLimitError.

        Args:
            api_call (callable): The API call function to be executed.
            max_retries (int): Maximum number of retries.

        Returns:
            Any: The result of the API call if successful.

        Raises:
            Exception: If the rate limit is still hit after the maximum number of retries.
        """

        try:
            if self.total_tokens > 10000:
                st.write("Over 10K tokens in last prompt. Clearing Chat History...")
                with get_openai_callback() as cb:
                    result = self.chain.invoke(
                        {
                            "input": st.session_state.messages[-1]["content"],
                            "url_sources": convert_to_messages(additional_sources),
                            "chat_history": convert_to_messages([]),
                        },
                        config={"configurable": {"session_id": "[001]"}},
                    )

            with get_openai_callback() as cb:
                result = self.chain.invoke(
                    {
                        "input": st.session_state.messages[-1]["content"],
                        "url_sources": convert_to_messages(additional_sources),
                    },
                    config={"configurable": {"session_id": "[001]"}},
                )
        except Exception as e:
            return f"Ran into an error while making a request to GPT 3.5-Turbo. The following error was raised {e}"

        self.total_tokens = cb.total_tokens

        print(self.total_tokens)

        print(result["input"])
        print(result["url_sources"])
        print(result["chat_history"])
        print(result["context"])
        print(result["answer"])
        return result["answer"]

    def trim_messages(self, chain_input):
        stored_messages = self.store["[001]"].messages
        if len(stored_messages) <= 2:
            return False

        Config.store["[001]"].clear()

        for message in stored_messages[-2:]:
            self.store["[001]"].add_message(message)

        return True


class TokenLimitExceededException(Exception):
    """
    Exception raised when the token limit is exceeded.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class APIRequestException(Exception):
    """Exception raised for errors in API requests."""

    def __init__(self, message):
        super().__init__(message)


class DocumentRetrievalException(Exception):
    """Exception raised for errors retrieving documents."""

    def __init__(self, message):
        super().__init__(message)


class SourceExtractionException(Exception):
    """Exception raised for errors extracting document sources."""

    def __init__(self, message):
        super().__init__(message)


class URLCleaningException(Exception):
    """Exception raised for errors cleaning URLs."""

    def __init__(self, message):
        super().__init__(message)


class LLMChainException(Exception):
    """Exception raised for errors with LLMChain."""

    def __init__(self, message):
        super().__init__(message)
