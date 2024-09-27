import re
import streamlit as st
from langsmith import Client
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from bs4 import BeautifulSoup as Soup
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    DirectoryLoader,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain


class Config:
    """
    Manages the configuration settings for the application, including API keys, model limits, and directory paths.

    Args:
        openai_api_key (str): The OpenAI API key.
        pinecone_api_key (str): The Pinecone API key.
        cohere_api_key (str): The Cohere API key.
        directory (str): The directory for storing data.
        embeddings_model (str): The name of the embeddings model to use.
        index_name (str): The name of the Pinecone index.
        model_name (str): The name of the language model to use.
        retrieved_documents (int): The number of documents to retrieve.
        prompt_template (str): The template for the prompt used in the RAG (Retrieval Augmented Generation) model.
        contextualize_q_system_prompt (str): The system prompt for the contextualize_q model.
        contextualize_q_prompt (ChatPromptTemplate): The prompt template for the contextualize_q model.
        rag_prompt (ChatPromptTemplate): The prompt template for the RAG model.
        total_tokens (int): The total number of tokens used in the current session.
    """

    def __init__(self):
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        self.cohere_api_key = st.secrets["COHERE_API_KEY"]
        self.directory = "data"
        self.embeddings_model = "text-embedding-ada-002"
        self.index_name = "git-buddy-index"
        self.model_name = "gpt-4o-mini"
        self.client = Client()
        self.retrieved_documents = 100  # Can vary for different retrieval methods
        self.contextualize_q_prompt = self.client.pull_prompt(
            "contextualize-query-with-chat-history-prompt"
        )
        self.rag_prompt = self.client.pull_prompt("git-buddy-full-prompt")
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
    Initializes and manages the components required for the application, including embeddings, index, retriever, and other necessary objects.

    Attributes:
        config (Config): The configuration object containing settings for the application.
        top_docs (int): The number of top documents to retrieve.
        temperature (float): The temperature parameter for the language model.
        store (dict): A dictionary to store session history for each user.

    Methods:
        initialize_components() -> tuple:
            Initializes the components required for the application, such as embeddings, index, retriever, and language model. Returns a tuple containing the initialized components.
        get_session_history(session_id) -> BaseChatMessageHistory:
            Retrieves the chat message history for the given session ID. If the session ID is not found in the store, a new ChatMessageHistory object is created and stored.
    """

    def __init__(self, config):
        self.config = config
        self.top_docs = 5
        self.temperature = 0.5
        self.store = {}

    def initialize_components(self):
        """
        Initializes and returns the necessary components for the application, including embeddings, index, retriever, compressor, and language model.

        Returns:
            tuple: A tuple containing the initialized components:
                - conversational_rag_chain: A RunnableWithMessageHistory object that combines a retrieval-augmented generation (RAG) chain with message history.
                - history_aware_retriever: A ContextualCompressionRetriever object that retrieves and compresses documents based on the chat history.
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
        """
        Retrieves and manages the chat history for a given session.

        Args:
            session_id (str): The unique identifier for the current session.

        Returns:
            BaseChatMessageHistory: The chat message history for the given session.
        """
        if session_id not in self.store:
            self.store[session_id] = StreamlitChatMessageHistory()
        elif session_id in self.store and len(self.store[session_id].messages) > 16:
                self.store[session_id].messages = self.store[session_id].messages[-14:]
        return self.store[session_id]


class APIHandler(Config):
    """
    The `APIHandler` class is responsible for handling API requests and managing the retrieval and processing of additional sources based on a user's query.

    The class has the following key responsibilities:

    1. Initializes the necessary components and configurations for the API handling process.
    2. Provides a method `find_additional_sources` to generate search queries based on a list of sources and retrieve additional contextual documents.
    3. Implements a `make_request_with_retry` method to attempt making an API call with a retry mechanism in case of rate limit errors.

    The class uses various helper methods and exceptions to handle different aspects of the API handling process, such as URL cleaning, source extraction, and error handling.
    """

    def __init__(self, chain, retriever_chain, max_retries=5):
        self.conf_obj = Config()
        self.comp_obj = ComponentInitializer(self.conf_obj)
        self.search = DuckDuckGoSearchResults()
        self.max_retries = max_retries
        self.chain = chain
        self.retriever_chain = retriever_chain
        self.total_tokens = 0

    def find_additional_sources(self, query) -> list:
        """
        Retrieves additional contextual documents based on the given query.

        Args:
            query (str): The user's query to be used for searching additional sources.

        Returns:
            list: A list of cleaned URLs for the additional sources.
        """

        st.write("Retrieving Contextual Documents...")
        doc_list = self.retriever_chain.invoke({"input": query})

        sources = [doc.metadata["source"] for doc in doc_list]
        search_list = []

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
                    self.parse_urls(self.search.invoke(f"{query} {link}"))
                    for link in search_list 
                ]

            else:
                url_list = [self.parse_urls(self.search.run(query))]

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
        """
        Cleans a list of URLs by removing known problematic URLs.

        Args:
            dup_url_list (list): A list of URLs to be cleaned.

        Returns:
            list: A cleaned list of unique URLs.
        """

        urls_to_remove = [
            "https://playrusvulkan.org/tortoise-git-quick-guide",
            "data\\TortoiseGit-Manual.pdf",
            "data\\TortoiseGitMerge-Manual.pdf",
            "https://debfaq.com/using-tortoisemerge-as-your-git-merge-tool-on-windows/",
        ]  # URLs with known issues
        interim_url_list = [
            element.replace("link: ", "") for element in dup_url_list if element.replace("link: ", "") not in urls_to_remove
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
        pattern = r"link: https://[^\]]+"
        return re.findall(pattern, search_results)

    def make_request_with_retry(self, user_input, additional_sources, session_id):
        """
        Makes a request to the chain with retries in case of exceptions.

        Args:
            additional_sources (list): A list of additional sources to include in the request.

        Returns:
            str: The answer from the chain.

        Raises:
            LLMChainException: If an error occurs while making the request to the LLMChain.
        """

        try:
            with get_openai_callback() as cb:
                result = self.chain.invoke(
                    {
                        "input": user_input,
                        "url_sources": additional_sources,
                    },
                    config={"configurable": {"session_id": f"{session_id}"}},
                )
        except Exception as e:
            return f"Ran into an error while making a request to GPT-4o-mini. The following error was raised {e}"

        return result["answer"]
