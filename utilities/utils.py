import re
import time
import random
import tiktoken
import pinecone
import streamlit as st
from openai import RateLimitError
from langchain.chains import LLMChain
from bs4 import BeautifulSoup as Soup
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


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
        self.model_token_limit_per_query = 16000
        self.directory = "data"
        self.embeddings_model = "text-embedding-ada-002"
        self.index_name = "git-buddy-index"
        self.model_name = "gpt-3.5-turbo-1106"
        self.retrieved_documents = 100  # Can vary for different retrieval methods
        self.prompt_template = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
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
        self.encoding = tiktoken.get_encoding("cl100k_base")


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


class ComponentInitializer:
    """
    Initializer class used to initialize all Pinecone, Cohere, Langchain, and OpenAI Components

    Attributes:
        config (Config): Configuration settings for the component initializer.
        memory (int): Maximum number of Human/AI interactions retained in model memory
        top_docs (int): Maximum number of documents retained after document reranking
        temperature (float): Number indicating the level of predictability as it pertains to model output
    """

    def __init__(
        self,
        config,
        memory=4,
        top_docs=5,
        temperature=0.5,
    ):
        self.config = config
        self.doc_memory = memory
        self.top_docs = top_docs
        self.temperature = temperature

    def initialize_components(self):
        """
        Initialize components for Pinecone and LangChain including embeddings, index, retriever, etc.

        Returns:
            tuple: A tuple containing initialized components such as prompt, index, QA LLM, search, memory, and compression retriever.
        """
        pinecone.init(api_key=self.config.pinecone_api_key, environment="gcp-starter")
        embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
        index = Pinecone.from_existing_index(self.config.index_name, embeddings)
        retriever = index.as_retriever(
            search_type="mmr", search_kwargs={"k": self.config.retrieved_documents}
        )
        llm = ChatOpenAI(
            model_name=self.config.model_name, temperature=self.temperature
        )
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="human_input",
            k=self.doc_memory,
        )
        prompt = PromptTemplate(
            input_variables=["chat_history", "context", "human_input", "url_sources"],
            template=self.config.prompt_template,
        )
        qa_llm = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
        search = DuckDuckGoSearchResults()

        compressor = CohereRerank(top_n=self.top_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        return (
            prompt,
            qa_llm,
            search,
            memory,
            compression_retriever,
        )


class APIHandler:
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

    def __init__(
        self, config, prompt_parser, llm_prompt, chat_memory, qa_llm, max_retries=5
    ):
        self.config = config
        self.prompt_parser = prompt_parser
        self.llm_prompt = llm_prompt
        self.chat_memory = chat_memory
        self.qa_llm = qa_llm
        self.max_retries = max_retries

    def make_request_with_retry(self, api_call):
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
        for i in range(self.max_retries):
            try:
                return api_call()
            except RateLimitError:
                st.write(
                    "Rate Limit was hit. Waiting a little while before trying again..."
                )
                wait_time = (2**i) + random.random()
                time.sleep(wait_time)
        raise APIRequestException(
            "OpenAI API Request Rate Limit was reached. Please wait a few minutes before trying again."
        )

    def verify_api_limits(self, query, relevant_docs, clean_url_list):
        """
        Verify if the generated prompt is within the token limit of the OpenAI API and run the LLM chain.

        Args:
            query (str): The user query to process.
            relevant_docs (list): List of relevant documents to include in the prompt.
            clean_url_list (list): List of cleaned URLs to include in the prompt.

        Returns:
            Any: The result from the LLM chain if the prompt is within the token limit.

        Raises:
            TokenLimitExceededException: If the final prompt exceeds the token limit.
        """
        try:
            chat_history_dict = self.chat_memory.load_memory_variables({})
            formatted_prompt = self.llm_prompt.format(
                human_input=query,
                context=relevant_docs,
                chat_history=chat_history_dict["chat_history"],
                url_sources=clean_url_list,
            )

            if (
                len(self.config.encoding.encode(formatted_prompt))
                < self.config.model_token_limit_per_query
            ):
                st.write("Within the OpenAI API token limit. Running LLM...")
                return self.qa_llm.run(
                    {
                        "context": relevant_docs,
                        "human_input": query,
                        "chat_history": self.chat_memory.load_memory_variables({}),
                        "url_sources": clean_url_list,
                    }
                )

            st.write(
                "Reached the OpenAI API token limit: Removing interactions from chat history..."
            )
            reduced_chat_history_dict = self.prompt_parser.reduce_chat_history_tokens(
                chat_history_dict
            )
            formatted_prompt_with_reduced_history = self.llm_prompt.format(
                human_input=query,
                context=relevant_docs,
                chat_history=reduced_chat_history_dict["chat_history"],
                url_sources=clean_url_list,
            )

            if (
                len(self.config.encoding.encode(formatted_prompt_with_reduced_history))
                < self.config.model_token_limit_per_query
            ):
                st.write(
                    "Within the OpenAI API token limit after reducing chat history. Running LLM..."
                )
                return self.qa_llm.run(
                    {
                        "context": relevant_docs,
                        "human_input": query,
                        "chat_history": reduced_chat_history_dict["chat_history"],
                        "url_sources": clean_url_list,
                    }
                )

            st.write(
                "Still at the OpenAI API token limit: Reducing number of retrieved documents..."
            )
            shorter_relevant_docs = self.prompt_parser.reduce_doc_tokens(
                relevant_docs,
                formatted_prompt_with_reduced_history,
                query,
                reduced_chat_history_dict,
                clean_url_list,
            )
            processed_prompt = self.llm_prompt.format(
                human_input=query,
                context=shorter_relevant_docs,
                chat_history=reduced_chat_history_dict["chat_history"],
                url_sources=clean_url_list,
            )
            if (
                len(self.config.encoding.encode(processed_prompt))
                < self.config.model_token_limit_per_query
            ):
                st.write(
                    "Within the OpenAI API token limit after reducing documents and chat history. Running LLM..."
                )
                return self.qa_llm.run(
                    {
                        "context": relevant_docs,
                        "human_input": query,
                        "chat_history": reduced_chat_history_dict["chat_history"],
                        "url_sources": clean_url_list,
                    }
                )
        except Exception:
            st.write("Couldn't fall under the OpenAI API token limit...")
            return TokenLimitExceededException(
                "Final prompt still exceeds 16,000 tokens. Please ask your question again with less words."
            )


class PromptParser:
    """
    Prompt parsing class used to modify the LLM prompt if errors are hit

    Attributes:
        config (Config): Configuration settings for the prompt parser
        llm_prompt (str): Prompt passed into the Large Language Model
        memory (dict): Dictionary containing the last 4 Human/AI interactions in chat_history
    """

    def __init__(self, config, memory, llm_prompt):
        self.config = config
        self.memory = memory
        self.llm_prompt = llm_prompt

    @staticmethod
    def reduce_chat_history_tokens(chat_history_dict):
        """
        Reduce the token count of chat history by dropping earlier interactions.

        Args:
            chat_history_dict (dict): A dictionary containing chat history.

        Returns:
            dict: A dictionary with reduced chat history.
        """
        chat_history = chat_history_dict.get("chat_history", "")

        pattern = r"(Human:.*?)(?=Human:|$)|(AI:.*?)(?=AI:|$)"
        matches = re.findall(pattern, chat_history, re.DOTALL)

        # Each match is a tuple, where one of the elements is empty. We join the tuple to get the full text.
        combos = ["".join(match) for match in matches]

        while len(combos) > 2:
            combos.pop(0)

        return {"chat_history": "\n".join(combos)}

    def clear_memory(self):
        """
        Clear the conversation memory.
        """
        self.memory.clear()

    def reduce_doc_tokens(
        self,
        docs,
        incoming_prompt,
        user_query,
        reduced_chat_history,
        url_list,
    ):
        """
        Reduce the number of documents to ensure the total token count is below the token limit.

        Args:
            docs (list): List of similar documents.
            incoming_prompt (str): The current prompt to be sent to the model.
            user_query (str): The user's query.
            reduced_chat_history (dict): Reduced chat history.
            url_list (list): List of URL sources.
            token_limit (int): The maximum token limit.

        Returns:
            list: Reduced list of similar documents.
        """
        while (
            len(docs) > 1
            and len(self.config.encoding.encode(str(incoming_prompt)))
            > self.config.model_token_limit_per_query
        ):
            docs.pop(0)  # Remove documents from the start until the token limit is met
            incoming_prompt = self.llm_prompt.format(
                human_input=user_query,
                context=docs,
                chat_history=reduced_chat_history["chat_history"],
                url_sources=url_list,
            )
        return docs


class DocumentParser:
    """
    Document parsing class used to parse document sources and search for supplemental links

    Attributes:
        search (DuckDuckGoSearchResults): Search tool used to find supplemental links for TortoiseGit documents
    """

    def __init__(self, search):
        self.search = search

    def get_search_query(self, query, sources: list) -> list:
        """
        Generate search queries based on a list of sources.

        Args:
            sources (list): A list of sources to generate search queries for.
            query (string): the query sent through the chatbot from the user

        Returns:
            list: A list of generated search queries.
        """
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

        url_list = [
            DocumentParser.parse_urls(self.search.run(f"{query} {link}"))
            for link in search_list
        ]

        for url in url_list:
            sources.extend(url)

        return sources

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


class GitBuddyChatBot:
    """
    Git Buddy class used to return LLM results to the streamlit app

    Attributes:
        config (Config): Configuration settings for the chat bot
        api_handler (APIHandler): API Rate Limit handling for the LLM
        doc_retriever (ContextualCompressionRetriever): Retriever used to find and rerank Pinecone documents
        doc_parser (DocumentParser): Parser used to extract sources from documents
    """

    def __init__(self, config, api_handler, doc_retriever, doc_parser):
        self.config = config
        self.api_handler = api_handler
        self.doc_retriever = doc_retriever
        self.doc_parser = doc_parser

    def get_improved_answer(self, query):
        try:
            st.write("Retrieving relevant documents from Pinecone....")
            relevant_docs = self.doc_retriever.get_relevant_documents(query)
        except Exception:
            return DocumentRetrievalException(
                "Error occurred retrieving relevant documents. Please try query again. If error persists create an issue on GitHub."
            )
        try:
            st.write("Extracting sources from documents...")
            sources = [doc.metadata["source"] for doc in relevant_docs]
            st.write("Finding related webpages...")
            links = self.doc_parser.get_search_query(query, sources)
        except Exception:
            return SourceExtractionException(
                "Error occurred while extracting document sources. Please try query again. If error persists create an issue on GitHub."
            )
        try:
            urls_to_remove = [
                "https://playrusvulkan.org/tortoise-git-quick-guide",
                "data\\TortoiseGit-Manual.pdf",
                "data\\TortoiseGitMerge-Manual.pdf",
                "https://debfaq.com/using-tortoisemerge-as-your-git-merge-tool-on-windows/",
            ]  # URLs with known issues
            interim_url_list = [
                element for element in links if element not in urls_to_remove
            ]
            clean_url_list = list(set(interim_url_list))
        except Exception:
            return URLCleaningException(
                "Error occurred while cleaning source URLs. Please try query again. If error persists create an issue on GitHub."
            )
        try:
            st.write("Verifying we are within API limitations...")
            return self.api_handler.make_request_with_retry(
                lambda: self.api_handler.verify_api_limits(
                    query, relevant_docs, clean_url_list
                )
            )
        except Exception as e:
            return LLMChainException(
                f"Error occurred while generating an answer with LLMChain. Please try query again. Additional error information: {e}"
            )

    @staticmethod
    def set_chat_messages(chat_response):
        """
        Append a new chat message to the session's message list.

        Args:
            chat_response (str): The chat response to append.
        """
        message = {
            "role": "assistant",
            "content": chat_response,
        }
        st.session_state.messages.append(message)


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
