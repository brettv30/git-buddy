import re
import time
from bs4 import BeautifulSoup as Soup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def stream_data(input: str):
    for word in input.split(" "):
        yield word + " "
        time.sleep(0.02)