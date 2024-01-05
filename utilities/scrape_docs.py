import re
import streamlit as st
from bs4 import BeautifulSoup as Soup
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


OPENAI_API_KEY = st.secrets["SCRAPE_KEY"]
embeddings_model = "text-embedding-ada-002"
directory = "data"


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


def load_docs(url, max_depth=3):
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


# List of URLs we want to iterate through and add to documentation
url_list = [
    "https://docs.github.com/en",
    "https://git-scm.com/book/en/v2",
]

docs = [load_docs(url) for url in url_list]

pdfs = load_pdfs(directory)

flattened_list = flatten_list_of_lists(docs)

full_list = flattened_list + pdfs

transformed_doc = clean_docs(full_list)

chunked_documents = split_docs(transformed_doc)
