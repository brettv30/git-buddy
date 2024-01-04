from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from bs4 import BeautifulSoup as Soup
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader
import streamlit as st

OPENAI_API_KEY = st.secrets["SCRAPE_KEY"]
embeddings_model = "text-embedding-ada-002"
directory = "data"


def remove_extra_whitespace(my_str):
    # Remove useless page content from GitHub docs
    string_to_remove1 = "\nThis book is available in\n  English.\n \n  Full translation available in\n  \nazərbaycan dili,\nбългарски език,\nDeutsch,\nEspañol,\nFrançais,\nΕλληνικά,\n日本語,\n한국어,\nNederlands,\nРусский,\nSlovenščina,\nTagalog,\nУкраїнська\n简体中文,\n\n \n  Partial translations available in\n  \nČeština,\nМакедонски,\nPolski,\nСрпски,\nЎзбекча,\n繁體中文,\n\n \n  Translations started for\n  \nБеларуская,\nفارسی,\nIndonesian,\nItaliano,\nBahasa Melayu,\nPortuguês (Brasil),\nPortuguês (Portugal),\nSvenska,\nTürkçe.\n\n \nThe source of this book is hosted on GitHub.\nPatches, suggestions and comments are welcome.\n"
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
    # "https://docs.github.com/en/get-started",
    "https://docs.github.com/en/authentication",
    # "https://docs.github.com/en/repositories",
    # "https://docs.github.com/en/pull-requests",
    # "https://docs.github.com/en/copilot",
    # "https://docs.github.com/en/actions",
    # "https://docs.github.com/en/pages",
    # "https://docs.github.com/en/migrations"
    # "https://docs.github.com/en/code-security",
    # "https://docs.github.com/en/issues",
    # "https://docs.github.com/en/search-github",
    # "https://git-scm.com/book/en/v2",
]

docs = [load_docs(url) for url in url_list]

pdfs = load_pdfs(directory)

flattened_list = flatten_list_of_lists(docs)

full_list = flattened_list + pdfs

transformed_doc = clean_docs(full_list)

chunked_documents = split_docs(transformed_doc)

# initialize the bm25 retriever and chroma retriever
bm25_retriever = BM25Retriever.from_texts(chunked_documents)
bm25_retriever.k = 5

chroma_retriever = Chroma.from_documents(
    chunked_documents, OpenAIEmbeddings(model=embeddings_model)
).as_retriever(search_kwargs={"k": 5})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

results = ensemble_retriever.get_relevant_documents(
    "What is the difference between Git and GitHub?"
)

print(results)
