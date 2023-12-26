from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utilities.scrape_docs import *

# List of URLs we want to iterate through and add to documentation
url_list = [
    "https://docs.github.com/en/get-started",
    "https://docs.github.com/en/authentication",
    "https://docs.github.com/en/repositories",
    # "https://docs.github.com/en/pull-requests",
    # "https://docs.github.com/en/copilot",
    # "https://docs.github.com/en/actions",
    # "https://docs.github.com/en/pages",
    # "https://docs.github.com/en/migrations"
    # "https://docs.github.com/en/code-security",
    # "https://docs.github.com/en/issues",
    # "https://docs.github.com/en/search-github",
    "https://git-scm.com/book/en/v2",
    # "https://tortoisegit.org/docs/tortoisegitmerge"
    "https://tortoisegit.org/docs/tortoisegit",
]

docs = [load_docs(url) for url in url_list]

flattened_list = flatten_list_of_lists(docs)

transformed_doc = clean_docs(flattened_list)

chunked_documents = split_docs(transformed_doc)

retriever = FAISS.from_documents(chunked_documents, OpenAIEmbeddings()).as_retriever()

results = retriever.get_relevant_documents("What is Git?")

print(results)
