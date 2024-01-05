# Build the Knowledge Base
import time
import pinecone
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from utils import load_docs, load_pdfs, flatten_list_of_lists, clean_docs, split_docs

directory = "data"
index_name = "git-buddy-index"
embeddings_model = "text-embedding-ada-002"

openai_api_key_env = st.secrets["OPENAI_API_KEY"]
pinecone_api_key_env = st.secrets["PINECONE_API_KEY"]

# List of URLs we want to iterate through and add to documentation
url_list = [
    "https://docs.github.com/en",
    "https://git-scm.com/book/en/v2",
]

# Load in knowledge base
docs = [load_docs(url) for url in url_list]

pdfs = load_pdfs(directory)

# Flatten the list of lists into just a single list
flattened_list = flatten_list_of_lists(docs)

full_list = flattened_list + pdfs

# Clean the documents/pdfs
transformed_doc = clean_docs(full_list)

# Create docs to pass to langchain
chunked_documents = split_docs(transformed_doc)

# Initialize the Pinecone Vector Database
pinecone.init(environment="gcp-starter")  # next to api key in console

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# we create a new index
pinecone.create_index(
    name=index_name,
    metric="dotproduct",  # https://www.pinecone.io/learn/vector-similarity/
    dimension=1536,  # 1536 dim of text-embedding-ada-002
)

# wait for index to be initialized
while not pinecone.describe_index(index_name).status["ready"]:
    time.sleep(1)

# Initialize the Pinecone Vector Database
pinecone.init(environment="gcp-starter")  # next to api key in console

# if you already have an index, you can load it like this
embeddings = OpenAIEmbeddings(model=embeddings_model)


# This actually loads the data/embeddings into your index
index = Pinecone.from_documents(chunked_documents, embeddings, index_name=index_name)

print("Index Creation Complete!")
