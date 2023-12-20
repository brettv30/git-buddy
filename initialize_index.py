# Build the Knowledge Base
import time
import pinecone
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai_api_key_env = st.secrets("OPENAI_API_KEY")
pinecone_api_key_env = st.secrets("PINECONE_API_KEY")
directory = "data"
index_name = "git-buddy-index"
embeddings_model = "text-embedding-ada-002"
llm_model = "gpt-3.5-turbo"


def load_docs(directory):
    loader = DirectoryLoader(directory)
    return loader.load()


def split_docs(documents, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


documents = load_docs(directory)
docs = split_docs(documents)


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
# index = Pinecone.from_existing_index(index_name, embeddings)

# This actually loads the data/embeddings into your index
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
