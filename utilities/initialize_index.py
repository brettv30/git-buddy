import time
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from utils import DocumentManager, Config

# Initialize the configuration componenets
config = Config()
doc_manager = DocumentManager(config)

openai_api_key_env = config.openai_api_key
pinecone_api_key_env = config.pinecone_api_key

# List of URLs we want to iterate through and add to documentation
url_list = [
    "https://docs.github.com/en",
    "https://git-scm.com/book/en/v2",
]

# Load in knowledge base
docs = [doc_manager.load_docs(url) for url in url_list]

pdfs = doc_manager.load_pdfs()

# Flatten the list of lists into just a single list
flattened_list = [element for sublist in docs for element in sublist]

full_list = flattened_list + pdfs

# Clean the documents/pdfs
transformed_doc = doc_manager.clean_docs(full_list)

# Create docs to pass to langchain
chunked_documents = doc_manager.split_docs(transformed_doc)

# Initialize the Pinecone Vector Database
pinecone.init(environment="gcp-starter")  # next to api key in console

if config.index_name in pinecone.list_indexes():
    pinecone.delete_index(config.index_name)

# we create a new index
pinecone.create_index(
    name=config.index_name,
    metric="dotproduct",  # https://www.pinecone.io/learn/vector-similarity/
    dimension=1536,  # 1536 dim of text-embedding-ada-002
)

# wait for index to be initialized
while not pinecone.describe_index(config.index_name).status["ready"]:
    time.sleep(1)

# Initialize the Pinecone Vector Database
pinecone.init(environment="gcp-starter")  # next to api key in console

# if you already have an index, you can load it using the steps below
embeddings = OpenAIEmbeddings(model=config.embeddings_model)


# This loads the embedded documents into your Pinecone Index
index = Pinecone.from_documents(
    chunked_documents, embeddings, index_name=config.index_name
)

print("Index Creation Complete!")
