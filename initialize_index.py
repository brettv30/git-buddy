# Build the Knowledge Base
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.agents import Tool, load_tools, initialize_agent
from dotenv import load_dotenv
import os
import pinecone
import time

load_dotenv()

openai_api_key_env = os.getenv("OPENAI_API_KEY")
pinecone_api_key_env = os.getenv("PINECONE_API_KEY")
directory = "data"
index_name = "git-buddy-index"
embeddings_model = "text-embedding-ada-002"
llm_model = "gpt-3.5-turbo"


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


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
