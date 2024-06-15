import time
from pinecone import Pinecone
from utils import DocumentManager, Config
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initialize the configuration componenets
config = Config()
doc_manager = DocumentManager(config)

openai_api_key_env = config.openai_api_key
pinecone_api_key_env = config.pinecone_api_key

# Configure Client
pc = Pinecone(api_key=pinecone_api_key_env)

# Only run if the index cannot be found
if config.index_name not in pc.list_indexes().names():
    # Create a new index
    pc.create_index(
        name=config.index_name,
        metric="dotproduct",  # https://www.pinecone.io/learn/vector-similarity/
        dimension=1536,  # 1536 dim of text-embedding-ada-002
    )

    # wait for index to be initialized
    while not pc.describe_index(config.index_name).status["ready"]:
        time.sleep(1)

    # Grab embeddings model
    embeddings = OpenAIEmbeddings(model=config.embeddings_model)

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

    # Create docs to pass to Pinecone
    chunked_documents = doc_manager.split_docs(transformed_doc)

    # Load the embedded documents into your Pinecone Vector DB
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_documents, embedding=embeddings, index_name=config.index_name
    )

    print("Index Creation Complete!")
else:
    print("Index already exists!")
