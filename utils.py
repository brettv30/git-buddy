from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchResults
import pinecone
from dotenv import load_dotenv
import os
import logging
import re

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set Const Variables
EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_NAME = "git-buddy-index"
MODEL_NAME = "gpt-3.5-turbo"
PROMPT_TEMPLATE = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
You should only ever answer questions about Git, GitHub, or TortoiseGit. Never answer any other questions even if you think you know the correct answer. 
If possible, please provide example code and always provide the name of the file you pulled an answer from if you pulled an answer from the context.

If a question is ambiguous please refer to the conversation history to see if that helps in answering the question at the end:
 {chat_history}

Use the following pieces of context to answer the question at the end: 
{context}

Question: {human_input}
Answer:
Source(s):"""

# Validate environment variables
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logging.error(
        "Missing required environment variables. Please check your .env file."
    )
    exit(1)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
index = Pinecone.from_existing_index(INDEX_NAME, embeddings)

# Set up LangChain
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.5)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="human_input",
    k=3,
)
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "human_input"], template=PROMPT_TEMPLATE
)
qa_llm = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

search = DuckDuckGoSearchResults()


def get_similar_docs(query: str, k: int = 3, score: bool = False) -> list:
    """Retrieve similar documents from the index based on the given query."""
    return (
        index.similarity_search_with_score(query, k=k)
        if score
        else index.similarity_search(query, k=k)
    )


def get_sources(docs: str) -> str:
    # Extracting 'source' from each item's metadata
    sources = [doc["metadata"]["source"] for doc in docs]

    return sources


def get_answer(query: str) -> str:
    """Generate an answer based on similar documents and the provided query."""
    similar_docs = get_similar_docs(query)
    print(similar_docs[0].metadata["source"])
    # Regular expression to extract information between '\' and '.'
    pattern = r"\\(.*?)\."
    match = re.search(pattern, similar_docs[0].metadata["source"])

    if match:
        result = match.group(1)
        print("Extracted text:", result)

        search_results = search.run(f"{result}")
        print(search_results)
    else:
        print("No match found")

    answer = qa_llm.run(
        {
            "context": similar_docs,
            "human_input": query,
            "chat_history": memory.load_memory_variables({}),
        }
    )
    return answer


print(get_answer("What are GitHub Issues?"))
