from dotenv import load_dotenv
import os
import logging
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchTool, BaseTool
from langchain.agents import Tool, initialize_agent

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

# Set up the turbo LLM
turbo_llm = ChatOpenAI(temperature=0.5, model_name=MODEL_NAME)

search = DuckDuckGoSearchTool()
# defining a single tool
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Useful for when you need to provide links for documentation pulled from the Pinecone database.",
    )
]

tools = [search]

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=3, return_messages=True
)

# create our agent
conversational_agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=turbo_llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
)
