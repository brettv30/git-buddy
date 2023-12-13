from dotenv import load_dotenv
import os
import logging
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchRun, BaseTool
from langchain.agents import Tool, initialize_agent
from CustomPromptTemplate import *

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

search = DuckDuckGoSearchRun()


def duck_wrapper_github(input_text):
    search_results = search.run(f"site:https://docs.github.com/en {input_text}")
    return search_results


def duck_wrapper_git(input_text):
    search_results = search.run(f"site:https://git-scm.com {input_text}")
    return search_results


tools = [
    Tool(
        name="Search GitHub",
        func=duck_wrapper_github,
        description="Useful for when you need to answer GitHub documentation",
    )
]

tools = ["Search GitHub"]


# Set up the base template
template = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
You should only ever answer questions about Git, GitHub, or TortoiseGit. Never answer any other questions even if you think you know the correct answer. You have access to the following tools: 

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a helpful assistant when giving your final answer.

Question: {input}
{agent_scratchpad}"""

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)

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
