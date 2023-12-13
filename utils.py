from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pinecone
from dotenv import load_dotenv
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Set Const Variables
EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_NAME = "git-buddy-index"
MODEL_NAME = "gpt-3.5-turbo"
PROMPT_TEMPLATE = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
You should only ever answer questions about Git, GitHub, or TortoiseGit. Never answer any other questions even if you think you know the correct answer. Only use the serpAPI tool to provide links to additional documentation. Never use the serpAPI tool to search the web to answer a question unrelated to Git, GitHub, or TortoiseGit.
Do the following when responding to questions:
    If possible, please provide example code
    Always provide the name of the file you pulled an answer from if you pulled an answer from the context
    When you provide the name of the file you pulled an answer from, use the serpAPI tool to provide the actual link where you can find the exact information on the internet

If a question is ambiguous please refer to the conversation history to see if that helps in answering the question at the end:
 {chat_history}

Use the following pieces of context to answer the question at the end: 
{context}

Question: {human_input}
Answer:"""

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


def get_similar_docs(query: str, k: int = 3, score: bool = False) -> list:
    """Retrieve similar documents from the index based on the given query."""
    return (
        index.similarity_search_with_score(query, k=k)
        if score
        else index.similarity_search(query, k=k)
    )


def get_answer(query: str) -> str:
    """Generate an answer based on similar documents and the provided query."""
    similar_docs = get_similar_docs(query)
    answer = qa_llm.run(
        {
            "context": similar_docs,
            "human_input": query,
            "chat_history": memory.load_memory_variables({}),
        }
    )
    return answer


# Agent Creation Business
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=index.as_retriever(),
#     verbose=True,
#     chain_type_kwargs={"verbose": True, "prompt": prompt, "memory": memory},
# )
# Create Tools for Agent
# search = SerpAPIWrapper()
# tools = load_tools([], llm=llm)

# tools.append(
#     Tool.from_function(
#         func=qa.run,
#         name="Git Buddy",
#         description="Use this tool when answering any question related to Git, GitHub, or TortoiseGit",
#     )
# )
# tools.append(
#     Tool.from_function(
#         func=search.run,
#         name="serpAPI",
#         description="Only use this tool when providing website links for documentation that was pulled from model context",
#     )
# )

# Initialize Agent
# agent = initialize_agent(
#     agent="chat-conversational-react-description",
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=2,
#     early_stopping_method="generate",
#     memory=memory,
# )

# def get_answer_agent(query: str) -> str:
#     """Generate an answer based on similar documents and the provided query."""
#     similar_docs = get_similar_docs(query)

#     # Retrieve the current chat history from memory
#     # Ensure that you're accessing chat_history correctly as a list
#     current_chat_history = memory.load_memory_variables({})

#     # Verify that current_chat_history is a list
#     if not isinstance(current_chat_history, list):
#         current_chat_history = [current_chat_history] if current_chat_history else []

#     input_dict = {
#         "input": {
#             "context": similar_docs,
#             "question": query,
#             "chat_history": current_chat_history,  # Ensure this is a list
#         }
#     }
#     return agent.run(input_dict)
