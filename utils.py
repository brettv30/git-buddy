import os
import re
import time
import pinecone
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.cache import InMemoryCache
from langchain.vectorstores import Pinecone
from langchain.globals import set_llm_cache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set Const Variables
MODEL_REQUEST_LIMIT_PER_MINUTE = 500
EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_NAME = "git-buddy-index"
MODEL_NAME = "gpt-3.5-turbo"
PROMPT_TEMPLATE = """You are Git Buddy, a helpful assistant that teaches Git, GitHub, and TortoiseGit to beginners. Your responses are geared towards beginners. 
You should only ever answer questions about Git, GitHub, or TortoiseGit. Never answer any other questions even if you think you know the correct answer. 
If possible, please provide example code to help the beginner learn Git commands. Never use the sources from the context in an answer, only use the sources from url_sources.

If a question is ambiguous please refer to the conversation history to see if that helps in answering the question at the end:
{chat_history}

Use the following pieces of context to answer the question at the end: 
{context}

If there are links in the following sources then you MUST link all of the following sources at the end of your answer to the question. You can just keep the entire link in the output, no need to hyperlink with a different name. Do NOT change the links.
{url_sources}

Use the following format:

Question: What is Git?
Answer: Git is a distributed version control system that allows multiple people to collaborate on a project. It tracks changes made to files and allows users to easily manage and merge those changes. Git is known for its speed, scalability, and rich command set. It provides both high-level operations and full access to internals. Git is commonly used in software development to manage source code, but it can also be used for any type of file-based project.
Additional Sources: Here's some additional Git soures to get started! 
    - [Pro Git Book](https://git-scm.com/book/en/v2) 
    - [Git Introduction Videos](https://git-scm.com/videos)
    - [External Git Links](https://git-scm.com/doc/ext)

Begin!

Question: {human_input}
Answer:
Additional Sources: Here's some additional sources!"""

# Set the cache
set_llm_cache(InMemoryCache())


# Initialize Pinecone and LangChain components
def initialize_components():
    """Initialize Pinecone and LangChain components."""
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    index = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.5)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="human_input",
        k=2,
    )
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "human_input", "url_sources"],
        template=PROMPT_TEMPLATE,
    )
    qa_llm = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    search = DuckDuckGoSearchResults()
    return prompt, index, qa_llm, search, memory


# Initialize chatbot components
prompt, index, qa_llm, search, memory = initialize_components()


def get_similar_docs(index, query: str, k: int = 3, score: bool = False) -> list:
    """Retrieve similar documents from the index based on the given query."""
    return (
        index.similarity_search_with_score(query, k=k)
        if score
        else index.similarity_search(query, k=k)
    )


def get_sources(docs: str) -> list:
    """Extract the 'source' from each document's metadata."""
    return [doc.metadata["source"] for doc in docs]


def get_search_query(sources: list) -> list:
    """Generate search queries from the list of sources."""
    pattern = r"\\(.*?)\."
    searches = [re.findall(pattern, source) for source in sources]
    # Flatten list and remove duplicates
    return list(set(element for sublist in searches for element in sublist))


def parse_urls(search_results: str) -> list:
    """Extract URLs from the search results."""
    pattern = r"https://[^\]]+"
    return re.findall(pattern, search_results)


def remove_specific_string_from_list(nested_list: list, string_to_remove: str) -> list:
    """
    Removes a specific string from all elements of a nested list.

    :param nested_list: List of lists containing strings.
    :param string_to_remove: String to be removed from each element.
    :return: A new nested list with the specific string removed from each element.
    """
    return [
        [element.replace(string_to_remove, "") for element in sublist]
        for sublist in nested_list
    ]


def remove_specific_element_from_list(
    nested_list: list, element_to_remove: str
) -> list:
    """
    Removes a specific element from all sublists of a nested list.

    :param nested_list: List of lists containing elements.
    :param element_to_remove: Element to be removed from the lists.
    :return: A new nested list with the specific element removed.
    """
    return [
        [element for element in sublist if element != element_to_remove]
        for sublist in nested_list
    ]


def get_answer(query: str) -> str:
    """Generate an answer based on similar documents and the provided query."""
    similar_docs = get_similar_docs(index, query)
    sources = get_sources(similar_docs)
    queries = get_search_query(sources)
    url_list = [parse_urls(search.run(f"{link}")) for link in queries]
    string_to_remove = "/enterprise-server@3.6"  # No reason to look at enterprise docs, we just want to look at free version docs
    updated_list = remove_specific_string_from_list(url_list, string_to_remove)
    url_to_remove = "https://playrusvulkan.org/tortoise-git-quick-guide"  # found a dead link, no need to keep including it in the url list
    clean_url_list = remove_specific_element_from_list(updated_list, url_to_remove)

    time.sleep(
        60.0 / MODEL_REQUEST_LIMIT_PER_MINUTE
    )  # Implement a mandatory sleep time for each request before passing to LLM (this controls hitting request limits)

    # figure out how to implement 60K token rate limit so we don't go over and cause errors
    print(
        prompt.format(
            human_input=query,
            context=similar_docs,
            chat_history=memory.load_memory_variables({}),
            url_sources=clean_url_list,
        )
    )

    answer = qa_llm.run(
        {
            "context": similar_docs,
            "human_input": query,
            "chat_history": memory.load_memory_variables({}),
            "url_sources": clean_url_list,
        }
    )
    return answer


print(get_answer("What is Git?"))
