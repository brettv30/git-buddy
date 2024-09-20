from typing import Literal
import time
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever

DDRSearch = DuckDuckGoSearchRun()

openai_model = "gpt-4o-mini"
cohere_model = "rerank-english-v3.0"
embeddings_model = "text-embedding-ada-002"
index_name = "git-buddy-index"
retrieved_docs = 100
top_docs = 5

embeddings = OpenAIEmbeddings(model=embeddings_model)
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name=index_name
)
retriever = docsearch.as_retriever(
    search_type="mmr", search_kwargs={"k": retrieved_docs}
)
compressor = CohereRerank(model=cohere_model, top_n=top_docs)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def search(query: str):
    """Call to surf the web"""
    return DDRSearch.run(query)


@tool
def document_retriever(query: str):
    """Call to retrieve documents related to Git, GitHub, and TortoiseGit Documentation"""
    return compression_retriever.invoke(query)


tools = [search, document_retriever]

tool_node = ToolNode(tools)

llm_with_tools = ChatOpenAI(model=openai_model, temperature=0).bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]: # type: ignore
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def set_graph_agent():
    workflow = StateGraph(State)

    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )

    workflow.add_edge("tools", "agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


def stream_data(input: str):
    for word in input.split(" "):
        yield word + " "
        time.sleep(0.02)
