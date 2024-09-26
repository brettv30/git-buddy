import streamlit as st
from typing import Literal, Annotated
from utilities import repototxt
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langgraph.graph.message import add_messages
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.graph import END, START, StateGraph, MessagesState

class Config():
    def __init__(self):
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        self.cohere_api_key = st.secrets["COHERE_API_KEY"]
        self.openai_model = "gpt-4o-mini"
        self.cohere_model = "rerank-english-v3.0"
        self.embeddings_model = "text-embedding-ada-002"
        self.index_name = "git-buddy-index"
        self.retrieved_docs = 100
        self.top_docs = 5
        self.directory = "data"

    def initialize_retriever(self):
        embeddings = OpenAIEmbeddings(model=self.embeddings_model)
        docsearch = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name=self.index_name
        )
        retriever = docsearch.as_retriever(
            search_type="mmr", search_kwargs={"k": self.retrieved_docs}
        )
        compressor = CohereRerank(model=self.cohere_model, top_n=self.top_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def initialize_search(self):
        DDRSearch = DuckDuckGoSearchRun()
        return DDRSearch
    
    def initialize_llm(self, tools):
        llm_with_tools = ChatOpenAI(model=self.openai_model, temperature=0).bind_tools(tools)
        return llm_with_tools


class GitBuddyToolkit(Config):
    config = Config()
    compression_retriever = config.initialize_retriever()
    ddr_search = config.initialize_search()

    def get_tools():
        tools = [GitBuddyToolkit.search, GitBuddyToolkit.document_retriever, GitBuddyToolkit.github_repo_summarization]
        return tools

    @tool
    def search(query: str):
        """Call to to find additional sources related to Git, GitHub, and TortoiseGit.
        Only use this tool when you need to include additional sources related to Git, GitHub, and TortoiseGit in your response to the user
        """
        return GitBuddyToolkit.ddr_search.run(query)


    @tool
    def document_retriever(query: str):
        """Call to retrieve documents related to Git, GitHub, and TortoiseGit Documentation.
        Use this tool every time you need to supplement your knowledge base with the ground facts related to Git, GitHub, and TortoiseGit.
        """
        return GitBuddyToolkit.compression_retriever.invoke(query)
    
    @tool
    def check_for_github_repository_link():
        """Call this tool on every user input to see if they shared a github repository link that we need to summarize"""
        github_repo_link_schema = {}
        # Create a quick prompt
        # Pass the schema and prompt to the LLM
        # Call the LLM and get the structured response
        # Return the response
        pass

    @tool
    def github_repo_summarization(query: str):
        """Call to summarize a github repository. The input should be a link to a github repository that the user submits
        Always use this tool when the user includes a link to a github repository in their message to you. 
        """
        repototxt.set_github_token()


class State(TypedDict):
    messages: Annotated[list, add_messages]

class GitBuddyAgentManager(GitBuddyToolkit, Config):
    config = Config()
    tools = GitBuddyToolkit.get_tools()
    tool_node = ToolNode(tools)
    llm_with_tools = config.initialize_llm(tools)

    def chatbot(state: State):
        messages = GitBuddyAgentManager.filter_messages(state["messages"])
        response = GitBuddyAgentManager.llm_with_tools.invoke(messages)
        return {"messages": response}


    # Define the method that determines whether to continue or not
    def should_continue(state: MessagesState) -> Literal["tools", END]:  # type: ignore
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    # Define the method that determines the amount of messages to keep in history
    def filter_messages(messages: list):
        return messages[-14:]


    def set_graph_agent():
        workflow = StateGraph(State)

        workflow.add_node("agent", GitBuddyAgentManager.chatbot)
        workflow.add_node("tools", GitBuddyAgentManager.tool_node)

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent", GitBuddyAgentManager.should_continue, {"continue": "tools", "end": END}
        )

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
        )

        workflow.add_edge("tools", "agent")

        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()

        return workflow.compile(checkpointer=checkpointer)