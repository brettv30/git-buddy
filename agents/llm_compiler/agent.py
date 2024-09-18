from typing import Sequence, List, Dict, Any, Union, Iterable
from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
import sys
import os
# Append the repo directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from output_parser import LLMCompilerPlanParser, Task
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing_extensions import TypedDict
import itertools
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated

class SearchTool:
    @staticmethod
    def search(query: str) -> str:
        """
        Perform a search on DuckDuckGo with the input query string.
        
        Args:
            query (str): The search query string.
        
        Returns:
            str: The search results from DuckDuckGo.
        """
        search_tool = DuckDuckGoSearchRun()
        return search_tool.run(query)

class LLMCompilerAgent:
    def __init__(self):
        self.prompt = hub.pull("wfh/llm-compiler")
        self.joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.tools = self.create_tools()
        self.planner = self.create_planner()
        self.chain = self.create_chain()

    def create_tools(self):
        return [SearchTool.search]

    def create_planner(self):
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.__doc__}\n" for i, tool in enumerate(self.tools)
        )
        planner_prompt = self.prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )
        replanner_prompt = self.prompt.partial(
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

        def should_replan(state: list):
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}

        return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | self.llm
            | LLMCompilerPlanParser(tools=self.tools)
        )

    def _get_observations(self, messages: List[BaseMessage]) -> Dict[int, Any]:
        results = {}
        for message in messages[::-1]:
            if isinstance(message, FunctionMessage):
                results[int(message.additional_kwargs["idx"])] = message.content
        return results

    def _execute_task(self, task, observations, config):
        tool_to_use = task["tool"]
        if isinstance(tool_to_use, str):
            return tool_to_use
        args = task["args"]
        try:
            if isinstance(args, str):
                resolved_args = self._resolve_arg(args, observations)
            elif isinstance(args, dict):
                resolved_args = {
                    key: self._resolve_arg(val, observations) for key, val in args.items()
                }
            else:
                resolved_args = args
        except Exception as e:
            return (
                f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
                f" Args could not be resolved. Error: {repr(e)}"
            )
        try:
            return tool_to_use.invoke(resolved_args, config)
        except Exception as e:
            return (
                f"ERROR(Failed to call {tool_to_use.name} with args {args}."
                + f" Args resolved to {resolved_args}. Error: {repr(e)})"
            )

    def _resolve_arg(self, arg: Union[str, Any], observations: Dict[int, Any]):
        ID_PATTERN = r"\$\{?(\d+)\}?"

        def replace_match(match):
            idx = int(match.group(1))
            return str(observations.get(idx, match.group(0)))

        if isinstance(arg, str):
            return re.sub(ID_PATTERN, replace_match, arg)
        elif isinstance(arg, list):
            return [self._resolve_arg(a, observations) for a in arg]
        else:
            return str(arg)

    @as_runnable
    def schedule_task(self, task_inputs, config):
        task: Task = task_inputs["task"]
        observations: Dict[int, Any] = task_inputs["observations"]
        try:
            observation = self._execute_task(task, observations, config)
        except Exception:
            import traceback
            observation = traceback.format_exception()
        observations[task["idx"]] = observation

    def schedule_pending_task(self, task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
        while True:
            deps = task["dependencies"]
            if deps and (any([dep not in observations for dep in deps])):
                time.sleep(retry_after)
                continue
            self.schedule_task.invoke({"task": task, "observations": observations})
            break

    @as_runnable
    def schedule_tasks(self, scheduler_input: Dict):
        tasks = scheduler_input["tasks"]
        args_for_tasks = {}
        messages = scheduler_input["messages"]
        observations = self._get_observations(messages)
        task_names = {}
        originals = set(observations)
        futures = []
        retry_after = 0.25
        with ThreadPoolExecutor() as executor:
            for task in tasks:
                deps = task["dependencies"]
                task_names[task["idx"]] = (
                    task["tool"] if isinstance(task["tool"], str) else task["tool"].name
                )
                args_for_tasks[task["idx"]] = task["args"]
                if deps and (any([dep not in observations for dep in deps])):
                    futures.append(
                        executor.submit(
                            self.schedule_pending_task, task, observations, retry_after
                        )
                    )
                else:
                    self.schedule_task.invoke(dict(task=task, observations=observations))
            wait(futures)
        new_observations = {
            k: (task_names[k], args_for_tasks[k], observations[k])
            for k in sorted(observations.keys() - originals)
        }
        tool_messages = [
            FunctionMessage(
                name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}, tool_call_id = k
            )
            for k, (name, task_args, obs) in new_observations.items()
        ]
        return tool_messages

    @as_runnable
    def plan_and_schedule(self, state):
        messages = state["messages"]
        tasks = self.planner.stream(messages)
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            tasks = iter([])
        scheduled_tasks = self.schedule_tasks.invoke(
            {
                "messages": messages,
                "tasks": tasks,
            }
        )
        return {"messages": scheduled_tasks}

    def _parse_joiner_output(self, decision: Any) -> Dict[str, List[BaseMessage]]:
        response = [AIMessage(content=f"Thought: {decision.thought}")]
        if isinstance(decision.action, self.Replan):
            return {"messages": response + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
            }
        else:
            return {"messages": response + [AIMessage(content=decision.action.response)]}

    def select_recent_messages(self, state) -> dict:
        messages = state["messages"]
        selected = []
        for msg in messages[::-1]:
            selected.append(msg)
            if isinstance(msg, HumanMessage):
                break
        return {"messages": selected[::-1]}

    def create_chain(self):
        class FinalResponse(BaseModel):
            response: str

        class Replan(BaseModel):
            feedback: str = Field(
                description="Analysis of the previous attempts and recommendations on what needs to be fixed."
            )

        class JoinOutputs(BaseModel):
            thought: str = Field(
                description="The chain of thought reasoning for the selected action"
            )
            action: Union[FinalResponse, Replan]

        self.Replan = Replan  # Store for later use

        llm = ChatOpenAI(model="gpt-4o")
        runnable = self.joiner_prompt | llm.with_structured_output(JoinOutputs)
        joiner = self.select_recent_messages | runnable | self._parse_joiner_output

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph_builder = StateGraph(State)
        graph_builder.add_node("plan_and_schedule", self.plan_and_schedule)
        graph_builder.add_node("join", joiner)
        graph_builder.add_edge("plan_and_schedule", "join")

        def should_continue(state):
            messages = state["messages"]
            if isinstance(messages[-1], AIMessage):
                return END
            return "plan_and_schedule"

        graph_builder.add_conditional_edges(
            "join",
            should_continue,
        )
        graph_builder.add_edge(START, "plan_and_schedule")
        return graph_builder.compile()

    def stream(self, input_message: str, session_id: str):
        return self.chain.stream(
            {"messages": [HumanMessage(content=input_message)]}
        )

    def run(self, input_message: str, session_id: str):
        return self.chain.invoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

# Usage
if __name__ == "__main__":
    agent = LLMCompilerAgent()
    for step in agent.stream("What is Git?", "test_session"):
        print(step)
        print("---")