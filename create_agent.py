from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import logging
from functools import partial
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from dotenv import load_dotenv

from utils.prompt_loader import load_prompt
from tools.tools import get_tools

# Configure logging
logging.basicConfig(level=logging.INFO)

PROMPT = load_prompt()
load_dotenv()


MAX_ITERATIONS = 5

class MessageState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    actor: str
    recipient: str
    date: str
    iteration: int


def should_continue(state: MessageState):
    """
    Determines whether to continue looping or stop
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: #type: ignore
        return END
    return "tools"


def call_model(model, state: MessageState) -> MessageState:
    """
    Calls the model with the current state, increments the iteration counter,
    and returns the updated state.
    """
    try:
        messages = state["messages"]
        full_messages = [PROMPT] + messages # type: ignore
        response = model.invoke(full_messages)
        logging.debug(f"Model response: {response}")
        # Update the iteration counter.
        current_iteration = state.get("iteration", 0)
        state["iteration"] = current_iteration + 1
        state["messages"] = messages + [response] # type: ignore
        return state
    except GraphRecursionError:
        logging.error("Recursion error occurred. Please check the graph structure.")
        return state


def create_workflow(graph, model):
    """
    Creates a workflow with a loop that continues until the user decides to stop.
    Accepts a loaded graph object for the tools and a pre-initialized model.
    """
    tools = get_tools(graph)
    tool_node = ToolNode(tools)

    model_with_tools = model.bind_tools(tools)

    workflow = StateGraph(MessageState)
    workflow.add_node("agent", partial(call_model, model_with_tools))
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()
    return app

def run_agent(query: str, graph, model, actor: str = "", recipient: str = "", date: str = ""):
    """
    Run the agent with a query and optional context parameters.
    Loads the graph from the given path.
    """
    workflow = create_workflow(graph, model)

    # Log the type and content of query
    logging.info(f"Type of query: {type(query)}")

    # Ensure query is a string
    if isinstance(query, list):
        query = " ".join(str(item) for item in query)

    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "actor": actor,
        "recipient": recipient,
        "date": date,
        "iteration": 0
    }

    # Run the workflow
    result = workflow.invoke(initial_state)
    return result
