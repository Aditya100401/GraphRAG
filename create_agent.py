from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages  

from typing import Sequence, TypedDict, Annotated 
import operator
import os
import argparse

from tools.tools import get_node_edge_connections_tool, print_node_attributes_tool, calculate_event_type_frequency_tool, search_and_extract_news
from utils.create_prompt import PROMPT




tool_nodes = ToolNode([
    get_node_edge_connections_tool,
    print_node_attributes_tool,
    calculate_event_type_frequency_tool,
    search_and_extract_news
])

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.2
    )

MAX_ITERATIONS = 5

class State(TypedDict):
    """
        State to interpret agent's messages and context.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Changed to add_messages
    actor: str
    recipient: str
    date: str


def prediction_node(state: State) -> State:
    """
    Initial Prediction node to predict the type of event. 

    Args:
        state (State): The current state with messages and context

    Returns:
        State: Updated state with the model's prediction
    """
    
    messages = state["messages"]
    actor = state.get("actor", "")
    recipient = state.get("recipient", "")
    date = state.get("date", "")
    
    prompt = PROMPT.format(actor=actor, recipient=recipient, date=date)
    
    # Check if the user has already provided a prediction
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        system_message = SystemMessage(content=prompt)
        response = model.invoke([system_message] + messages)
    else:
        response = model.invoke(messages)
    
    return {"messages": [response], "actor": actor, "recipient": recipient, "date": date}


def feedback_node(state: State) -> State:
    """
    Feedback node to provide personalized feedback.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the agent is already reaching a conclusion
    if isinstance(last_message, AIMessage) and "Answer:" in last_message.content:
        return state
    
    # Create a system prompt for the feedback LLM
    feedback_system_prompt = """
    Based on the information gathered, continue your reasoning. Follow the Thought, Action, PAUSE format, or provide a final Answer if you have sufficient information.
    """
    
    # Use the model to generate personalized feedback
    feedback_messages = [
        SystemMessage(content=feedback_system_prompt),
        HumanMessage(content=f"Here is the conversation so far: {[m.content for m in messages]}")
    ]
    
    feedback_response = model.invoke(feedback_messages)
    
    # Add the feedback as a human message to guide the agent
    feedback_message = HumanMessage(content=feedback_response.content)
    
    return {
        "messages": [feedback_message],
        "actor": state["actor"],
        "recipient": state["recipient"],
        "date": state["date"]
    }

def get_num_iterations(state: State) -> int:
    """
    Count the number of iterations the agent has performed.
    An iteration is counted as a sequence of prediction -> tools -> feedback.
    
    Args:
        state (State): The current state with messages
        
    Returns:
        int: Number of iterations
    """
    # Count the number of AI messages as a proxy for iterations
    count = 0
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            count += 1
    return count


def event_loop(state: State) -> str:
    """
    Determine whether to continue the workflow loop or end based on iterations.
    Also checks for an Answer in AI messages.
    
    Args:
        state (State): The current state with messages and context
        
    Returns:
        str: 'prediction' to continue or 'end' to stop
    """
    # Check if max iterations reached
    if get_num_iterations(state) >= MAX_ITERATIONS:
        return "end"  
    
    # Check if an answer has been found
    messages = state["messages"]
    for message in reversed(messages):
        if isinstance(message, AIMessage) and "Answer:" in message.content:
            return "end"  
    
    # Continue the loop
    return "prediction"


def prediction_router(state: State) -> str:
    """
    Routes the flow based on the content of the last AI message.
    
    Args:
        state (State): The current state with messages
        
    Returns:
        str: Routing decision: "tool", "end", or ""
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage):
        return ""
    
    # Check for Answer
    if "Answer:" in last_message.content:
        return "end"
    
    # Check for Action
    if "Action:" in last_message.content:
        return "tool"
    
    # Default case
    return ""


def create_workflow():
    """
    Create and compile the LangGraph workflow with the defined nodes.
    
    Returns:
        The compiled workflow
    """
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("prediction", prediction_node)
    workflow.add_node("tools", tool_nodes)
    workflow.add_node("feedback", feedback_node)
    
    # Set entry point
    workflow.set_entry_point("prediction")
    
    # Connect prediction to tools or feedback using custom router
    workflow.add_conditional_edges(
        "prediction",
        prediction_router,  
        {
            "tool": "tools",     
            "end": END,         
            "": "feedback"      
        }
    )
    
    # Connect tools back to feedback
    workflow.add_edge("tools", "feedback")
    
    
    workflow.add_conditional_edges(
        "feedback",
        event_loop,
        {
            "prediction": "prediction",
            "end": END  
        }
    )
    
    # Compile the graph
    return workflow.compile()


def run_agent(query: str, actor: str = "", recipient: str = "", date: str = ""):
    """
    Run the agent with a query and optional context parameters.
    
    Args:
        query (str): The user query
        actor (str, optional): The actor name
        recipient (str, optional): The recipient name
        date (str, optional): The date in YYYY-MM-DD format
        
    Returns:
        The final state after workflow completion
    """
    workflow = create_workflow()
    
    # Check if the flowchart image already exists
    if not os.path.exists("flowchart.png"):
        png_data = workflow.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

        # Save the image to a file
        with open("flowchart.png", "wb") as f:
            f.write(png_data)
        print("Flowchart image generated.")
    else:
        print("Flowchart image already exists, skipping generation.")
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "actor": actor,
        "recipient": recipient,
        "date": date
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run the event prediction agent.')
    parser.add_argument('--query', '-q', type=str, required=True, 
                        help='The question to ask the agent (enclose in quotes if contains spaces)')
    parser.add_argument('--actor', '-a', type=str, default="", help='The actor name')
    parser.add_argument('--recipient', '-r', type=str, default="", help='The recipient name')
    parser.add_argument('--date', '-d', type=str, default="", help='The date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    result = run_agent(
        args.query,
        actor=args.actor,
        recipient=args.recipient,
        date=args.date
    )
    
# Print the complete conversation flow including thoughts, actions, and observations
final_messages = result["messages"]
print("\n=== FULL CONVERSATION FLOW ===\n")

# Track message numbers for better readability
message_counter = 1

for message in final_messages:
    # Print different message types with formatting for clarity
    if isinstance(message, HumanMessage):
        print(f"\n[{message_counter}] 👤 HUMAN: {message.content}\n")
    elif isinstance(message, AIMessage):
        print(f"\n[{message_counter}] 🤖 AI: {message.content}\n")
    else:
        print(f"\n[{message_counter}] MESSAGE ({type(message).__name__}): {message.content}\n")
    
    message_counter += 1
    print("-" * 80)  
    