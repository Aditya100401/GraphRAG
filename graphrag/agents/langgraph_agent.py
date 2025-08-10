"""
LangGraph-based agent implementation.
Based on the original create_agent.py implementation.
"""

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Dict, Any, List, Optional
from functools import partial
import logging
import yaml
from pathlib import Path

from .base import BaseAgent
from .tools import create_graph_tools
from config.settings import settings

logger = logging.getLogger(__name__)

class MessageState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    actor: str
    recipient: str
    date: str
    iteration: int

class LangGraphAgent(BaseAgent):
    """LangGraph-based agent for event prediction."""
    
    def __init__(self, graph, model, max_iterations: int = 0, prompt_path: Optional[str] = None):
        """
        Initialize LangGraph agent.
        
        Args:
            graph: NetworkX graph with event data
            model: Language model instance
            max_iterations: Maximum number of agent iterations
            prompt_path: Optional path to system prompt file
        """
        self.max_iterations = max_iterations or settings.MAX_ITERATIONS
        self.system_prompt = self._load_system_prompt(prompt_path)
        super().__init__(graph, model)
        self.workflow = self._create_workflow()
        
    def _setup_tools(self) -> List:
        """Setup graph-based tools for the agent."""
        return create_graph_tools(self.graph)
    
    def _load_system_prompt(self, prompt_path: Optional[str] = None) -> BaseMessage:
        """
        Load system prompt from configuration.
        
        Args:
            prompt_path: Optional path to the prompt file. If not provided,
                        uses the default path from settings.
        """
        if prompt_path is None:
            prompt_path = str(Path(settings.PROJECT_ROOT) / "config" / "system_prompt.yaml")
        
        try:
            with open(prompt_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
                prompt_text = prompt_data.get('template', '')
                return HumanMessage(content=prompt_text, name="system")
        except Exception as e:
            logger.error(f"Error loading system prompt from {prompt_path}: {e}")
            # Fallback prompt
            return HumanMessage(content="You are an expert geopolitical event prediction agent.", name="system")
    
    def _should_continue(self, state: MessageState) -> str:
        """Determine whether to continue looping or stop."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check iteration limit
        if state.get("iteration", 0) >= self.max_iterations:
            logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
            return END
        
        # Check if agent wants to use tools
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls: # type: ignore
            return END
        
        return "tools"
    
    def _call_model(self, state: MessageState) -> MessageState:
        """Call the model with the current state and update iteration counter."""
        try:
            messages = state["messages"]
            full_messages = [self.system_prompt] + messages # type: ignore
            response = self.model.invoke(full_messages)
            
            logger.debug(f"Model response: {response}")
            
            # Update the iteration counter
            current_iteration = state.get("iteration", 0)
            state["iteration"] = current_iteration + 1
            state["messages"] = messages + [response] # type: ignore
            
            return state
            
        except GraphRecursionError:
            logger.error("Recursion error occurred. Please check the graph structure.")
            return state
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return state
    
    def _create_workflow(self):
        """Create the LangGraph workflow."""
        tool_node = ToolNode(self.tools)
        
        workflow = StateGraph(MessageState)
        workflow.add_node("agent", partial(self._call_model))
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run(self, query: str, actor: str = "", recipient: str = "", date: str = "") -> Dict[str, Any]:
        """
        Run the agent with a query and optional context parameters.
        
        Args:
            query: The question/task for the agent
            actor: Actor name for context
            recipient: Recipient name for context
            date: Date for temporal context
            
        Returns:
            Dictionary with agent response and metadata
        """
        # Validate inputs
        validation = self.validate_inputs(query, actor, recipient, date)
        if not validation["valid"]:
            return {
                "error": "Input validation failed",
                "issues": validation["issues"],
                "messages": [],
                "iterations": 0
            }
        
        # Log the type and content of query
        logger.info(f"Type of query: {type(query)}")
        
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
        
        try:
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            return {
                "messages": result.get("messages", []),
                "actor": result.get("actor", actor),
                "recipient": result.get("recipient", recipient),
                "date": result.get("date", date),
                "iterations": result.get("iteration", 0),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error running agent workflow: {e}")
            return {
                "error": str(e),
                "messages": [],
                "actor": actor,
                "recipient": recipient,
                "date": date,
                "iterations": 0,
                "success": False
            }
    
    def get_last_response(self, result: Dict[str, Any]) -> str:
        """Extract the last response content from agent result."""
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            return getattr(last_message, 'content', str(last_message))
        return ""
    
    def extract_predictions(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract event type predictions from agent result.
        
        Args:
            result: Agent result dictionary
            
        Returns:
            List of predicted event types
        """
        import re
        
        last_response = self.get_last_response(result)
        
        # Try to extract predictions from "Answer:" format
        answer_patterns = [
            r"Answer:\s*([A-Z,\s]+)",  # Answer: EVENT1, EVENT2, EVENT3
            r"Answer:\s*\[([^\]]+)\]",  # Answer: [EVENT1, EVENT2, EVENT3]
            r"most likely.*?:\s*([A-Z,\s]+)",  # "most likely are: EVENT1, EVENT2"
            r"predictions?.*?:\s*([A-Z,\s]+)",  # "predictions: EVENT1, EVENT2"
            r"(?:top|first)\s*3.*?:\s*([A-Z,\s]+)",  # "top 3 are: EVENT1, EVENT2"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, last_response, re.IGNORECASE | re.DOTALL)
            if match:
                answer_text = match.group(1)
                # Split by comma and clean up
                parts = [p.strip().strip('[]').strip().upper() for p in answer_text.split(",")]
                parts = [p for p in parts if p and p in settings.EVENT_TYPES]  # Filter to valid event types
                if parts:  # Only return if we found valid predictions
                    return (parts + ["", "", ""])[:3]
        
        # Last resort: extract any event types mentioned in the response
        found_events = []
        for event_type in settings.EVENT_TYPES:
            if event_type in last_response.upper():
                found_events.append(event_type)
        
        if found_events:
            return (found_events + ["", "", ""])[:3]
        
        logger.warning(f"Could not extract predictions from response: {last_response[:200]}...")
        return ["", "", ""]