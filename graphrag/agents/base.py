"""
Base agent class for GraphRAG framework.
Provides common functionality for all agent implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for GraphRAG agents."""
    
    def __init__(self, graph: nx.DiGraph, model: Any):
        """
        Initialize base agent.
        
        Args:
            graph: NetworkX graph with event data
            model: Language model instance
        """
        self.graph = graph
        self.model = model
        self.tools = self._setup_tools()
        
    @abstractmethod
    def _setup_tools(self) -> List:
        """Setup tools for the agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run(self, query: str, actor: str = "", recipient: str = "", date: str = "") -> Dict[str, Any]:
        """
        Run the agent with a query. Must be implemented by subclasses.
        
        Args:
            query: The question/task for the agent
            actor: Actor name (optional)
            recipient: Recipient name (optional)
            date: Date for temporal context (optional)
            
        Returns:
            Dictionary with agent response and metadata
        """
        pass
    
    def validate_inputs(self, query: str, actor: str, recipient: str, date: str) -> Dict[str, Any]:
        """
        Validate agent inputs.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        if not query or not query.strip():
            issues.append("Query cannot be empty")
        
        if date:
            try:
                import pandas as pd
                pd.to_datetime(date)
            except ValueError:
                issues.append(f"Invalid date format: {date}. Use YYYY-MM-DD format")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the loaded graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": self._get_node_type_counts(),
            "edge_types": self._get_edge_type_counts()
        }
    
    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get count of nodes by type."""
        type_counts = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _get_edge_type_counts(self) -> Dict[str, int]:
        """Get count of edges by relation type."""
        type_counts = {}
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("relation", "unknown")
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts