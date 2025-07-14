"""
GraphRAG: Agentic Reasoning for Social Event Extrapolation: Integrating Knowledge Graphs and Language Models

A framework for geopolitical event prediction using temporal knowledge graphs
and LLM agents with specialized tools.
"""

__version__ = "0.1.0"
__author__ = "Aditya Sampath"

# Core imports for easy access
from .data.loader import DataLoader
from .graph.builder import GraphBuilder
from .agents.base import BaseAgent
from .evaluation.evaluator import Evaluator

__all__ = [
    "DataLoader",
    "GraphBuilder", 
    "BaseAgent",
    "Evaluator",
]