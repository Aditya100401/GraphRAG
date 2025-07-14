"""Agents module for GraphRAG framework."""

from .base import BaseAgent
from .langgraph_agent import LangGraphAgent

__all__ = ["BaseAgent", "LangGraphAgent"]