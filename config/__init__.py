"""Configuration module for GraphRAG framework."""

from .settings import settings, get_settings
from .config import optimized_settings

__all__ = ["settings", "get_settings", "optimized_settings"]