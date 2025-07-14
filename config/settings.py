"""
Configuration management for GraphRAG framework.
Centralizes all settings and provides environment-based configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Centralized configuration management for GraphRAG."""
    
    def __init__(self, config_name: str = "default"):
        self.config_name = config_name
        self._load_base_config()
        self._load_environment_overrides()
    
    def _load_base_config(self):
        """Load base configuration settings."""
        
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.GRAPHS_DIR = self.DATA_DIR / "graphs"
        self.OUTPUTS_DIR = self.PROJECT_ROOT / "outputs"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Create directories if they don't exist
        for dir_path in [self.DATA_DIR, self.GRAPHS_DIR, self.OUTPUTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # API Keys and External Services
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        
        # Model Configuration
        self.DEFAULT_MODEL = "gpt-4o-mini"
        self.MODEL_TEMPERATURE = 0
        self.MAX_TOKENS = 2000
        self.MAX_ITERATIONS = 5
        
        # Event Types
        self.EVENT_TYPES = [
            "ACCUSE", "ASSAULT", "AID", "REQUEST", "PROTEST", "COERCE", "THREATEN",
            "RETREAT", "MOBILIZE", "SANCTION", "CONCEDE", "COOPERATE",
            "CONSULT", "REJECT"
        ]
        
        # Data Processing Configuration
        self.MIN_ENTITY_FREQUENCY = 5
        self.MIN_RELATION_FREQUENCY = 10
        self.TEST_SPLIT_RATIO = 0.2
        self.TEMPORAL_GAP_DAYS = 30
        
        # Graph Configuration
        self.GRAPH_FORMAT = "pkl"  # Options: pkl, graphml, both
        self.INCLUDE_TEMPORAL_EDGES = True
        self.MAX_NEIGHBORS = 50
        
        # Agent Configuration
        self.AGENT_TYPE = "langgraph"  # Options: react, langgraph
        self.TOOL_TIMEOUT = 30
        self.MAX_TOOL_CALLS = 10
        
        # Evaluation Configuration
        self.EVALUATION_SLEEP_TIME = 1.0
        self.ROUGE_USE_STEMMER = True
        self.METRICS_PRECISION = 4
        
        # Countries for analysis
        self.COUNTRIES = ["AFG", "IND", "RUS"]
        
    def _load_environment_overrides(self):
        """Load environment-specific overrides."""
        
        # Allow environment variables to override settings
        env_overrides = {
            "DEFAULT_MODEL": os.getenv("GRAPHRAG_MODEL", self.DEFAULT_MODEL),
            "MODEL_TEMPERATURE": float(os.getenv("GRAPHRAG_TEMPERATURE", str(self.MODEL_TEMPERATURE))),
            "AGENT_TYPE": os.getenv("GRAPHRAG_AGENT_TYPE", self.AGENT_TYPE),
            "EVALUATION_SLEEP_TIME": float(os.getenv("GRAPHRAG_SLEEP_TIME", str(self.EVALUATION_SLEEP_TIME))),
        }
        
        for key, value in env_overrides.items():
            setattr(self, key, value)
    
    def get_graph_path(self, country: str, split: str = "train") -> Path:
        """Get path for country-specific graph file."""
        return self.GRAPHS_DIR / f"graph_{country}_{split}.pkl"
    
    def get_data_path(self, country: str, split: str = "train") -> Path:
        """Get path for country-specific data file."""
        return self.DATA_DIR / "final_splits" / f"{split}_{country}.csv"
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for output files."""
        return self.OUTPUTS_DIR / filename
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present."""
        return {
            "openai": bool(self.OPENAI_API_KEY),
            "huggingface": bool(self.HUGGINGFACEHUB_API_TOKEN),
            "news": bool(self.NEWS_API_KEY),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

# Global settings instance
settings = Settings()

def get_settings(config_name: str = "default") -> Settings:
    """Get settings instance with optional config name."""
    return Settings(config_name)