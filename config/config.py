"""
Configuration settings for GraphRAG evaluation.
"""

from pathlib import Path
from config.settings import Settings

class OptimizedSettings(Settings):
    """Optimized configuration for batch evaluation."""
    
    def __init__(self, config_name: str = "optimized"):
        super().__init__(config_name)
        
        # Model Configuration - Use optimized model settings
        self.MODEL = "gpt-4o-mini"  # Efficient model choice
        self.MODEL_TEMPERATURE = 0  # Deterministic responses
        self.MAX_TOKENS = 1000  # Optimized response length
        self.MAX_ITERATIONS = 3  # Optimized iterations
        
        # Evaluation Configuration - Optimized for batch processing
        self.EVALUATION_SLEEP_TIME = 0.1  # Minimal sleep for rate limiting
        self.BATCH_SIZE = 10  # Process in smaller batches
        self.MAX_SAMPLES_PER_BATCH = 50  # Limit for testing
        
        # Tool Configuration - Optimized operations
        self.MAX_TOOL_CALLS = 5  # Optimized tool calls
        self.TOOL_TIMEOUT = 15  # Optimized timeout
        
        # Enable tracking
        self.TRACK_TOKENS = True
        self.TOKEN_LIMIT_PER_HOUR = 50000  # Set hourly limit
        self.COST_LIMIT_PER_RUN = 10.0  # Max cost in USD
        
    def get_prompt_variations_dir(self) -> Path:
        """Get directory for prompt variations."""
        prompt_dir = self.PROJECT_ROOT / "config" / "prompt_variations"
        prompt_dir.mkdir(exist_ok=True)
        return prompt_dir
    
    def get_cost_tracking_file(self) -> Path:
        """Get file for cost tracking."""
        return self.OUTPUTS_DIR / "cost_tracking.json"

# Global optimized settings
optimized_settings = OptimizedSettings()
