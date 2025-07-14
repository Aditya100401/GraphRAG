"""
Custom Agent Example: How to create and customize GraphRAG agents.

This example shows how to:
1. Create custom agents with different configurations
2. Add custom tools to agents
3. Customize the evaluation process
4. Compare different agent setups
"""

import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.agents.base import BaseAgent
from graphrag.agents import LangGraphAgent
from graphrag.agents.tools import create_graph_tools
from graphrag.graph import GraphSerializer
from graphrag.evaluation import Evaluator
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomGraphAgent(BaseAgent):
    """
    Custom agent implementation with additional features.
    """
    
    def __init__(self, graph, model, custom_tools: List = None):
        """Initialize custom agent with optional custom tools."""
        self.custom_tools = custom_tools or []
        super().__init__(graph, model)
    
    def _setup_tools(self) -> List:
        """Setup tools including custom ones."""
        base_tools = create_graph_tools(self.graph)
        return base_tools + self.custom_tools
    
    def run(self, query: str, actor: str = "", recipient: str = "", date: str = "") -> Dict[str, Any]:
        """Simple implementation for demonstration."""
        # This is a simplified implementation - in practice you'd implement
        # the full agent logic here
        return {
            "response": f"Custom agent response for: {query}",
            "predictions": ["THREATEN", "ACCUSE", "PROTEST"],
            "success": True,
            "custom_feature": "This agent has custom capabilities!"
        }

def create_custom_tools():
    """Create some custom tools for demonstration."""
    
    @tool("get_conflict_intensity")
    def get_conflict_intensity(actor: str, recipient: str) -> str:
        """
        Analyze the overall conflict intensity between two actors.
        Returns a qualitative assessment of their relationship.
        """
        # This is a mock implementation - in practice you'd analyze the graph
        intensity_map = {
            ("India", "Pakistan"): "High - ongoing territorial disputes",
            ("Russia", "Ukraine"): "Very High - active conflict",
            ("USA", "China"): "Medium - trade tensions"
        }
        
        key = (actor, recipient)
        reverse_key = (recipient, actor)
        
        return intensity_map.get(key, intensity_map.get(reverse_key, "Low - minimal tensions"))
    
    @tool("get_media_sentiment")
    def get_media_sentiment(actor: str, recipient: str, date: str) -> str:
        """
        Get media sentiment analysis for actor-recipient interactions.
        Returns sentiment score and summary.
        """
        # Mock implementation
        return f"Media sentiment between {actor} and {recipient} around {date}: Negative (-0.3). Recent coverage focuses on diplomatic tensions."
    
    return [get_conflict_intensity, get_media_sentiment]

def compare_agents():
    """Compare different agent configurations."""
    
    print("Custom Agent Example: Agent Comparison")
    print("=" * 50)
    
    # Try to load a graph
    try:
        serializer = GraphSerializer()
        
        # Try to load an existing graph
        try:
            graph = serializer.load_graph("quickstart_example", "pkl")
            print(f"✓ Loaded graph: {graph.number_of_nodes()} nodes")
        except:
            print("⚠ No existing graph found. Run quickstart.py first to create a graph.")
            return
        
        # Initialize model
        try:
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            print("✓ Model initialized")
        except Exception as e:
            print(f"⚠ Could not initialize model: {e}")
            print("This example requires an OpenAI API key.")
            return
        
        # 1. Standard LangGraph Agent
        print("\n1. Standard LangGraph Agent")
        print("-" * 30)
        standard_agent = LangGraphAgent(graph, model, max_iterations=3)
        
        query = "What are the likely follow-up events between India and Pakistan?"
        result1 = standard_agent.run(query, actor="India", recipient="Pakistan", date="2024-01-15")
        
        if result1.get('success', True):
            predictions1 = standard_agent.extract_predictions(result1)
            print(f"Standard agent predictions: {predictions1}")
            print(f"Iterations: {result1.get('iterations', 0)}")
        
        # 2. Custom Agent with Additional Tools
        print("\n2. Custom Agent with Additional Tools")
        print("-" * 40)
        custom_tools = create_custom_tools()
        custom_agent = CustomGraphAgent(graph, model, custom_tools=custom_tools)
        
        result2 = custom_agent.run(query, actor="India", recipient="Pakistan", date="2024-01-15")
        print(f"Custom agent response: {result2.get('response', 'No response')}")
        print(f"Custom feature: {result2.get('custom_feature', 'None')}")
        
        # 3. Different Model Configurations
        print("\n3. Different Model Configurations")
        print("-" * 35)
        
        # High temperature agent (more creative)
        creative_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        creative_agent = LangGraphAgent(graph, creative_model, max_iterations=2)
        
        result3 = creative_agent.run(query, actor="India", recipient="Pakistan", date="2024-01-15")
        if result3.get('success', True):
            predictions3 = creative_agent.extract_predictions(result3)
            print(f"Creative agent predictions: {predictions3}")
        
        # 4. Evaluation Comparison
        print("\n4. Evaluation Comparison")
        print("-" * 25)
        
        # Create sample test data
        import pandas as pd
        test_data = pd.DataFrame({
            'Event ID': [1, 2],
            'Actor Name': ['India', 'Pakistan'],
            'Recipient Name': ['Pakistan', 'India'],
            'Event Type': ['THREATEN', 'ACCUSE'],
            'Event Date': ['2024-01-15', '2024-01-16'],
            'Event Intensity': [-3, -2],
            'Contexts': ['Border dispute', 'Military exercise'],
            'Raw Placename': ['Kashmir', 'Punjab'],
            'Quad Code': ['14', '15']
        })
        
        # Quick evaluation (just 1 sample for demo)
        evaluator1 = Evaluator(standard_agent, sleep_time=0.1)
        results1 = evaluator1.evaluate_agent(test_data.head(1), debug=False)
        
        print(f"Standard agent ROUGE-1: {results1['rouge_scores']['rank1']:.3f}")
        
        # 5. Custom Evaluation Metrics
        print("\n5. Custom Evaluation Metrics")
        print("-" * 30)
        
        def custom_evaluation_metric(predictions: List[str], ground_truth: str) -> float:
            """Custom metric that rewards predictions in the same threat category."""
            threat_events = {'THREATEN', 'ASSAULT', 'COERCE', 'SANCTION'}
            coop_events = {'COOPERATE', 'AID', 'CONSULT', 'CONCEDE'}
            
            gt_category = 'threat' if ground_truth in threat_events else 'coop' if ground_truth in coop_events else 'neutral'
            
            score = 0
            for pred in predictions:
                pred_category = 'threat' if pred in threat_events else 'coop' if pred in coop_events else 'neutral'
                if pred_category == gt_category:
                    score += 0.5  # Partial credit for same category
                if pred == ground_truth:
                    score += 1.0  # Full credit for exact match
                    
            return min(score, 1.0)  # Cap at 1.0
        
        # Apply custom metric
        sample_predictions = ["THREATEN", "ASSAULT", "COOPERATE"]
        sample_ground_truth = "THREATEN"
        custom_score = custom_evaluation_metric(sample_predictions, sample_ground_truth)
        print(f"Custom metric score: {custom_score:.3f}")
        
        print("\n" + "=" * 50)
        print("Custom Agent Example completed! 🎉")
        print("\nKey takeaways:")
        print("- You can extend BaseAgent to create custom agent implementations")
        print("- Custom tools can be added to enhance agent capabilities")
        print("- Different model configurations affect prediction behavior")
        print("- Custom evaluation metrics can capture domain-specific requirements")
        
    except Exception as e:
        logger.error(f"Error in custom agent example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_agents()