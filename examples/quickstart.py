"""
QuickStart Example: Basic usage of the GraphRAG framework.

This example demonstrates how to:
1. Load and process event data
2. Build a temporal knowledge graph
3. Create and use an agent for predictions
4. Evaluate agent performance
"""

import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.data import DataLoader, DataCleaner
from graphrag.graph import GraphBuilder, GraphSerializer
from graphrag.agents import LangGraphAgent
from graphrag.evaluation import Evaluator
from langchain_openai import ChatOpenAI
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the quickstart example."""
    
    print("GraphRAG Framework - QuickStart Example")
    print("=" * 50)
    
    # Step 1: Load data for a specific country
    print("\n1. Loading event data...")
    loader = DataLoader()
    
    # Load training data for India
    try:
        train_data = loader.load_country_data("IND", "train")
        test_data = loader.load_country_data("IND", "test")
        print(f"✓ Loaded {len(train_data)} training events and {len(test_data)} test events for India")
    except FileNotFoundError:
        print("⚠ Country data not found. Using sample data...")
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'Event ID': range(10),
            'Actor Name': ['India'] * 5 + ['Pakistan'] * 5,
            'Recipient Name': ['Pakistan'] * 5 + ['India'] * 5,
            'Event Type': ['THREATEN', 'ACCUSE', 'PROTEST', 'REJECT', 'CONSULT'] * 2,
            'Event Date': pd.date_range('2023-01-01', periods=10),
            'Event Intensity': [-3, -2, -4, -1, 1, -2, -3, -5, -1, 2],
            'Raw Placename': ['Kashmir'] * 10,
            'Contexts': ['Border dispute'] * 10,
            'Quad Code': ['14'] * 10
        })
        train_data = sample_data[:8]
        test_data = sample_data[8:]
        print(f"✓ Created sample data: {len(train_data)} training, {len(test_data)} test events")
    
    # Step 2: Build temporal knowledge graph
    print("\n2. Building temporal knowledge graph...")
    builder = GraphBuilder()
    graph = builder.build_temporal_kg(train_data)
    
    # Get graph statistics
    stats = builder.get_graph_statistics(graph)
    print(f"✓ Built graph with {stats['total_nodes']} nodes and {stats['total_edges']} edges")
    print(f"  Node types: {stats['node_types']}")
    
    # Step 3: Save and load graph
    print("\n3. Saving and loading graph...")
    serializer = GraphSerializer()
    saved_files = serializer.save_graph(graph, "quickstart_example", ["pkl"])
    print(f"✓ Graph saved: {saved_files}")
    
    # Load it back
    loaded_graph = serializer.load_graph("quickstart_example", "pkl")
    print(f"✓ Graph loaded: {loaded_graph.number_of_nodes()} nodes")
    
    # Step 4: Initialize agent
    print("\n4. Initializing GraphRAG agent...")
    
    # Initialize model (you need OpenAI API key in environment)
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = LangGraphAgent(loaded_graph, model)
        print("✓ Agent initialized successfully")
        
        # Step 5: Make a prediction
        print("\n5. Making a prediction...")
        query = """Given the historical patterns between India and Pakistan, what are the 3 most likely 
        follow-up event types that could occur between these actors?"""
        
        result = agent.run(
            query=query,
            actor="India",
            recipient="Pakistan", 
            date="2024-01-15"
        )
        
        if result.get('success', True):
            response = agent.get_last_response(result)
            predictions = agent.extract_predictions(result)
            
            print(f"✓ Prediction completed in {result.get('iterations', 0)} iterations")
            print(f"Agent response: {response[:200]}...")
            print(f"Extracted predictions: {predictions}")
        else:
            print(f"⚠ Prediction failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"⚠ Could not initialize agent (likely missing OpenAI API key): {e}")
        print("Skipping agent demonstration...")
    
    # Step 6: Evaluation (if we have an agent)
    try:
        if 'agent' in locals():
            print("\n6. Running evaluation...")
            evaluator = Evaluator(agent, sleep_time=0.1)  # Fast evaluation for demo
            
            # Evaluate on a small subset
            small_test = test_data.head(2)  # Just 2 samples for demo
            results = evaluator.evaluate_agent(small_test, debug=False)
            
            rouge_scores = results['rouge_scores']
            print(f"✓ Evaluation completed on {len(small_test)} samples")
            print(f"ROUGE-1 scores: Rank1={rouge_scores['rank1']:.3f}, Best={rouge_scores['best']:.3f}")
        
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
    
    # Step 7: Advanced usage example
    print("\n7. Advanced usage - Custom data processing...")
    
    # Data cleaning example
    cleaner = DataCleaner()
    stats = cleaner.get_cleaning_statistics(train_data, train_data)  # Same data for demo
    print(f"✓ Data cleaning stats: {stats['original_size']} → {stats['cleaned_size']} events")
    
    # Multiple country graphs
    builder = GraphBuilder()
    try:
        countries = ['AFG', 'RUS']  # Try other countries
        for country in countries:
            try:
                country_graph = builder.build_country_graph(country, 'train')
                country_stats = builder.get_graph_statistics(country_graph)
                print(f"✓ {country} graph: {country_stats['total_nodes']} nodes")
            except:
                print(f"⚠ Could not load {country} data")
    except Exception as e:
        print(f"⚠ Multi-country demo failed: {e}")
    
    print("\n" + "=" * 50)
    print("QuickStart completed! 🎉")
    print("\nNext steps:")
    print("- Set up your OpenAI API key for full agent functionality")
    print("- Try the CLI scripts: python scripts/build_graph.py --help")
    print("- Explore the examples/ directory for more use cases")
    print("- Read the documentation in docs/")

if __name__ == "__main__":
    main()