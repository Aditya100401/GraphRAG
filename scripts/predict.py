#!/usr/bin/env python3
"""
CLI script for making single predictions with GraphRAG agents.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import graphrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.graph import GraphSerializer
from graphrag.agents import LangGraphAgent
from config.settings import settings

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_model(model_name: str = None, temperature: float = None):
    """Get the language model instance."""
    from langchain_openai import ChatOpenAI
    
    model_name = model_name or settings.DEFAULT_MODEL
    temperature = temperature if temperature is not None else settings.MODEL_TEMPERATURE
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=settings.MAX_TOKENS
    )

def main():
    parser = argparse.ArgumentParser(
        description='Make event predictions using GraphRAG agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple prediction
  python scripts/predict.py --graph-path data/graphs/graph_IND_train.pkl \\
    --query "What are the likely follow-up events between India and Pakistan?" \\
    --actor "India" --recipient "Pakistan" --date "2024-01-15"
  
  # Interactive mode
  python scripts/predict.py --graph-path data/graphs/graph_IND_train.pkl --interactive
        """
    )
    
    parser.add_argument('--graph-path', '-g', type=str, required=True,
                        help='Path to the knowledge graph file (.pkl)')
    parser.add_argument('--query', '-q', type=str,
                        help='Prediction query')
    parser.add_argument('--actor', '-a', type=str, default="",
                        help='Actor name for context')
    parser.add_argument('--recipient', '-r', type=str, default="",
                        help='Recipient name for context')
    parser.add_argument('--date', '-d', type=str, default="",
                        help='Date for temporal context (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default=settings.DEFAULT_MODEL,
                        help='Model name to use')
    parser.add_argument('--temperature', type=float, default=settings.MODEL_TEMPERATURE,
                        help='Model temperature')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--show-reasoning', action='store_true',
                        help='Show agent reasoning steps')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if not args.interactive and not args.query:
        parser.error("Must specify --query or use --interactive mode")
    
    try:
        # Load graph
        logger.info(f"Loading graph from {args.graph_path}")
        serializer = GraphSerializer()
        graph = serializer.load_graph(Path(args.graph_path).stem, format='pkl')
        
        # Get graph stats
        stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges()
        }
        print(f"Loaded graph: {stats['nodes']} nodes, {stats['edges']} edges")
        
        # Initialize model and agent
        logger.info(f"Initializing model: {args.model}")
        model = get_model(args.model, args.temperature)
        agent = LangGraphAgent(graph, model)
        
        def make_prediction(query: str, actor: str = "", recipient: str = "", date: str = ""):
            """Make a single prediction."""
            print(f"\nQuery: {query}")
            if actor or recipient or date:
                print(f"Context - Actor: {actor}, Recipient: {recipient}, Date: {date}")
            
            print("Thinking...")
            result = agent.run(query=query, actor=actor, recipient=recipient, date=date)
            
            if result.get('success', True):
                response = agent.get_last_response(result)
                predictions = agent.extract_predictions(result)
                
                print(f"\nAgent Response:")
                print("-" * 40)
                print(response)
                
                print(f"\nExtracted Predictions:")
                for i, pred in enumerate(predictions, 1):
                    if pred:
                        print(f"  {i}. {pred}")
                    else:
                        print(f"  {i}. (no prediction)")
                
                if args.show_reasoning:
                    print(f"\nReasoning Steps:")
                    print("-" * 40)
                    for i, message in enumerate(result.get('messages', []), 1):
                        role = getattr(message, 'role', type(message).__name__)
                        content = getattr(message, 'content', str(message))
                        print(f"Step {i} ({role}): {content[:200]}...")
                
                print(f"\nIterations: {result.get('iterations', 0)}")
                
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        if args.interactive:
            print("GraphRAG Interactive Prediction")
            print("=" * 40)
            print("Type 'quit' to exit")
            print("Use format: query|actor|recipient|date")
            print("Example: What will happen next?|India|Pakistan|2024-01-15")
            print()
            
            while True:
                try:
                    user_input = input("Enter prediction request: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    # Parse input
                    parts = user_input.split('|')
                    query = parts[0].strip()
                    actor = parts[1].strip() if len(parts) > 1 else ""
                    recipient = parts[2].strip() if len(parts) > 2 else ""
                    date = parts[3].strip() if len(parts) > 3 else ""
                    
                    make_prediction(query, actor, recipient, date)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        else:
            # Single prediction
            make_prediction(args.query, args.actor, args.recipient, args.date)
        
        print("\nDone!")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()