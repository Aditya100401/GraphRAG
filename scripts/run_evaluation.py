#!/usr/bin/env python3
"""
CLI script for evaluating GraphRAG agents on event prediction tasks.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import graphrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.data import DataLoader
from graphrag.graph import GraphSerializer
from graphrag.agents import LangGraphAgent
from graphrag.evaluation import Evaluator
from config.settings import settings

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOGS_DIR / 'evaluate.log')
        ]
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
        description='Evaluate GraphRAG agents on event prediction tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on specific country
  python scripts/evaluate.py --country IND --graph-path data/graphs/graph_IND_train.pkl
  
  # Evaluate with custom test data
  python scripts/evaluate.py --test-data data/custom_test.csv --graph-path data/graphs/graph_IND_train.pkl
  
  # Full evaluation with metrics report
  python scripts/evaluate.py --country IND --output results/evaluation_IND.csv --metrics-report results/metrics_IND.txt
        """
    )
    
    parser.add_argument('--country', '-c', type=str,
                        help='Country code for evaluation (e.g., AFG, IND, RUS)')
    parser.add_argument('--graph-path', '-g', type=str, required=True,
                        help='Path to the knowledge graph file (.pkl)')
    parser.add_argument('--test-data', '-t', type=str,
                        help='Path to test CSV file (overrides country)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output CSV file with predictions')
    parser.add_argument('--metrics-report', '-m', type=str,
                        help='Output file for metrics report')
    parser.add_argument('--model', type=str, default=settings.DEFAULT_MODEL,
                        help='Model name to use')
    parser.add_argument('--temperature', type=float, default=settings.MODEL_TEMPERATURE,
                        help='Model temperature')
    parser.add_argument('--sleep-time', type=float, default=settings.EVALUATION_SLEEP_TIME,
                        help='Sleep time between predictions (seconds)')
    parser.add_argument('--max-samples', type=int,
                        help='Maximum number of test samples to evaluate')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose or args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Load test data
        loader = DataLoader()
        
        if args.test_data:
            import pandas as pd
            logger.info(f"Loading test data from {args.test_data}")
            test_df = pd.read_csv(args.test_data)
        elif args.country:
            logger.info(f"Loading test data for {args.country}")
            test_df = loader.load_country_data(args.country, 'test')
        else:
            parser.error("Must specify either --country or --test-data")
        
        # Limit samples if requested
        if args.max_samples:
            test_df = test_df.head(args.max_samples)
            logger.info(f"Limited to {len(test_df)} samples")
        
        logger.info(f"Test data: {len(test_df)} samples")
        
        # Load graph
        logger.info(f"Loading graph from {args.graph_path}")
        serializer = GraphSerializer()
        graph = serializer.load_graph(Path(args.graph_path).stem, format='pkl')
        
        # Initialize model and agent
        logger.info(f"Initializing model: {args.model}")
        model = get_model(args.model, args.temperature)
        agent = LangGraphAgent(graph, model)
        
        # Initialize evaluator
        evaluator = Evaluator(agent, sleep_time=args.sleep_time)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_agent(
            test_data=test_df,
            output_file=args.output,
            debug=args.debug
        )
        
        # Calculate comprehensive metrics if requested
        if args.metrics_report:
            logger.info("Calculating comprehensive metrics...")
            metrics = evaluator.calculate_comprehensive_metrics(
                df=results['dataframe'],
                test_df=test_df,
                output_file=args.metrics_report
            )
            logger.info(f"Metrics report saved to {args.metrics_report}")
        
        # Print summary
        rouge_scores = results['rouge_scores']
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Test samples: {len(test_df)}")
        print(f"Failed queries: {results['failed_queries']}")
        print(f"ROUGE-1 @rank1: {rouge_scores['rank1']:.4f}")
        print(f"ROUGE-1 @rank2: {rouge_scores['rank2']:.4f}")
        print(f"ROUGE-1 @rank3: {rouge_scores['rank3']:.4f}")
        print(f"ROUGE-1 best-of-3: {rouge_scores['best']:.4f}")
        
        if args.output:
            print(f"\nDetailed results saved to: {args.output}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()