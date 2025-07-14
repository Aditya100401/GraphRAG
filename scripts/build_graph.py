#!/usr/bin/env python3
"""
CLI script for building temporal knowledge graphs from event data.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import graphrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.data import DataLoader, DataCleaner, DataSplitter
from graphrag.graph import GraphBuilder, GraphSerializer
from config.settings import settings

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOGS_DIR / 'build_graph.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description='Build temporal knowledge graphs from event data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build graph for a specific country
  python scripts/build_graph.py --country IND --split train
  
  # Build graphs for all countries
  python scripts/build_graph.py --all-countries
  
  # Clean raw data and build graphs
  python scripts/build_graph.py --clean-data --country AFG --split train
        """
    )
    
    parser.add_argument('--country', '-c', type=str, 
                        help='Country code (e.g., AFG, IND, RUS)')
    parser.add_argument('--split', '-s', type=str, choices=['train', 'test'], 
                        default='train', help='Data split to use')
    parser.add_argument('--all-countries', action='store_true',
                        help='Build graphs for all available countries')
    parser.add_argument('--clean-data', action='store_true',
                        help='Clean raw data before building graph')
    parser.add_argument('--input-file', '-i', type=str,
                        help='Input CSV file path (overrides country/split)')
    parser.add_argument('--output-name', '-o', type=str,
                        help='Output graph name (default: auto-generated)')
    parser.add_argument('--format', '-f', type=str, choices=['pkl', 'graphml', 'both'],
                        default='pkl', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.all_countries and not args.country and not args.input_file:
        parser.error("Must specify --country, --all-countries, or --input-file")
    
    try:
        loader = DataLoader()
        builder = GraphBuilder()
        serializer = GraphSerializer()
        
        if args.input_file:
            # Build graph from specific file
            import pandas as pd
            logger.info(f"Loading data from {args.input_file}")
            df = pd.read_csv(args.input_file)
            
            if args.clean_data:
                logger.info("Cleaning data...")
                cleaner = DataCleaner()
                df = cleaner.clean_dataset(df)
            
            logger.info("Building graph...")
            graph = builder.build_temporal_kg(df)
            
            output_name = args.output_name or Path(args.input_file).stem
            saved_files = serializer.save_graph(graph, output_name, [args.format])
            
            logger.info(f"Graph saved: {saved_files}")
            
        elif args.all_countries:
            # Build graphs for all countries
            countries = loader.get_available_countries()
            logger.info(f"Building graphs for countries: {countries}")
            
            for country in countries:
                try:
                    logger.info(f"Processing {country}...")
                    graph = builder.build_country_graph(country, args.split)
                    saved_files = serializer.save_country_graph(graph, country, args.split)
                    
                    stats = builder.get_graph_statistics(graph)
                    logger.info(f"{country} graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
                    
                except Exception as e:
                    logger.error(f"Failed to build graph for {country}: {e}")
                    
        else:
            # Build graph for specific country
            logger.info(f"Building graph for {args.country} ({args.split})")
            graph = builder.build_country_graph(args.country, args.split)
            saved_files = serializer.save_country_graph(graph, args.country, args.split)
            
            stats = builder.get_graph_statistics(graph)
            logger.info(f"Graph statistics: {stats}")
            logger.info(f"Graph saved: {saved_files}")
        
        logger.info("Graph building completed successfully!")
        
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()