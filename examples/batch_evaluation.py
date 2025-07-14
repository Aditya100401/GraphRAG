"""
Batch Evaluation Example: Large-scale evaluation across multiple countries and models.

This example demonstrates how to:
1. Run evaluations across multiple countries
2. Compare different models and configurations
3. Generate comprehensive reports
4. Analyze results across different settings
"""

import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.data import DataLoader
from graphrag.graph import GraphSerializer
from graphrag.agents import LangGraphAgent
from graphrag.evaluation import Evaluator, EventPredictionMetrics
from langchain_openai import ChatOpenAI
import pandas as pd
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEvaluator:
    """
    Handles large-scale batch evaluation across multiple settings.
    """
    
    def __init__(self, output_dir: str = "batch_results"):
        """Initialize batch evaluator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.loader = DataLoader()
        self.serializer = GraphSerializer()
        self.results = []
        
    def run_single_evaluation(self, country: str, model_name: str, temperature: float = 0.0, 
                            max_samples: int = None) -> dict:
        """
        Run evaluation for a single country-model combination.
        
        Args:
            country: Country code (e.g., 'IND', 'AFG', 'RUS')
            model_name: Model name (e.g., 'gpt-4o-mini', 'gpt-4')
            temperature: Model temperature
            max_samples: Maximum test samples to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation: {country} + {model_name} (temp={temperature})")
        
        try:
            # Load test data
            test_data = self.loader.load_country_data(country, 'test')
            if max_samples:
                test_data = test_data.head(max_samples)
            
            # Load graph
            graph = self.serializer.load_country_graph(country, 'train')
            
            # Initialize model and agent
            model = ChatOpenAI(model=model_name, temperature=temperature)
            agent = LangGraphAgent(graph, model, max_iterations=5)
            
            # Run evaluation
            evaluator = Evaluator(agent, sleep_time=0.5)  # Reasonable sleep time
            
            eval_results = evaluator.evaluate_agent(
                test_data=test_data,
                output_file=str(self.output_dir / f"predictions_{country}_{model_name.replace('-', '_')}_temp{temperature}.csv"),
                debug=False
            )
            
            # Calculate comprehensive metrics
            metrics = evaluator.calculate_comprehensive_metrics(
                df=eval_results['dataframe'],
                test_df=test_data,
                output_file=str(self.output_dir / f"metrics_{country}_{model_name.replace('-', '_')}_temp{temperature}.txt")
            )
            
            # Compile results
            result = {
                'country': country,
                'model': model_name,
                'temperature': temperature,
                'test_samples': len(test_data),
                'failed_queries': eval_results['failed_queries'],
                'rouge_scores': eval_results['rouge_scores'],
                'ranking_metrics': metrics.get('ranking', {}),
                'classification_metrics': metrics.get('rank1_classification', {}),
                'coverage_metrics': metrics.get('coverage', {}),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"✓ Completed: {country} + {model_name} - ROUGE-1 Best: {eval_results['rouge_scores']['best']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed: {country} + {model_name} - {e}")
            return {
                'country': country,
                'model': model_name,
                'temperature': temperature,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_evaluation(self, countries: list, models: list, temperatures: list = [0.0],
                           max_samples: int = None) -> list:
        """
        Run batch evaluation across multiple countries and models.
        
        Args:
            countries: List of country codes
            models: List of model names
            temperatures: List of temperatures to test
            max_samples: Maximum samples per evaluation
            
        Returns:
            List of evaluation results
        """
        total_evaluations = len(countries) * len(models) * len(temperatures)
        logger.info(f"Starting batch evaluation: {total_evaluations} total evaluations")
        
        results = []
        
        for country in countries:
            for model in models:
                for temp in temperatures:
                    try:
                        result = self.run_single_evaluation(country, model, temp, max_samples)
                        results.append(result)
                        self.results.append(result)
                        
                    except KeyboardInterrupt:
                        logger.info("Evaluation interrupted by user")
                        break
                    except Exception as e:
                        logger.error(f"Unexpected error in batch evaluation: {e}")
                        
        return results
    
    def generate_comparison_report(self, results: list) -> dict:
        """Generate comparison report across all evaluations."""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful evaluations to compare"}
        
        # Create comparison DataFrame
        comparison_data = []
        for result in successful_results:
            row = {
                'Country': result['country'],
                'Model': result['model'],
                'Temperature': result['temperature'],
                'Test_Samples': result['test_samples'],
                'Failed_Queries': result['failed_queries'],
                'ROUGE_Rank1': result['rouge_scores']['rank1'],
                'ROUGE_Rank2': result['rouge_scores']['rank2'],
                'ROUGE_Rank3': result['rouge_scores']['rank3'],
                'ROUGE_Best': result['rouge_scores']['best'],
                'Hit_at_1': result['ranking_metrics'].get('hit_at_1', 0),
                'Hit_at_3': result['ranking_metrics'].get('hit_at_3', 0),
                'MRR': result['ranking_metrics'].get('mrr', 0),
                'Macro_F1': result['classification_metrics'].get('macro_f1', 0),
                'Accuracy': result['classification_metrics'].get('accuracy', 0),
                'Event_Coverage': result['coverage_metrics'].get('event_type_coverage', 0)
            }
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = self.output_dir / "batch_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)
        
        # Generate summary statistics
        summary = {
            'total_evaluations': len(successful_results),
            'countries_tested': df_comparison['Country'].nunique(),
            'models_tested': df_comparison['Model'].nunique(),
            'best_performing': {
                'rouge_best': df_comparison.loc[df_comparison['ROUGE_Best'].idxmax()].to_dict(),
                'hit_at_1': df_comparison.loc[df_comparison['Hit_at_1'].idxmax()].to_dict(),
                'macro_f1': df_comparison.loc[df_comparison['Macro_F1'].idxmax()].to_dict()
            },
            'average_performance': {
                'rouge_best': df_comparison['ROUGE_Best'].mean(),
                'hit_at_1': df_comparison['Hit_at_1'].mean(),
                'hit_at_3': df_comparison['Hit_at_3'].mean(),
                'mrr': df_comparison['MRR'].mean(),
                'macro_f1': df_comparison['Macro_F1'].mean()
            },
            'by_country': df_comparison.groupby('Country')[['ROUGE_Best', 'Hit_at_1', 'Macro_F1']].mean().to_dict(),
            'by_model': df_comparison.groupby('Model')[['ROUGE_Best', 'Hit_at_1', 'Macro_F1']].mean().to_dict()
        }
        
        # Save summary
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Run the batch evaluation example."""
    
    print("Batch Evaluation Example")
    print("=" * 40)
    
    # Configuration
    COUNTRIES = ['IND', 'AFG', 'RUS']  # Countries to evaluate
    MODELS = ['gpt-4o-mini']  # Models to test
    TEMPERATURES = [0.0, 0.3]  # Temperature settings
    MAX_SAMPLES = 5  # Limit samples for demo (set to None for full evaluation)
    
    print(f"Configuration:")
    print(f"  Countries: {COUNTRIES}")
    print(f"  Models: {MODELS}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  Max samples per evaluation: {MAX_SAMPLES}")
    
    try:
        # Initialize batch evaluator
        batch_evaluator = BatchEvaluator("batch_results_demo")
        
        # Check for available data
        loader = DataLoader()
        available_countries = loader.get_available_countries()
        countries_to_test = [c for c in COUNTRIES if c in available_countries]
        
        if not countries_to_test:
            print("⚠ No country data available. Please run data processing first.")
            print("Available countries:", available_countries)
            return
        
        print(f"\nTesting available countries: {countries_to_test}")
        
        # Run batch evaluation
        results = batch_evaluator.run_batch_evaluation(
            countries=countries_to_test,
            models=MODELS,
            temperatures=TEMPERATURES,
            max_samples=MAX_SAMPLES
        )
        
        # Generate comparison report
        print(f"\nGenerating comparison report...")
        summary = batch_evaluator.generate_comparison_report(results)
        
        if 'error' not in summary:
            print(f"✓ Batch evaluation completed!")
            print(f"  Total evaluations: {summary['total_evaluations']}")
            print(f"  Average ROUGE-1 Best: {summary['average_performance']['rouge_best']:.3f}")
            print(f"  Average Hit@1: {summary['average_performance']['hit_at_1']:.3f}")
            print(f"  Average Macro F1: {summary['average_performance']['macro_f1']:.3f}")
            
            print(f"\nBest performing configurations:")
            best_rouge = summary['best_performing']['rouge_best']
            print(f"  ROUGE-1 Best: {best_rouge['Country']} + {best_rouge['Model']} (temp={best_rouge['Temperature']}) = {best_rouge['ROUGE_Best']:.3f}")
            
            print(f"\nResults saved to: {batch_evaluator.output_dir}")
            print(f"  - batch_comparison.csv: Detailed comparison")
            print(f"  - batch_summary.json: Summary statistics")
            print(f"  - Individual prediction and metric files")
        else:
            print(f"⚠ Comparison failed: {summary['error']}")
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 40)
    print("Batch Evaluation Example completed!")
    print("\nThis example shows how to:")
    print("- Run systematic evaluations across multiple settings")
    print("- Compare model performance across countries")
    print("- Generate comprehensive comparison reports")
    print("- Scale evaluation to production environments")

if __name__ == "__main__":
    main()