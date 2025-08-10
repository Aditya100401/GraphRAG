"""
Evaluation framework for GraphRAG agents.
Based on the original evals.py implementation.
"""

import pandas as pd
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm
import evaluate

from .metrics import EventPredictionMetrics
from ..agents.base import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)

class Evaluator:
    """Comprehensive evaluator for event prediction agents."""
    
    def __init__(self, agent: BaseAgent, sleep_time: float = None): #type: ignore
        """
        Initialize evaluator.
        
        Args:
            agent: Agent instance to evaluate
            sleep_time: Sleep time between predictions (seconds)
        """
        self.agent = agent
        self.sleep_time = sleep_time or settings.EVALUATION_SLEEP_TIME
        self.metrics_calculator = EventPredictionMetrics()
        
    def build_query(self, row: pd.Series) -> Tuple[str, str, str, str, float, str]:
        """
        Build a context-aware query that includes current event information.
        
        Args:
            row: DataFrame row with event data
            
        Returns:
            Tuple of (query, actor, recipient, date, event_intensity, event_type)
        """
        actor = str(row.get("Actor Name", "")) if pd.notna(row.get("Actor Name")) else ""
        recipient = str(row.get("Recipient Name", "")) if pd.notna(row.get("Recipient Name")) else ""
        date = str(row.get("Event Date", "")) if pd.notna(row.get("Event Date")) else ""
        event_type = str(row.get("Event Type", "")) if pd.notna(row.get("Event Type")) else ""
        event_intensity = row.get("Event Intensity", 0) if pd.notna(row.get("Event Intensity")) else 0
        contexts = str(row.get("Contexts", "")) if pd.notna(row.get("Contexts")) else ""
        
        # Get intensity description
        if event_intensity < -7:
            intensity_desc = "extremely negative (severe hostility/violence)"
        elif event_intensity < -5:
            intensity_desc = "highly negative (significant hostility)"
        elif event_intensity < -2:
            intensity_desc = "moderately negative (notable hostility)"
        elif event_intensity < 0:
            intensity_desc = "mildly negative (some tension)"
        elif event_intensity == 0:
            intensity_desc = "neutral"
        else:
            intensity_desc = "positive (cooperative/supportive)"
        
        # Build comprehensive query that matches system prompt expectations
        query = f"""Given this context:
- Current event intensity: {event_intensity} - {intensity_desc}
- Event context: {contexts}
- Actors involved: {actor} and {recipient}  
- Date: {date}

Based on this current event and the historical relationship patterns between these actors, what are the 3 most likely follow-up event types that could occur between {actor} and {recipient}?

Consider the current event intensity ({event_intensity}) and apply the intensity-based prediction rules from your guidelines.

Choose your top 3 predictions from: {', '.join(settings.EVENT_TYPES)}"""
        
        return query, actor, recipient, date, event_intensity, event_type
    
    def get_agent_predictions(self, query: str, actor: str, recipient: str, date: str, 
                            debug: bool = False) -> Tuple[List[str], List, str]:
        """
        Get agent predictions without any validation or correction.
        
        Args:
            query: Query string
            actor: Actor name
            recipient: Recipient name
            date: Date string
            debug: Whether to log debug information
            
        Returns:
            Tuple of (predictions, messages, all_messages_str)
        """
        if debug:
            logger.info(f"Query: {query}")
        
        try:
            result = self.agent.run(
                query=query,
                actor=actor,
                recipient=recipient,
                date=date
            )
        except Exception as e:
            logger.error(f"Error running agent: {e}", exc_info=True)
            return ["", "", ""], [], str(e)

        if not result.get("messages"):
            return ["", "", ""], [], "No messages returned"

        messages = result["messages"]
        
        all_messages_str = "\n\n".join(
            f"{getattr(m, 'role', type(m).__name__)}: {getattr(m, 'content', str(m))}"
            for m in messages
        )

        # Extract predictions from answer - NO VALIDATION
        predictions = self.agent.extract_predictions(result) #type: ignore
        
        if debug:
            logger.info(f"Agent predictions: {predictions}")
        
        return predictions, messages, all_messages_str
    
    def analyze_agent_reasoning(self, df_results: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Analyze agent's logical reasoning patterns without correcting them.
        
        Args:
            df_results: DataFrame with evaluation results
            
        Returns:
            Tuple of (analysis_dict, violation_examples)
        """
        analysis = {
            'total_predictions': 0,
            'logical_violations': 0,
            'intensity_violations': 0,
            'empty_predictions': 0,
            'valid_format': 0
        }
        
        violation_examples = []
        
        for idx, row in df_results.iterrows():
            event_type = row.get('Event Type', '')
            event_intensity = row.get('Event Intensity', 0)
            pred1 = row.get('pred1', '')
            
            analysis['total_predictions'] += 1
            
            if not pred1:
                analysis['empty_predictions'] += 1
                continue
                
            if pred1 in settings.EVENT_TYPES:
                analysis['valid_format'] += 1
            
            # Check for logical violations (for analysis only, NOT correction)
            violation = None
            if event_intensity < -2 and pred1 == "COOPERATE":
                violation = "cooperative_after_hostility"
                analysis['intensity_violations'] += 1
            elif event_intensity > 2 and pred1 in ["ASSAULT", "THREATEN"]:
                violation = "hostility_after_cooperation"
                analysis['intensity_violations'] += 1
            elif event_type == "Assault" and pred1 == "COOPERATE":
                violation = "cooperative_after_assault"
                analysis['logical_violations'] += 1
            
            if violation:
                violation_examples.append({
                    'row': idx,
                    'event_type': event_type,
                    'intensity': event_intensity,
                    'prediction': pred1,
                    'violation_type': violation
                })
        
        return analysis, violation_examples
    
    def evaluate_agent(self, test_data: pd.DataFrame, output_file: Optional[str] = None, 
                      debug: bool = False) -> Dict[str, Any]:
        """
        Clean evaluation of agent performance without any prediction validation.
        
        Args:
            test_data: Test DataFrame
            output_file: Optional output file path
            debug: Whether to enable debug logging
            
        Returns:
            Dictionary with evaluation results
        """
        df = test_data.copy()
        refs = df["Event Type"].astype(str).str.upper().tolist()

        # Initialize result containers
        preds1, preds2, preds3 = [], [], []
        raw_outputs = []
        all_messages_list = []
        failed_queries = 0

        logger.info(f"Starting clean evaluation on {len(df)} test cases...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Build context-aware query
            query_str, actor, recipient, date, event_intensity, event_type = self.build_query(row)
            
            if debug:
                logger.info(f"\nRow {idx}: {event_type} (intensity: {event_intensity})")
                logger.info(f"Actor: {actor}, Recipient: {recipient}, Date: {date}")

            try:
                # Get agent predictions - NO VALIDATION OR CORRECTION
                top3, messages, all_messages_str = self.get_agent_predictions(
                    query_str, actor, recipient, date, debug=debug
                )
                
                if not top3[0]:  # Empty prediction
                    failed_queries += 1
                                
            except Exception as e:
                logger.error(f"Exception at row {idx}: {e}", exc_info=True)
                top3 = ["", "", ""]
                messages, all_messages_str = [], ""
                failed_queries += 1

            # Store RAW predictions
            preds1.append(top3[0] if len(top3) > 0 and top3[0] else "")
            preds2.append(top3[1] if len(top3) > 1 and top3[1] else "")
            preds3.append(top3[2] if len(top3) > 2 and top3[2] else "")
            raw_outputs.append(messages[-1].content if messages else "")
            all_messages_list.append(all_messages_str)
            
            time.sleep(self.sleep_time)

        # Calculate ranking metrics (Hit@1, Hit@3, MRR) instead of ROUGE
        logger.info("Calculating ranking metrics (Hit@1, Hit@3, MRR)...")
        
        hit_at_1 = []
        hit_at_3 = []
        mrr_scores = []
        
        for i, true_event in enumerate(refs):
            pred_list = [preds1[i], preds2[i], preds3[i]]
            
            # Hit@1: True if first prediction matches
            hit_1 = 1 if pred_list[0] == true_event else 0
            hit_at_1.append(hit_1)
            
            # Hit@3: True if any of top 3 predictions match
            hit_3 = 1 if true_event in pred_list else 0
            hit_at_3.append(hit_3)
            
            # MRR: Mean Reciprocal Rank
            mrr = 0
            for rank, pred in enumerate(pred_list, 1):
                if pred == true_event:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)
        
        # Calculate exact match scores for compatibility
        exact_match_1 = [1 if p == r else 0 for p, r in zip(preds1, refs)]
        exact_match_2 = [1 if p == r else 0 for p, r in zip(preds2, refs)]
        exact_match_3 = [1 if p == r else 0 for p, r in zip(preds3, refs)]

        # Add results to dataframe
        df["pred1"], df["pred2"], df["pred3"] = preds1, preds2, preds3
        df["exact_match_1"], df["exact_match_2"], df["exact_match_3"] = exact_match_1, exact_match_2, exact_match_3
        df["hit_at_1"], df["hit_at_3"], df["mrr"] = hit_at_1, hit_at_3, mrr_scores
        df["raw_output"] = raw_outputs
        df["all_agent_messages"] = all_messages_list

        # Analyze agent reasoning patterns
        analysis, violation_examples = self.analyze_agent_reasoning(df)

        # Print results
        print("\n" + "="*60)
        print("AGENT EVALUATION RESULTS")
        print("="*60)
        print(f"Hit@1 (Rank 1 Accuracy): {sum(hit_at_1)/len(hit_at_1):.4f}")
        print(f"Hit@3 (Top 3 Accuracy):  {sum(hit_at_3)/len(hit_at_3):.4f}")
        print(f"Mean Reciprocal Rank:    {sum(mrr_scores)/len(mrr_scores):.4f}")
        print(f"Exact Match @rank1:      {sum(exact_match_1)/len(exact_match_1):.4f}")
        print(f"Exact Match @rank2:      {sum(exact_match_2)/len(exact_match_2):.4f}")
        print(f"Exact Match @rank3:      {sum(exact_match_3)/len(exact_match_3):.4f}")
        
        # Prediction distribution analysis
        print("\nPREDICTION DISTRIBUTION:")
        pred1_counts = Counter([p for p in preds1 if p])
        for pred, count in pred1_counts.most_common():
            percentage = (count / len(preds1)) * 100
            print(f"  {pred}: {count}/{len(preds1)} ({percentage:.1f}%)")
        
        # Agent reasoning analysis
        print("\nAGENT REASONING ANALYSIS:")
        print(f"  Total predictions: {analysis['total_predictions']}")
        print(f"  Failed queries: {failed_queries} ({failed_queries/len(df)*100:.1f}%)")
        print(f"  Valid format: {analysis['valid_format']}/{analysis['total_predictions']} ({analysis['valid_format']/analysis['total_predictions']*100:.1f}%)")
        print(f"  Intensity violations: {analysis['intensity_violations']} ({analysis['intensity_violations']/analysis['total_predictions']*100:.1f}%)")
        print(f"  Logic violations: {analysis['logical_violations']} ({analysis['logical_violations']/analysis['total_predictions']*100:.1f}%)")
        
        if violation_examples[:3]:  # Show first 3 examples
            print("\nExample violations:")
            for ex in violation_examples[:3]:
                print(f"  Row {ex['row']}: {ex['event_type']} (intensity {ex['intensity']}) → {ex['prediction']} [{ex['violation_type']}]")

        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nSaved results to {output_file}")

        return {
            'dataframe': df,
            'ranking_metrics': {
                'hit_at_1': sum(hit_at_1)/len(hit_at_1),
                'hit_at_3': sum(hit_at_3)/len(hit_at_3),
                'mrr': sum(mrr_scores)/len(mrr_scores)
            },
            'exact_match_scores': {
                'rank1': sum(exact_match_1)/len(exact_match_1),
                'rank2': sum(exact_match_2)/len(exact_match_2), 
                'rank3': sum(exact_match_3)/len(exact_match_3)
            },
            'analysis': analysis,
            'violations': violation_examples,
            'failed_queries': failed_queries
        }
    
    def calculate_comprehensive_metrics(self, df: pd.DataFrame, 
                                        train_df: Optional[pd.DataFrame] = None,
                                        test_df: Optional[pd.DataFrame] = None,
                                        output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics using the metrics calculator.
        
        Args:
            df: DataFrame with predictions
            train_df: Training data for baseline calculations
            test_df: Test data for baseline calculations  
            output_file: Optional output file for report
            
        Returns:
            Dictionary with all metrics
        """
        return self.metrics_calculator.generate_report(
            df=df, 
            train_df=train_df, 
            test_df=test_df, 
            output_file=output_file
        )