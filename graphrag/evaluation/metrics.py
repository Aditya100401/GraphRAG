"""
Comprehensive evaluation metrics for event prediction tasks.
Based on the original calc_metrics.py implementation.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging
from config.settings import settings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EventPredictionMetrics:
    """
    Comprehensive metrics calculator for event prediction tasks.
    Based on temporal knowledge graph and sociopolitical event prediction literature.
    """
    
    def __init__(self, event_types: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            event_types: List of valid event types. Uses settings default if None.
        """
        self.event_types = set(event_types or settings.EVENT_TYPES)
    
    def calculate_ranking_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate ranking-based metrics (Hit@k, MRR).
        Standard metrics used in temporal knowledge graph literature.
        """
        hit_at_1 = hit_at_3 = hit_at_10 = 0
        mrr_scores = []
        reciprocal_ranks = []
        total_samples = len(df)

        for idx, row in df.iterrows():
            target = str(row["Event Type"]).upper()
            predictions = []

            # Collect predictions from all ranks
            for rank in [1, 2, 3]:
                pred_col = f"pred{rank}"
                if pred_col in df.columns:
                    pred = str(row[pred_col]).upper() if pd.notna(row[pred_col]) else ""
                    predictions.append(pred)

            # Calculate reciprocal rank
            rank_position = None
            for i, pred in enumerate(predictions):
                if pred == target:
                    rank_position = i + 1
                    break

            if rank_position is not None:
                rr = 1.0 / rank_position
                mrr_scores.append(rr)
                reciprocal_ranks.append(rank_position)

                # Update Hit@k metrics
                if rank_position <= 1:
                    hit_at_1 += 1
                if rank_position <= 3:
                    hit_at_3 += 1
                if rank_position <= 10:
                    hit_at_10 += 1
            else:
                mrr_scores.append(0.0)
                reciprocal_ranks.append(float('inf'))

        return {
            'hit_at_1': hit_at_1 / total_samples,
            'hit_at_3': hit_at_3 / total_samples,
            'hit_at_10': hit_at_10 / total_samples,
            'mrr': np.mean(mrr_scores),
            'total_samples': total_samples,
            'correct_at_rank_1': hit_at_1,
            'correct_at_rank_3': hit_at_3,
            'mean_rank': np.mean([r for r in reciprocal_ranks if r != float('inf')])
        }

    def calculate_position_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze where correct predictions appear (rank 1, 2, or 3).
        Important for understanding agent confidence calibration.
        """
        position_counts = {'rank_1': 0, 'rank_2': 0, 'rank_3': 0}
        total_correct = 0

        for idx, row in df.iterrows():
            target = str(row["Event Type"]).upper()

            for rank in [1, 2, 3]:
                pred_col = f"pred{rank}"
                if pred_col in df.columns:
                    pred = str(row[pred_col]).upper() if pd.notna(row[pred_col]) else ""
                    if pred == target:
                        position_counts[f'rank_{rank}'] += 1
                        total_correct += 1
                        break  # Only count first occurrence

        # Calculate percentages
        position_percentages = {}
        if total_correct > 0:
            for rank, count in position_counts.items():
                position_percentages[rank] = count / total_correct
        else:
            position_percentages = {'rank_1': 0, 'rank_2': 0, 'rank_3': 0}

        return {
            'position_counts': position_counts,
            'position_percentages': position_percentages,
            'total_correct_predictions': total_correct,
            'confidence_calibration': 'under_confident' if position_percentages['rank_2'] > position_percentages['rank_1'] else 'well_calibrated'
        }

    def calculate_comprehensive_classification_metrics(self, df: pd.DataFrame, rank: int = 1) -> Dict[str, Any]:
        """
        Calculate comprehensive precision, recall, F1 metrics for specified rank.
        Includes macro, micro, and weighted averages.
        """
        # Get predictions
        refs = df["Event Type"].astype(str).str.upper().tolist()
        preds = df[f"pred{rank}"].fillna("").astype(str).str.upper().tolist()
        
        # Get unique event types 
        all_event_types = sorted(set(refs + [p for p in preds if p and p != ""]))
        
        # Calculate per-class metrics using sklearn
        precision, recall, f1, support = precision_recall_fscore_support(
            refs, preds, labels=all_event_types, average=None, zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, event_type in enumerate(all_event_types):
            per_class_metrics[event_type] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # Aggregate metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages 
        total_support = sum(support)
        if total_support > 0:
            weighted_precision = np.average(precision, weights=support)
            weighted_recall = np.average(recall, weights=support)
            weighted_f1 = np.average(f1, weights=support)
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        # Micro averages 
        micro_precision = micro_recall = micro_f1 = sum(refs[i] == preds[i] for i in range(len(refs))) / len(refs)
        
        return {
            'per_class': per_class_metrics,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'accuracy': micro_f1
        }

    def calculate_best_of_k_metrics(self, df: pd.DataFrame, k: int = 3) -> Dict[str, Any]:
        """
        Calculate metrics when considering best prediction within top-k.
        """
        refs = df["Event Type"].astype(str).str.upper().tolist()
        
        # For each sample, check if ANY of the top-k predictions is correct
        best_predictions = []
        for idx, row in df.iterrows():
            target = refs[idx]
            found_correct = False
            
            for rank in range(1, k+1):
                pred_col = f"pred{rank}"
                if pred_col in df.columns:
                    pred = str(row[pred_col]).upper() if pd.notna(row[pred_col]) else ""
                    if pred == target:
                        best_predictions.append(target)
                        found_correct = True
                        break
            
            if not found_correct:
                # Use the first prediction if none are correct
                pred1 = str(row["pred1"]).upper() if pd.notna(row["pred1"]) else ""
                best_predictions.append(pred1)
        
        # Calculate comprehensive metrics for best-of-k
        all_event_types = sorted(set(refs + [p for p in best_predictions if p and p != ""]))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            refs, best_predictions, labels=all_event_types, average=None, zero_division=0
        )
        
        return {
            'macro_f1': np.mean(f1),
            'macro_precision': np.mean(precision),
            'macro_recall': np.mean(recall),
            'accuracy': sum(refs[i] == best_predictions[i] for i in range(len(refs))) / len(refs)
        }

    def calculate_baseline_metrics(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate baseline metrics for comparison.
        """
        if test_df is None or len(test_df) == 0:
            return {"error": "Test data not provided"}
        
        test_distribution = test_df['Event Type'].value_counts()
        
        # Most frequent class baseline
        most_frequent = test_distribution.index[0]
        majority_baseline = test_distribution.iloc[0] / len(test_df)
        
        # Random baseline 
        test_probs = (test_distribution / len(test_df)).values
        random_baseline = sum(p**2 for p in test_probs)
        
        return {
            'majority_class_accuracy': majority_baseline,
            'majority_class_name': most_frequent,
            'random_weighted_accuracy': random_baseline
        }

    def calculate_event_coverage_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate event coverage and diversity metrics.
        Important for understanding agent's breadth of predictions.
        """
        # Get all predictions across ranks
        all_preds = []
        refs = df["Event Type"].astype(str).str.upper().tolist()

        for rank in [1, 2, 3]:
            pred_col = f"pred{rank}"
            if pred_col in df.columns:
                preds = df[pred_col].fillna("").astype(str).str.upper().tolist()
                all_preds.extend([p for p in preds if p and p not in ["", "NAN"]])

        actual_events = set(refs)
        predicted_events = set(all_preds)

        # Calculate prediction distribution and entropy
        pred_counts = Counter(all_preds)
        total_preds = len(all_preds)

        if total_preds > 0:
            pred_probs = [count/total_preds for count in pred_counts.values()]
            entropy = -sum(p * np.log2(p) for p in pred_probs if p > 0)
        else:
            entropy = 0

        return {
            "event_type_coverage": len(predicted_events & actual_events) / len(actual_events) if actual_events else 0,
            "prediction_diversity": len(predicted_events),
            "total_actual_events": len(actual_events),
            "prediction_entropy": entropy,
            "never_predicted_events": list(actual_events - predicted_events),
            "hallucinated_events": list(predicted_events - actual_events),
            "prediction_distribution": dict(pred_counts)
        }

    def calculate_intensity_stratified_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance by event intensity.
        Critical for sociopolitical event prediction.
        """
        if "Event Intensity" not in df.columns:
            return {"error": "Event Intensity column not found"}

        # Define intensity ranges
        intensity_ranges = {
            "extremely_hostile": (-10, -7),
            "very_hostile": (-7, -5),
            "moderately_hostile": (-5, -2),
            "mildly_hostile": (-2, 0),
            "neutral_positive": (0, 10)
        }

        results = {}

        for range_name, (min_int, max_int) in intensity_ranges.items():
            mask = (df["Event Intensity"] >= min_int) & (df["Event Intensity"] < max_int)
            subset = df[mask]

            if len(subset) > 0:
                hit_1 = hit_3 = 0
                mrr_scores = []
                total = len(subset)

                for idx, row in subset.iterrows():
                    target = str(row["Event Type"]).upper()
                    predictions = [str(row[f"pred{i}"]).upper() if pd.notna(row[f"pred{i}"]) else "" for i in [1, 2, 3]]

                    # Calculate metrics for this subset
                    if predictions and predictions[0] == target:
                        hit_1 += 1
                    if target in predictions:
                        hit_3 += 1

                    # MRR for this sample
                    rank_position = None
                    for i, pred in enumerate(predictions):
                        if pred == target:
                            rank_position = i + 1
                            break

                    if rank_position is not None:
                        mrr_scores.append(1.0 / rank_position)
                    else:
                        mrr_scores.append(0.0)

                results[range_name] = {
                    "hit_at_1": hit_1 / total if total > 0 else 0,
                    "hit_at_3": hit_3 / total if total > 0 else 0,
                    "mrr": np.mean(mrr_scores),
                    "count": total,
                    "avg_intensity": subset["Event Intensity"].mean(),
                    "intensity_range": f"[{min_int}, {max_int})"
                }

        return results

    def generate_report(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None, 
                       test_df: Optional[pd.DataFrame] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive event prediction evaluation report.
        Enhanced with additional classification metrics and baselines.
        """
        # Collect all metrics
        ranking_metrics = self.calculate_ranking_metrics(df)
        position_analysis = self.calculate_position_analysis(df)
        rank1_classification = self.calculate_comprehensive_classification_metrics(df, rank=1)
        best_of_3_metrics = self.calculate_best_of_k_metrics(df, k=3)
        coverage_metrics = self.calculate_event_coverage_metrics(df)
        intensity_metrics = self.calculate_intensity_stratified_metrics(df)
        baseline_metrics = self.calculate_baseline_metrics(test_df or df)

        # Prepare the output lines in the desired format
        output_lines = []

        def add_section_title(title):
            output_lines.append("\n" + "=" * 80)
            output_lines.append(title.center(80))
            output_lines.append("=" * 80 + "\n")

        # Basic Statistics
        add_section_title("Basic Statistics")
        output_lines.append(f"{'Total samples:':<40} {ranking_metrics['total_samples']}")
        output_lines.append(f"{'Valid predictions (rank 1):':<40} {ranking_metrics['correct_at_rank_1']}")
        output_lines.append(f"{'Unique event types:':<40} {len(set(df['Event Type']))}")

        # Baseline Metrics
        if "error" not in baseline_metrics:
            add_section_title("Baseline Metrics")
            output_lines.append(f"{'Majority class baseline:':<40} {baseline_metrics['majority_class_accuracy']:.3f}")
            output_lines.append(f"{'Majority class name:':<40} {baseline_metrics['majority_class_name']}")
            output_lines.append(f"{'Random weighted baseline:':<40} {baseline_metrics['random_weighted_accuracy']:.3f}")

        # Ranking Quality Metrics
        add_section_title("Ranking Quality Metrics")
        output_lines.append(f"{'MRR (Mean Reciprocal Rank):':<40} {ranking_metrics['mrr']:.3f}")
        output_lines.append(f"{'Hit@1:':<40} {ranking_metrics['hit_at_1']:.3f}")
        output_lines.append(f"{'Hit@3:':<40} {ranking_metrics['hit_at_3']:.3f}")
        output_lines.append(f"{'Hit@10:':<40} {ranking_metrics['hit_at_10']:.3f}")

        # Accuracy Metrics
        add_section_title("Accuracy Metrics")
        for rank in [1, 2, 3]:
            count = position_analysis['position_counts'][f'rank_{rank}']
            accuracy = count / ranking_metrics['total_samples']
            output_lines.append(f"{f'Rank {rank} Accuracy:':<40} {accuracy:.3f} ({count}/{ranking_metrics['total_samples']})")

        total_correct = position_analysis['total_correct_predictions']
        best_of_3_accuracy = total_correct / ranking_metrics['total_samples']
        output_lines.append(f"{'Best Of 3 Accuracy:':<40} {best_of_3_accuracy:.3f} ({total_correct}/{ranking_metrics['total_samples']})")

        # Classification Metrics (Rank 1)
        add_section_title("Classification Metrics (Rank 1)")
        output_lines.append(f"{'Macro Precision:':<40} {rank1_classification['macro_precision']:.3f}")
        output_lines.append(f"{'Macro Recall:':<40} {rank1_classification['macro_recall']:.3f}")
        output_lines.append(f"{'Macro F1:':<40} {rank1_classification['macro_f1']:.3f}")
        output_lines.append(f"{'Weighted Precision:':<40} {rank1_classification['weighted_precision']:.3f}")
        output_lines.append(f"{'Weighted Recall:':<40} {rank1_classification['weighted_recall']:.3f}")
        output_lines.append(f"{'Weighted F1:':<40} {rank1_classification['weighted_f1']:.3f}")
        output_lines.append(f"{'Micro Precision (Accuracy):':<40} {rank1_classification['micro_precision']:.3f}")

        # Best-of-3 Classification Metrics
        add_section_title("Best-of-3 Classification Metrics")
        output_lines.append(f"{'Best-of-3 Macro Precision:':<40} {best_of_3_metrics['macro_precision']:.3f}")
        output_lines.append(f"{'Best-of-3 Macro Recall:':<40} {best_of_3_metrics['macro_recall']:.3f}")
        output_lines.append(f"{'Best-of-3 Macro F1:':<40} {best_of_3_metrics['macro_f1']:.3f}")
        output_lines.append(f"{'Best-of-3 Accuracy:':<40} {best_of_3_metrics['accuracy']:.3f}")

        # Prediction Diversity Metrics
        add_section_title("Prediction Diversity Metrics")
        output_lines.append(f"{'Prediction Entropy:':<40} {coverage_metrics['prediction_entropy']:.3f}")
        output_lines.append(f"{'Event Type Coverage:':<40} {coverage_metrics['event_type_coverage']:.3f}")
        output_lines.append(f"{'Unique Predictions:':<40} {coverage_metrics['prediction_diversity']}")
        output_lines.append(f"{'Total Predictions:':<40} {sum(coverage_metrics['prediction_distribution'].values())}")

        if coverage_metrics['never_predicted_events']:
            output_lines.append(f"\nNever predicted events: {', '.join(coverage_metrics['never_predicted_events'])}")
        
        if coverage_metrics['hallucinated_events']:
            output_lines.append(f"Hallucinated events: {', '.join(coverage_metrics['hallucinated_events'])}")

        output_lines.append("\nPrediction Distribution:")
        sorted_preds = sorted(coverage_metrics['prediction_distribution'].items(), key=lambda x: x[1], reverse=True)
        for event, count in sorted_preds:
            output_lines.append(f"{event:<20} {count}")

        # Per-Event Type Metrics (Rank 1)
        add_section_title("Per-Event Type Metrics (Rank 1)")
        output_lines.append(f"{'Event Type':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        output_lines.append("-" * 65)
        
        for event_type, metrics in sorted(rank1_classification['per_class'].items()):
            if metrics['support'] > 0:  # Only show classes that appear in test
                output_lines.append(f"{event_type:<15} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
                                f"{metrics['f1']:<12.3f} {metrics['support']:<12}")

        # Temporal/Intensity Consistency Metrics
        add_section_title("Temporal/Intensity Consistency Metrics")
        if "error" not in intensity_metrics:
            for level, metrics in intensity_metrics.items():
                level_name = level.replace('_', ' ').title()
                output_lines.append(f"{level_name:<25} {'Hit@3:':<15} {metrics['hit_at_3']:.3f} {'Count:':<10} {metrics['count']}")
        else:
            output_lines.append("Temporal metrics unavailable: missing Event Intensity column")

        # Print to console
        for line in output_lines:
            print(line)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                for line in output_lines:
                    f.write(line + '\n')

        # Return the raw metrics dictionary
        return {
            "ranking": ranking_metrics,
            "position_analysis": position_analysis,
            "rank1_classification": rank1_classification,
            "best_of_3_classification": best_of_3_metrics,
            "coverage": coverage_metrics,
            "intensity": intensity_metrics,
            "baselines": baseline_metrics
        }