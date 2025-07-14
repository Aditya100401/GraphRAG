"""
Data splitting utilities for temporal train/test splits.
Ensures proper temporal separation to prevent data leakage.
"""

import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class DataSplitter:
    """Handles temporal data splitting for event prediction tasks."""
    
    def __init__(self, 
                 test_split_ratio: float = None, 
                 temporal_gap_days: int = None,
                 min_relation_frequency: int = None):
        """
        Initialize DataSplitter.
        
        Args:
            test_split_ratio: Fraction of data for testing
            temporal_gap_days: Minimum gap between train and test data
            min_relation_frequency: Minimum frequency for actor-recipient pairs
        """
        self.test_split_ratio = test_split_ratio or settings.TEST_SPLIT_RATIO
        self.temporal_gap_days = temporal_gap_days or settings.TEMPORAL_GAP_DAYS
        self.min_relation_frequency = min_relation_frequency or settings.MIN_RELATION_FREQUENCY
        
    def temporal_split(self, df: pd.DataFrame, 
                      by_relations: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/test split with proper separation.
        
        Args:
            df: DataFrame with events sorted by date
            by_relations: If True, split by actor-recipient relations
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if "Event Date" not in df.columns:
            raise ValueError("Event Date column is required for temporal splitting")
        
        # Ensure data is sorted by date
        df = df.sort_values("Event Date").reset_index(drop=True)
        
        if by_relations:
            return self._split_by_relations(df)
        else:
            return self._split_by_time(df)
    
    def _split_by_relations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by actor-recipient relations, using last events for testing.
        
        Args:
            df: Sorted DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Find frequent actor-recipient pairs
        relation_counts = df.groupby(["Actor Name", "Recipient Name"]).size()
        frequent_relations = relation_counts[relation_counts >= self.min_relation_frequency].index
        
        logger.info(f"Found {len(frequent_relations)} frequent relations (>= {self.min_relation_frequency} events)")
        
        train_data = []
        test_data = []
        
        for actor, recipient in frequent_relations:
            # Get all events for this relation
            relation_events = df[
                (df["Actor Name"] == actor) & 
                (df["Recipient Name"] == recipient)
            ].sort_values("Event Date")
            
            if len(relation_events) < self.min_relation_frequency:
                continue
            
            # Calculate split point
            n_events = len(relation_events)
            n_test = max(1, int(n_events * self.test_split_ratio))
            n_train = n_events - n_test
            
            # Ensure temporal gap
            train_events = relation_events.iloc[:n_train]
            test_events = relation_events.iloc[n_train:]
            
            if len(test_events) > 0 and len(train_events) > 0:
                # Check temporal gap
                last_train_date = train_events["Event Date"].max()
                first_test_date = test_events["Event Date"].min()
                gap = (first_test_date - last_train_date).days
                
                if gap >= self.temporal_gap_days:
                    train_data.append(train_events)
                    test_data.append(test_events)
                else:
                    logger.warning(f"Insufficient temporal gap for {actor}-{recipient}: {gap} days")
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        logger.info(f"Split into {len(train_df)} train and {len(test_df)} test events")
        logger.info(f"Train date range: {train_df['Event Date'].min()} to {train_df['Event Date'].max()}")
        logger.info(f"Test date range: {test_df['Event Date'].min()} to {test_df['Event Date'].max()}")
        
        return train_df, test_df
    
    def _split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by time cutoff with temporal gap.
        
        Args:
            df: Sorted DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Calculate split point
        total_events = len(df)
        split_idx = int(total_events * (1 - self.test_split_ratio))
        
        # Find dates at split point
        train_end_date = df.iloc[split_idx]["Event Date"]
        test_start_date = train_end_date + timedelta(days=self.temporal_gap_days)
        
        # Create splits with temporal gap
        train_df = df[df["Event Date"] <= train_end_date].copy()
        test_df = df[df["Event Date"] >= test_start_date].copy()
        
        logger.info(f"Time-based split: {len(train_df)} train, {len(test_df)} test events")
        logger.info(f"Temporal gap: {self.temporal_gap_days} days")
        
        return train_df, test_df
    
    def country_specific_split(self, df: pd.DataFrame, 
                              country: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create country-specific train/test split.
        
        Args:
            df: Full DataFrame
            country: Country code to filter for
            
        Returns:
            Tuple of (train_df, test_df) for the specified country
        """
        if "Country Code" not in df.columns:
            raise ValueError("Country Code column is required for country-specific splitting")
        
        # Filter to country
        country_df = df[df["Country Code"] == country].copy()
        logger.info(f"Filtering to {country}: {len(country_df)} events")
        
        if len(country_df) == 0:
            logger.warning(f"No events found for country {country}")
            return pd.DataFrame(), pd.DataFrame()
        
        return self.temporal_split(country_df)
    
    def multi_country_split(self, df: pd.DataFrame, 
                           countries: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create splits for multiple countries.
        
        Args:
            df: Full DataFrame
            countries: List of country codes
            
        Returns:
            Dictionary mapping country to (train_df, test_df) tuples
        """
        results = {}
        
        for country in countries:
            try:
                train_df, test_df = self.country_specific_split(df, country)
                results[country] = (train_df, test_df)
                logger.info(f"Split {country}: {len(train_df)} train, {len(test_df)} test")
            except Exception as e:
                logger.error(f"Error splitting {country}: {e}")
                results[country] = (pd.DataFrame(), pd.DataFrame())
        
        return results
    
    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   country: str, output_dir: Optional[str] = None):
        """
        Save train/test splits to files.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            country: Country code
            output_dir: Output directory (uses settings default if None)
        """
        if output_dir is None:
            output_dir = settings.DATA_DIR / "final_splits"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files
        train_path = output_dir / f"train_{country}.csv"
        test_path = output_dir / f"test_{country}.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved {country} splits:")
        logger.info(f"  Train: {train_path} ({len(train_df)} events)")
        logger.info(f"  Test: {test_path} ({len(test_df)} events)")
    
    def get_split_statistics(self, train_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics about the data split.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            
        Returns:
            Dictionary with split statistics
        """
        stats = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "total_size": len(train_df) + len(test_df),
            "test_ratio": len(test_df) / (len(train_df) + len(test_df)) if len(train_df) + len(test_df) > 0 else 0,
            "train_actors": train_df["Actor Name"].nunique() if len(train_df) > 0 else 0,
            "test_actors": test_df["Actor Name"].nunique() if len(test_df) > 0 else 0,
            "train_recipients": train_df["Recipient Name"].nunique() if len(train_df) > 0 else 0,
            "test_recipients": test_df["Recipient Name"].nunique() if len(test_df) > 0 else 0,
            "overlap_actors": len(set(train_df["Actor Name"]) & set(test_df["Actor Name"])) if len(train_df) > 0 and len(test_df) > 0 else 0,
            "overlap_recipients": len(set(train_df["Recipient Name"]) & set(test_df["Recipient Name"])) if len(train_df) > 0 and len(test_df) > 0 else 0,
        }
        
        # Date statistics
        if len(train_df) > 0 and "Event Date" in train_df.columns:
            stats["train_date_range"] = {
                "start": train_df["Event Date"].min(),
                "end": train_df["Event Date"].max()
            }
        
        if len(test_df) > 0 and "Event Date" in test_df.columns:
            stats["test_date_range"] = {
                "start": test_df["Event Date"].min(),
                "end": test_df["Event Date"].max()
            }
            
            # Calculate temporal gap
            if len(train_df) > 0:
                gap = (test_df["Event Date"].min() - train_df["Event Date"].max()).days
                stats["temporal_gap_days"] = gap
        
        return stats