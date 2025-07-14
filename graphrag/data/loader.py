"""
Data loading utilities for GraphRAG framework.
Handles loading and basic preprocessing of event data.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading of event data from various sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Optional path to data directory. Uses settings default if None.
        """
        self.data_dir = data_dir or settings.DATA_DIR
        
    def load_raw_data(self, years: List[int] = [2022, 2023, 2024]) -> pd.DataFrame:
        """
        Load raw NGEC data files for specified years.
        
        Args:
            years: List of years to load data for
            
        Returns:
            Combined DataFrame with all events
        """
        dfs = []
        
        for year in years:
            file_path = self.data_dir / "Raw_data" / f"ngecEvents.DV.{year}.txt"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep='\t', low_memory=False)
                    df['source_year'] = year
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} events from {year}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not dfs:
            raise FileNotFoundError(f"No data files found in {self.data_dir / 'Raw_data'}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} total events")
        
        return combined_df
    
    def load_country_data(self, country: str, split: str = "train") -> pd.DataFrame:
        """
        Load country-specific data split.
        
        Args:
            country: Country code (e.g., 'AFG', 'IND', 'RUS')
            split: Data split ('train' or 'test')
            
        Returns:
            DataFrame with country-specific events
        """
        file_path = settings.get_data_path(country, split)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Country data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} {split} events for {country}")
        
        return df
    
    def load_combined_data(self, country: str) -> pd.DataFrame:
        """
        Load combined train+test data for a country.
        
        Args:
            country: Country code
            
        Returns:
            Combined DataFrame
        """
        file_path = self.data_dir / "country_sets" / f"combined_data_{country}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Combined data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} combined events for {country}")
        
        return df
    
    def get_available_countries(self) -> List[str]:
        """
        Get list of available countries based on data files.
        
        Returns:
            List of country codes
        """
        countries = []
        splits_dir = self.data_dir / "final_splits"
        
        if splits_dir.exists():
            for file_path in splits_dir.glob("train_*.csv"):
                country = file_path.stem.replace("train_", "")
                countries.append(country)
        
        return sorted(countries)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "total_events": len(df),
            "unique_actors": df["Actor Name"].nunique() if "Actor Name" in df.columns else 0,
            "unique_recipients": df["Recipient Name"].nunique() if "Recipient Name" in df.columns else 0,
            "unique_event_types": df["Event Type"].nunique() if "Event Type" in df.columns else 0,
            "date_range": None,
            "missing_values": {},
            "issues": []
        }
        
        # Check date range
        if "Event Date" in df.columns:
            try:
                dates = pd.to_datetime(df["Event Date"], errors='coerce')
                stats["date_range"] = {
                    "start": dates.min(),
                    "end": dates.max(),
                    "invalid_dates": dates.isna().sum()
                }
            except Exception as e:
                stats["issues"].append(f"Date parsing error: {e}")
        
        # Check for missing values
        for col in ["Actor Name", "Recipient Name", "Event Type", "Event Date"]:
            if col in df.columns:
                missing = df[col].isna().sum()
                stats["missing_values"][col] = missing
                if missing > 0:
                    stats["issues"].append(f"{missing} missing values in {col}")
        
        # Check event type validity
        if "Event Type" in df.columns:
            invalid_types = set(df["Event Type"].unique()) - set(settings.EVENT_TYPES)
            if invalid_types:
                stats["issues"].append(f"Invalid event types: {invalid_types}")
        
        return stats