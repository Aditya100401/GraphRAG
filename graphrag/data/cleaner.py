"""
Data cleaning and preprocessing utilities for GraphRAG framework.
Implements the cleaning pipeline from the original notebooks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import Counter
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from config.settings import settings

logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles data cleaning and preprocessing operations."""
    
    def __init__(self):
        """Initialize DataCleaner with default settings."""
        self.geocoder = Nominatim(user_agent="graphrag_geocoder")
        self.min_entity_frequency = settings.MIN_ENTITY_FREQUENCY
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete cleaning pipeline to raw data.
        
        Args:
            df: Raw event DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {len(df)} events")
        
        # Basic cleaning
        df = self._basic_cleaning(df)
        logger.info(f"After basic cleaning: {len(df)} events")
        
        # Remove events with missing actor/recipient
        df = self._remove_missing_entities(df)
        logger.info(f"After removing missing entities: {len(df)} events")
        
        # Geocoding for missing country codes
        df = self._geocode_missing_countries(df)
        logger.info(f"After geocoding: {len(df)} events")
        
        # Filter by entity frequency
        df = self._filter_by_frequency(df)
        logger.info(f"After frequency filtering: {len(df)} events")
        
        # Final validation
        df = self._final_validation(df)
        logger.info(f"Final cleaned dataset: {len(df)} events")
        
        return df
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning operations."""
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_size - len(df)} duplicate events")
        
        # Clean string columns
        string_columns = ["Actor Name", "Recipient Name", "Event Type", "Place"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        
        # Convert date column
        if "Event Date" in df.columns:
            df["Event Date"] = pd.to_datetime(df["Event Date"], errors='coerce')
        
        # Convert intensity to numeric
        if "Event Intensity" in df.columns:
            df["Event Intensity"] = pd.to_numeric(df["Event Intensity"], errors='coerce')
        
        return df
    
    def _remove_missing_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove events with missing actor or recipient names."""
        initial_size = len(df)
        
        # Remove rows where Actor Name or Recipient Name is missing
        df = df.dropna(subset=["Actor Name", "Recipient Name"])
        
        # Remove rows where actor and recipient are the same
        df = df[df["Actor Name"] != df["Recipient Name"]]
        
        logger.info(f"Removed {initial_size - len(df)} events with missing/invalid entities")
        return df
    
    def _geocode_missing_countries(self, df: pd.DataFrame, max_geocode: int = 5000) -> pd.DataFrame:
        """
        Fill missing country codes using geocoding.
        
        Args:
            df: DataFrame with potential missing country codes
            max_geocode: Maximum number of places to geocode
            
        Returns:
            DataFrame with filled country codes
        """
        if "Country Code" not in df.columns or "Place" not in df.columns:
            logger.warning("Country Code or Place column not found, skipping geocoding")
            return df
        
        # Find missing country codes
        missing_mask = df["Country Code"].isna() & df["Place"].notna()
        missing_places = df.loc[missing_mask, "Place"].value_counts()
        
        if len(missing_places) == 0:
            logger.info("No missing country codes to geocode")
            return df
        
        logger.info(f"Found {len(missing_places)} unique places with missing country codes")
        
        # Geocode most frequent places first
        geocoded_countries = {}
        geocoded_count = 0
        
        for place, count in missing_places.head(max_geocode).items():
            if geocoded_count >= max_geocode:
                break
                
            try:
                location = self.geocoder.geocode(place, timeout=10)
                if location and hasattr(location, 'raw') and 'display_name' in location.raw:
                    # Extract country from display_name (usually the last part)
                    country = location.raw['display_name'].split(', ')[-1]
                    geocoded_countries[place] = country
                    geocoded_count += 1
                    
                    if geocoded_count % 100 == 0:
                        logger.info(f"Geocoded {geocoded_count} places...")
                        
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                logger.warning(f"Geocoding failed for {place}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected geocoding error for {place}: {e}")
                continue
        
        # Fill missing country codes
        for place, country in geocoded_countries.items():
            mask = (df["Place"] == place) & df["Country Code"].isna()
            df.loc[mask, "Country Code"] = country
        
        logger.info(f"Filled {len(geocoded_countries)} country codes through geocoding")
        return df
    
    def _filter_by_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter entities by frequency to improve data quality."""
        initial_size = len(df)
        
        # Count entity frequencies
        actor_counts = df["Actor Name"].value_counts()
        recipient_counts = df["Recipient Name"].value_counts()
        
        # Keep entities that appear at least min_frequency times
        frequent_actors = set(actor_counts[actor_counts >= self.min_entity_frequency].index)
        frequent_recipients = set(recipient_counts[recipient_counts >= self.min_entity_frequency].index)
        
        # Filter dataframe
        df = df[
            df["Actor Name"].isin(frequent_actors) & 
            df["Recipient Name"].isin(frequent_recipients)
        ]
        
        logger.info(f"Filtered to entities with frequency >= {self.min_entity_frequency}")
        logger.info(f"Kept {len(frequent_actors)} actors and {len(frequent_recipients)} recipients")
        logger.info(f"Removed {initial_size - len(df)} events due to low frequency entities")
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final validation and cleanup."""
        initial_size = len(df)
        
        # Remove events with invalid event types
        if "Event Type" in df.columns:
            valid_types = set(settings.EVENT_TYPES)
            df = df[df["Event Type"].isin(valid_types)]
            logger.info(f"Removed {initial_size - len(df)} events with invalid event types")
        
        # Remove events with missing dates
        if "Event Date" in df.columns:
            df = df.dropna(subset=["Event Date"])
            logger.info(f"Dataset has events from {df['Event Date'].min()} to {df['Event Date'].max()}")
        
        # Sort by date
        if "Event Date" in df.columns:
            df = df.sort_values("Event Date").reset_index(drop=True)
        
        return df
    
    def filter_by_country(self, df: pd.DataFrame, country_codes: List[str]) -> pd.DataFrame:
        """
        Filter dataset to specific countries.
        
        Args:
            df: Event DataFrame
            country_codes: List of country codes to keep
            
        Returns:
            Filtered DataFrame
        """
        if "Country Code" not in df.columns:
            logger.warning("Country Code column not found, returning original data")
            return df
        
        initial_size = len(df)
        df = df[df["Country Code"].isin(country_codes)]
        
        logger.info(f"Filtered to countries {country_codes}: {len(df)} events ({initial_size - len(df)} removed)")
        return df
    
    def get_cleaning_statistics(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics about the cleaning process.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: DataFrame after cleaning
            
        Returns:
            Dictionary with cleaning statistics
        """
        stats = {
            "original_size": len(original_df),
            "cleaned_size": len(cleaned_df),
            "reduction_percentage": ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100,
            "original_actors": original_df["Actor Name"].nunique() if "Actor Name" in original_df.columns else 0,
            "cleaned_actors": cleaned_df["Actor Name"].nunique() if "Actor Name" in cleaned_df.columns else 0,
            "original_recipients": original_df["Recipient Name"].nunique() if "Recipient Name" in original_df.columns else 0,
            "cleaned_recipients": cleaned_df["Recipient Name"].nunique() if "Recipient Name" in cleaned_df.columns else 0,
            "original_event_types": original_df["Event Type"].nunique() if "Event Type" in original_df.columns else 0,
            "cleaned_event_types": cleaned_df["Event Type"].nunique() if "Event Type" in cleaned_df.columns else 0,
        }
        
        return stats