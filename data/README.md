# Data Directory

This directory contains the datasets used by the GraphRAG framework.

## Directory Structure

- **country_sets/** - Country-specific event datasets
- **country_specific_graphs/** - Pre-built graphs for different countries
- **final_splits/** - Train/test data splits for evaluation
- **graphs/** - Generated knowledge graphs (excluded from git)
- **Raw_data/** - Raw NGEC corpus data (excluded from git)

## Usage

The framework automatically loads data from these directories based on country codes (e.g., IND, AFG, RUS).

## Data Sources

- NGEC (Next Generation Event Corpus) for geopolitical events
- Country-specific event datasets for temporal analysis
- Pre-processed splits for consistent evaluation

## Note

Large data files (*.pkl, raw datasets) are excluded from version control via .gitignore.