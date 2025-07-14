# Archive - Original Implementation

This directory contains the original implementation files that were modularized into the new package structure.

## Files and Directories

### Core Implementation Files
- **calc_metrics.py** - Original evaluation metrics calculations
- **evals.py** - Original evaluation pipeline and agent testing
- **create_agent.py** - Original agent creation logic

### Supporting Modules
- **tools/** - Directory containing the original agent tools
  - `tools.py` - All 6 specialized agent tools (graph analysis, news search, etc.)
- **utils/** - Utility functions directory
  - `load_graph.py` - Original graph loading functionality

### Research and Development Files
- **initial/** - Original research notebooks and experiments
- **test_runs/** - Historical test run results
- **metrics/** - Evaluation metrics from previous experiments
- **prompts/** - System prompts and templates
- **outputs/** - Generated outputs and results

### Additional Files
- **events.txt** - Event data file
- **flowchart.png** - System architecture diagram
- **run_agent.sh** - Shell script for running agents

## Migration Notes

All functionality from these files has been preserved and integrated into the new modularized structure:

- `calc_metrics.py` → `graphrag/evaluation/metrics.py`
- `evals.py` → `graphrag/evaluation/evaluator.py`
- `create_agent.py` → `graphrag/agents/langgraph_agent.py`
- `tools/tools.py` → `graphrag/agents/tools/graph_tools.py`
- `utils/load_graph.py` → `graphrag/graph/serializer.py`

## Usage

These files are kept for reference and historical purposes. The new modularized implementation should be used for all future development.

## Date Archived

July 2024 - During repository modularization for research framework accessibility.