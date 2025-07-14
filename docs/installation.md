# Installation Guide

This guide will help you install and set up the GraphRAG framework.

## Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key (for LLM functionality)
- At least 8GB RAM (for processing large graphs)
- 5GB free disk space (for data and model caches)

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/GraphRAG.git
cd GraphRAG
```

### 2. Create Virtual Environment

```bash
python -m venv graphrag-env
source graphrag-env/bin/activate  # On Windows: graphrag-env\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Minimal Installation

```bash
pip install -e .
```

#### Option B: Full Installation (Recommended)

```bash
pip install -e ".[full]"
```

#### Option C: Development Installation

```bash
pip install -e ".[dev]"
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env  # If example exists
# Or create manually:
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
echo "NEWS_API_KEY=your-news-api-key-here" >> .env  # Optional
```

### 5. Verify Installation

```bash
python -c "import graphrag; print('GraphRAG installed successfully!')"
```

## Detailed Installation Options

### Dependencies Explained

The framework has several dependency groups:

- **Core**: Essential packages for basic functionality
- **Full**: All packages from requirements.txt (recommended for research)
- **Dev**: Additional tools for development (testing, formatting, etc.)
- **Evaluation**: Extra packages for comprehensive evaluation metrics
- **Geopy**: Geographic processing capabilities
- **News**: News article retrieval and processing

### Custom Installation

```bash
# Install with specific extras
pip install -e ".[evaluation,geopy]"

# Or install from requirements.txt directly
pip install -r requirements.txt
pip install -e .
```

## Configuration

### Environment Variables

The framework uses the following environment variables:

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM models | None |
| `NEWS_API_KEY` | No | News API key for article retrieval | None |
| `GRAPHRAG_MODEL` | No | Default LLM model | gpt-4o-mini |
| `GRAPHRAG_TEMPERATURE` | No | Default model temperature | 0.0 |
| `GRAPHRAG_AGENT_TYPE` | No | Default agent type | langgraph |

*Required for agent functionality

### Data Directory Structure

After installation, set up your data directories:

```bash
mkdir -p data/{Raw_data,country_sets,graphs,final_splits}
mkdir -p outputs logs
```

Expected structure:

```bash
data/
├── Raw_data/           # Raw NGEC files
├── country_sets/       # Country-specific data
├── graphs/            # Generated knowledge graphs
└── final_splits/      # Train/test splits
outputs/               # Evaluation results
logs/                 # Application logs
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# If you get import errors, ensure you're in the right environment
which python
pip list | grep graphrag
```

#### Memory Issues

```bash
# For large graphs, increase available memory
export PYTHONHASHSEED=0
ulimit -v 8000000  # Limit virtual memory to 8GB
```

#### Permission Errors

```bash
# On Unix systems, you might need to make scripts executable
chmod +x scripts/*.py
```

### Platform-Specific Notes

#### Windows

- Use `Scripts\activate` instead of `bin/activate`
- Install Microsoft C++ Build Tools if compilation fails
- Consider using Windows Subsystem for Linux (WSL)

#### macOS

- Ensure Xcode command line tools are installed: `xcode-select --install`
- If using Apple Silicon, some packages might need Rosetta 2

#### Linux

- Install system dependencies: `sudo apt-get install build-essential python3-dev`
- For Ubuntu/Debian: `sudo apt-get install graphviz graphviz-dev`

## Verification

### Test Basic Functionality

```python
# Test data loading
from graphrag.data import DataLoader
loader = DataLoader()
print("✓ Data module working")

# Test graph building
from graphrag.graph import GraphBuilder
builder = GraphBuilder()
print("✓ Graph module working")

# Test agent (requires API key)
try:
    from graphrag.agents import LangGraphAgent
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o-mini")
    print("✓ Agent module working")
except Exception as e:
    print(f"⚠ Agent module needs API key: {e}")
```

### Run Examples

```bash
# Test with the quickstart example
python examples/quickstart.py

# Test CLI tools
python scripts/build_graph.py --help
python scripts/evaluate.py --help
python scripts/predict.py --help
```

## Next Steps

1. **Get Data**: Obtain NGEC event data files
2. **Process Data**: Run data cleaning and splitting
3. **Build Graphs**: Create knowledge graphs for your countries
4. **Try Examples**: Run the provided examples
5. **Read Documentation**: Check out the API reference and tutorials
