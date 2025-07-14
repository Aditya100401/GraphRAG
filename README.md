# Agentic Reasoning for Social Event Extrapolation: Integrating Knowledge Graphs and Language Models

A comprehensive framework for geopolitical event prediction using temporal knowledge graphs and LLM agents with specialized tools.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

## 🎯 Overview

This is a modular framework that combines temporal knowledge graphs with LLM agents to predict geopolitical events. The framework processes large-scale event data (NGEC corpus), builds temporal knowledge graphs, and uses intelligent agents with specialized tools for event prediction.

### Key Features

- **Temporal Knowledge Graphs**: Build graphs connecting actors, events, and contexts over time
- **Intelligent Agents**: LLM agents with specialized graph query tools
- **Comprehensive Evaluation**: Multiple metrics including Hit@k, MRR, ROUGE, and custom domain metrics
- **Modular Design**: Easy to extend and customize for research needs
- **Production Ready**: CLI tools, proper logging, and configuration management

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Aditya100401/GraphRAG
cd GraphRAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export NEWS_API_KEY="your-news-api-key"  # Optional for news tools
```

### Basic Usage

```python
from graphrag.data import DataLoader
from graphrag.graph import GraphBuilder
from graphrag.agents import LangGraphAgent
from langchain_openai import ChatOpenAI

# 1. Load data
loader = DataLoader()
train_data = loader.load_country_data("IND", "train")

# 2. Build graph
builder = GraphBuilder()
graph = builder.build_temporal_kg(train_data)

# 3. Create agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LangGraphAgent(graph, model)

# 4. Make prediction
result = agent.run(
    query="What are the likely follow-up events between India and Pakistan?",
    actor="India",
    recipient="Pakistan",
    date="2024-01-15"
)

predictions = agent.extract_predictions(result)
print(f"Predictions: {predictions}")
```

### CLI Usage

```bash
# Build knowledge graph
python scripts/build_graph.py --country IND --split train

# Run evaluation
python scripts/evaluate.py --country IND --graph-path data/graphs/graph_IND_train.pkl

# Make single prediction
python scripts/predict.py --graph-path data/graphs/graph_IND_train.pkl \
  --query "What will happen next?" --actor "India" --recipient "Pakistan"
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)

## 🏗️ Architecture

The framework consists of five main modules:

### 1. Data Processing (`graphrag.data`)

- **DataLoader**: Load raw and processed event data
- **DataCleaner**: Clean and preprocess event data with geocoding
- **DataSplitter**: Create temporal train/test splits

### 2. Graph Construction (`graphrag.graph`)

- **GraphBuilder**: Build temporal knowledge graphs from event data
- **GraphSerializer**: Save and load graphs in multiple formats

### 3. Agent System (`graphrag.agents`)

- **BaseAgent**: Abstract base class for all agents
- **LangGraphAgent**: LangGraph-based agent implementation
- **Tools**: Specialized graph query tools for agents

### 4. Evaluation Framework (`graphrag.evaluation`)

- **EventPredictionMetrics**: Comprehensive metrics calculation
- **Evaluator**: End-to-end evaluation pipeline

### 5. Configuration (`config`)

- **settings.py**: Centralized configuration management
- **system_prompt.yaml**: LLM system prompts

## 🔧 Agent Tools

The framework includes six specialized tools for LLM agents:

1. **get_node_edge_connections**: Retrieve historical interactions
2. **print_node_attributes**: Get detailed event metadata
3. **calculate_event_type_frequency**: Analyze historical patterns
4. **summarize_actor_recipient_history**: Get relationship timeline
5. **get_actor_timeline**: Track actor activity over time
6. **search-news**: Search and scrape news articles

## 📊 Evaluation Metrics

### Core Metrics

- **Hit@k**: Accuracy at ranks 1, 3, 10
- **MRR**: Mean Reciprocal Rank
- **ROUGE-1**: Token overlap scores
- **F1/Precision/Recall**: Multi-class classification metrics

### Advanced Metrics

- **Intensity-stratified**: Performance by event intensity levels
- **Event Coverage**: Prediction diversity and coverage
- **Temporal Consistency**: Logic violation analysis
- **Position Analysis**: Confidence calibration

## 📁 Project Structure

```text
GraphRAG/
├── graphrag/                    # Main package
│   ├── data/                   # Data processing
│   ├── graph/                  # Graph construction
│   ├── agents/                 # Agent system
│   ├── evaluation/             # Evaluation framework
│   └── utils/                  # Utilities
├── archive/                    # Previous Implementation
├── config/                     # Configuration 
├── scripts/                    # CLI scripts
├── examples/                   # Usage examples
├── docs/                       # Documentation
└── data/                       # Data storage
```

## 🧪 Examples

### QuickStart Example

```bash
python examples/quickstart.py
```

### Custom Agent Development

```bash
python examples/custom_agent.py
```

### Batch Evaluation

```bash
python examples/batch_evaluation.py
```

## 📈 Research Applications

This framework supports research in:

- **Temporal Knowledge Graphs**: Graph-based event modeling
- **Event Prediction**: Geopolitical forecasting
- **LLM Agents**: Tool-augmented language models
- **Evaluation Metrics**: Domain-specific assessment
- **Multi-modal Integration**: Combining structured and unstructured data

### Development Setup

1. Clone and install in development mode:

```bash
git clone https://github.com/Aditya100401/GraphRAG
cd GraphRAG
pip install -e ".[dev]"
```

2. Run tests:

```bash
pytest tests/
```

3. Format code:

```bash
black graphrag/
isort graphrag/
```

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Agentic Reasoning for Social Event Extrapolation: Integrating Knowledge Graphs and Language Models},
  author={},
  journal={},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [LangGraph](https://github.com/langchain-ai/langgraph): Agent workflow framework
- [NetworkX](https://networkx.org/): Graph analysis library
- [POLECAT](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AJGVIT): Dataset

## 📞 Support

- **Email**: <asampat1@charlotte.edu>

---

## Built with ❤️ for the research community
