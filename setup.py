"""
Setup configuration for GraphRAG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

# Core dependencies (subset of requirements.txt for minimal installation)
core_requirements = [
    "pandas>=2.0.0",
    "numpy>=1.20.0",
    "networkx>=3.0",
    "scikit-learn>=1.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.60.0",
    "requests>=2.25.0",
    "beautifulsoup4>=4.10.0",
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "langchain-openai>=0.1.0",
    "langgraph>=0.1.0",
    "openai>=1.0.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

# Extra dependencies for specific features
extras_require = {
    "dev": dev_requirements,
    "full": requirements,  # All requirements from requirements.txt
    "evaluation": [
        "evaluate>=0.4.0",
        "rouge_score>=0.1.0",
    ],
    "geopy": [
        "geopy>=2.2.0",
    ],
    "news": [
        "requests>=2.25.0",
        "beautifulsoup4>=4.10.0",
    ]
}

setup(
    name="graphrag",
    version="0.1.0",
    author="Aditya Sampath",
    author_email="your.email@university.edu",
    description="Agentic Reasoning for Social Event Extrapolation: Integrating Knowledge Graphs and Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/GraphRAG",
    project_urls={
        "Bug Reports": "https://github.com/your-username/GraphRAG/issues",
        "Source": "https://github.com/your-username/GraphRAG",
        "Documentation": "https://github.com/your-username/GraphRAG#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "graphrag": ["py.typed"],
        "config": ["*.yaml", "*.yml"],
    },
    entry_points={
        "console_scripts": [
            "graphrag-build=scripts.build_graph:main",
            "graphrag-evaluate=scripts.run_evaluation:main", 
            "graphrag-predict=scripts.predict:main",
        ],
    },
    keywords=[
        "knowledge graphs",
        "event prediction", 
        "temporal graphs",
        "LLM agents",
        "geopolitical analysis",
        "graph neural networks",
        "natural language processing",
        "machine learning",
    ],
    zip_safe=False,
)