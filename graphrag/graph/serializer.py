"""
Graph serialization utilities for saving and loading graphs.
Based on the original save/load functionality.
"""

import pickle
import networkx as nx
from pathlib import Path
from typing import Union, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class GraphSerializer:
    """Handles saving and loading of NetworkX graphs."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize GraphSerializer.
        
        Args:
            output_dir: Directory for saving graphs. Uses settings default if None.
        """
        self.output_dir = output_dir or settings.GRAPHS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_graph(self, graph: nx.DiGraph, filename: str, 
                   formats: Optional[list] = None) -> dict:
        """
        Save graph in specified formats.
        
        Args:
            graph: NetworkX graph to save
            filename: Base filename (without extension)
            formats: List of formats ['pkl', 'graphml']. Uses settings default if None.
            
        Returns:
            Dictionary with saved file paths
        """
        if formats is None:
            if settings.GRAPH_FORMAT == "both":
                formats = ["pkl", "graphml"]
            else:
                formats = [settings.GRAPH_FORMAT]
        
        saved_files = {}
        
        for fmt in formats:
            try:
                file_path = self.output_dir / f"{filename}.{fmt}"
                
                if fmt == "pkl":
                    self._save_pickle(graph, file_path)
                elif fmt == "graphml":
                    self._save_graphml(graph, file_path)
                else:
                    logger.warning(f"Unsupported format: {fmt}")
                    continue
                
                saved_files[fmt] = file_path
                logger.info(f"Saved {fmt.upper()} graph: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save graph as {fmt}: {e}")
        
        return saved_files
    
    def _save_pickle(self, graph: nx.DiGraph, file_path: Path):
        """Save graph as pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(graph, f)
    
    def _save_graphml(self, graph: nx.DiGraph, file_path: Path):
        """Save graph as GraphML file."""
        # GraphML has some restrictions, so we need to clean the graph
        clean_graph = self._prepare_for_graphml(graph)
        nx.write_graphml(clean_graph, file_path)
    
    def _prepare_for_graphml(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Prepare graph for GraphML export by cleaning problematic attributes.
        GraphML has restrictions on attribute types and values.
        """
        clean_graph = graph.copy()
        
        # Clean node attributes
        for node in clean_graph.nodes():
            attrs = clean_graph.nodes[node]
            cleaned_attrs = {}
            
            for key, value in attrs.items():
                # Convert problematic types to strings
                if value is None:
                    cleaned_attrs[key] = ""
                elif isinstance(value, (int, float, str, bool)):
                    cleaned_attrs[key] = value
                else:
                    cleaned_attrs[key] = str(value)
            
            # Update node attributes
            clean_graph.nodes[node].clear()
            clean_graph.nodes[node].update(cleaned_attrs)
        
        # Clean edge attributes
        for u, v in clean_graph.edges():
            attrs = clean_graph.edges[u, v]
            cleaned_attrs = {}
            
            for key, value in attrs.items():
                if value is None:
                    cleaned_attrs[key] = ""
                elif isinstance(value, (int, float, str, bool)):
                    cleaned_attrs[key] = value
                else:
                    cleaned_attrs[key] = str(value)
            
            # Update edge attributes
            clean_graph.edges[u, v].clear()
            clean_graph.edges[u, v].update(cleaned_attrs)
        
        return clean_graph
    
    def load_graph(self, filename: str, format: str = "pkl") -> nx.DiGraph:
        """
        Load graph from file.
        
        Args:
            filename: Filename (with or without extension)
            format: File format ('pkl' or 'graphml')
            
        Returns:
            Loaded NetworkX graph
        """
        # Handle filename with or without extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")
        
        try:
            if format == "pkl":
                return self._load_pickle(file_path)
            elif format == "graphml":
                return self._load_graphml(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to load graph from {file_path}: {e}")
            raise
    
    def _load_pickle(self, file_path: Path) -> nx.DiGraph:
        """Load graph from pickle file."""
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        
        logger.info(f"Loaded pickle graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    def _load_graphml(self, file_path: Path) -> nx.DiGraph:
        """Load graph from GraphML file."""
        graph = nx.read_graphml(file_path, node_type=str)
        
        # Convert back to DiGraph if needed
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
        
        logger.info(f"Loaded GraphML graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    def save_country_graph(self, graph: nx.DiGraph, country: str, split: str = "train") -> dict:
        """
        Save graph with standardized country naming.
        
        Args:
            graph: Graph to save
            country: Country code
            split: Data split ('train' or 'test')
            
        Returns:
            Dictionary with saved file paths
        """
        filename = f"graph_{country}_{split}"
        return self.save_graph(graph, filename)
    
    def load_country_graph(self, country: str, split: str = "train", 
                          format: str = "pkl") -> nx.DiGraph:
        """
        Load graph with standardized country naming.
        
        Args:
            country: Country code
            split: Data split
            format: File format
            
        Returns:
            Loaded graph
        """
        filename = f"graph_{country}_{split}"
        return self.load_graph(filename, format)
    
    def list_available_graphs(self) -> dict:
        """
        List all available graph files.
        
        Returns:
            Dictionary organized by format and country
        """
        available = {"pkl": [], "graphml": []}
        
        for format in ["pkl", "graphml"]:
            for file_path in self.output_dir.glob(f"*.{format}"):
                filename = file_path.stem
                available[format].append(filename)
        
        return available
    
    def get_graph_info(self, filename: str, format: str = "pkl") -> dict:
        """
        Get basic information about a graph file without fully loading it.
        
        Args:
            filename: Graph filename
            format: File format
            
        Returns:
            Dictionary with basic graph information
        """
        file_path = self.output_dir / f"{filename}.{format}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")
        
        info = {
            "filename": filename,
            "format": format,
            "file_size": file_path.stat().st_size,
            "file_path": str(file_path),
            "last_modified": file_path.stat().st_mtime
        }
        
        # Try to get graph stats without loading full graph
        try:
            if format == "pkl":
                # For pickle, we need to load to get stats
                graph = self.load_graph(filename, format)
                info.update({
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "is_directed": graph.is_directed()
                })
            elif format == "graphml":
                # GraphML files can be parsed for basic info
                import xml.etree.ElementTree as ET
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Count nodes and edges from XML
                nodes = len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}node"))
                edges = len(root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"))
                
                info.update({
                    "nodes": nodes,
                    "edges": edges,
                    "is_directed": True  # Assume directed for now
                })
                
        except Exception as e:
            logger.warning(f"Could not extract graph statistics: {e}")
            info["error"] = str(e)
        
        return info

# Convenience function for backward compatibility
def load_graph(file_path: Union[str, Path]) -> nx.DiGraph:
    """
    Load graph from file path (backward compatibility).
    
    Args:
        file_path: Path to graph file
        
    Returns:
        Loaded NetworkX graph
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Graph file not found: {file_path}")
    
    if file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_path.suffix == ".graphml":
        return nx.read_graphml(file_path, node_type=str)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")