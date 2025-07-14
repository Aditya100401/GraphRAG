"""
Graph building utilities for temporal knowledge graphs.
Based on the original create_graph.py implementation.
"""

import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds temporal knowledge graphs from event data."""
    
    def __init__(self, include_temporal_edges: bool = None):
        """
        Initialize GraphBuilder.
        
        Args:
            include_temporal_edges: Whether to include temporal "next_event" edges
        """
        self.include_temporal_edges = include_temporal_edges if include_temporal_edges is not None else settings.INCLUDE_TEMPORAL_EDGES
        
    def build_temporal_kg(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build temporal knowledge graph from event data.
        
        Args:
            df: DataFrame with event data
            
        Returns:
            NetworkX DiGraph with temporal relationships
        """
        logger.info(f"Building temporal knowledge graph from {len(df)} events")
        
        G = nx.DiGraph()
        df = df.copy()
        df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
        
        # Track events by actor for temporal linking
        actor_events = {}
        
        # Process each event
        for _, row in df.iterrows():
            self._add_event_node(G, row)
            self._add_entity_nodes(G, row)
            
            # Track for temporal relationships
            if self.include_temporal_edges:
                actor = self._get_actor_name(row)
                if actor and pd.notna(row['Event Date']):
                    actor_events.setdefault(actor, []).append((row['Event Date'], str(row['Event ID'])))
        
        # Add temporal edges between events for each actor
        if self.include_temporal_edges:
            self._add_temporal_edges(G, actor_events)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _add_event_node(self, G: nx.DiGraph, row: pd.Series):
        """Add event node with all metadata."""
        eid = str(row['Event ID'])
        date = row['Event Date']
        date_str = date.strftime('%Y-%m-%d') if pd.notna(date) else ''
        ts = date.timestamp() if pd.notna(date) else None
        
        # Build event node attributes
        event_attrs = {
            'node_type': 'event',
            'event_date': date_str,
            'event_ts': float(ts) if ts is not None else None,
            'event_type': str(row.get('Event Type', '')),
            'place': str(row.get('Raw Placename', '')),
            'quad_code': str(row['Quad Code']) if pd.notna(row.get('Quad Code')) else "",
            'contexts': self._process_contexts(row.get('Contexts', '')),
        }
        
        # Add optional fields if they exist
        optional_fields = ['Event Intensity', 'Country Code', 'Event Text']
        for field in optional_fields:
            if field in row and pd.notna(row[field]):
                event_attrs[field.lower().replace(' ', '_')] = row[field]
        
        G.add_node(eid, **event_attrs)
    
    def _add_entity_nodes(self, G: nx.DiGraph, row: pd.Series):
        """Add entity nodes (actors, recipients, locations, etc.) and connect to event."""
        eid = str(row['Event ID'])
        
        # Add actor
        actor_name = self._get_actor_name(row)
        if actor_name:
            self._add_entity(G, eid, actor_name, "actor", row.get('Actor Title', ''))
        
        # Add recipient
        recipient_name = self._get_recipient_name(row)
        if recipient_name:
            self._add_entity(G, eid, recipient_name, "recipient", row.get('Recipient Title', ''))
        
        # Add event type
        event_type = row.get('Event Type')
        if event_type:
            self._add_entity(G, eid, event_type, "eventType")
        
        # Add location
        location = row.get('Raw Placename')
        if location:
            self._add_entity(G, eid, location, "location")
        
        # Add source if available
        source = row.get('Source')
        if source:
            self._add_entity(G, eid, source, "source")
    
    def _add_entity(self, G: nx.DiGraph, event_id: str, name: Any, role: str, extra_title: str = ""):
        """Helper to add entity node and connect to event."""
        n = str(name).strip()
        if n and n.lower() not in ("none", "nan", ""):
            # Add entity node if it doesn't exist
            if not G.has_node(n):
                G.add_node(n, node_type=role, title=str(extra_title))
            
            # Add edge from event to entity
            G.add_edge(event_id, n, relation=role)
    
    def _add_temporal_edges(self, G: nx.DiGraph, actor_events: Dict[str, List[Tuple]]):
        """Add temporal 'next_event' edges between consecutive events for each actor."""
        temporal_edges_added = 0
        
        for actor, events in actor_events.items():
            # Sort events by timestamp
            timeline = sorted(events, key=lambda x: x[0])
            
            # Connect consecutive events
            for i in range(len(timeline) - 1):
                _, e1 = timeline[i]
                _, e2 = timeline[i + 1]
                G.add_edge(e1, e2, relation="next_event", actor=str(actor))
                temporal_edges_added += 1
        
        logger.info(f"Added {temporal_edges_added} temporal edges")
    
    def _get_actor_name(self, row: pd.Series) -> Optional[str]:
        """Get actor name with fallback to country."""
        actor = row.get('Actor Name') or row.get('Actor Country')
        return str(actor).strip() if actor and pd.notna(actor) else None
    
    def _get_recipient_name(self, row: pd.Series) -> Optional[str]:
        """Get recipient name with fallback to country."""
        recipient = row.get('Recipient Name') or row.get('Recipient Country')
        return str(recipient).strip() if recipient and pd.notna(recipient) else None
    
    def _process_contexts(self, contexts: Any) -> str:
        """Process and clean context field."""
        if not contexts or pd.isna(contexts):
            return ""
        
        context_str = str(contexts)
        # Split by semicolon and clean
        context_list = [c.strip() for c in context_str.split(';') if c.strip()]
        return ";".join(context_list)
    
    def build_country_graph(self, country: str, split: str = "train") -> nx.DiGraph:
        """
        Build graph for a specific country and data split.
        
        Args:
            country: Country code (e.g., 'AFG', 'IND', 'RUS')
            split: Data split ('train' or 'test')
            
        Returns:
            NetworkX DiGraph for the country
        """
        from ..data.loader import DataLoader
        
        loader = DataLoader()
        df = loader.load_country_data(country, split)
        
        return self.build_temporal_kg(df)
    
    def build_multi_country_graphs(self, countries: List[str], 
                                  split: str = "train") -> Dict[str, nx.DiGraph]:
        """
        Build graphs for multiple countries.
        
        Args:
            countries: List of country codes
            split: Data split to use
            
        Returns:
            Dictionary mapping country to graph
        """
        graphs = {}
        
        for country in countries:
            try:
                graph = self.build_country_graph(country, split)
                graphs[country] = graph
                logger.info(f"Built {country} graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            except Exception as e:
                logger.error(f"Failed to build graph for {country}: {e}")
                graphs[country] = nx.DiGraph()  # Empty graph as fallback
        
        return graphs
    
    def get_graph_statistics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary with graph statistics
        """
        # Basic stats
        stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
        }
        
        # Node type distribution
        node_types = {}
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats['node_types'] = node_types
        
        # Edge type distribution
        edge_types = {}
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats['edge_types'] = edge_types
        
        # Temporal statistics
        event_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'event']
        if event_nodes:
            dates = []
            for node in event_nodes:
                date_str = G.nodes[node].get('event_date')
                if date_str:
                    try:
                        dates.append(pd.to_datetime(date_str))
                    except:
                        pass
            
            if dates:
                stats['temporal_range'] = {
                    'start': min(dates),
                    'end': max(dates),
                    'span_days': (max(dates) - min(dates)).days
                }
        
        # Connectivity stats
        if G.number_of_nodes() > 0:
            try:
                # Calculate degree statistics for directed graphs
                in_degrees = [d for n, d in G.in_degree()]
                out_degrees = [d for n, d in G.out_degree()]
                
                stats['degree_stats'] = {
                    'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
                    'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
                    'max_in_degree': max(in_degrees) if in_degrees else 0,
                    'max_out_degree': max(out_degrees) if out_degrees else 0,
                }
            except Exception as e:
                logger.warning(f"Could not calculate degree statistics: {e}")
        
        return stats