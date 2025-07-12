from langchain.tools import tool
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()

def get_tools(graph):
    @tool(
        "get_node_edge_connections",
        description=(
            "Retrieve up to 10 past events from the knowledge graph where either the specified actor or recipient participated, before the provided date. "
            "For each event, return the source entity, the relationship type, the target entity, and the event date (as a timestamp). "
            "Use this tool to understand the historical interactions involving the given parties."
        ),
    )
    def get_node_edge_connections_tool(actor: str, recipient: str, date: str):
        """
        Returns up to 10 edges where the actor or recipient is involved before the given date.
        Each result includes source, relation type, target, and timestamp.
        """
        ref_date = pd.to_datetime(date)
        relevant_edges = []

        # Check edges connected to event nodes
        for u, v, data in graph.edges(data=True):
            # Get event nodes that connect to actor or recipient
            event_node = None
            connected_entity = None
            
            # Check if u is an event node and v is actor/recipient
            if graph.nodes[u].get("node_type") == "event" and v in [actor, recipient]:
                event_node = u
                connected_entity = v
            # Check if v is an event node and u is actor/recipient  
            elif graph.nodes[v].get("node_type") == "event" and u in [actor, recipient]:
                event_node = v
                connected_entity = u
                
            if event_node:
                event_date_str = graph.nodes[event_node].get("event_date")
                if event_date_str:
                    event_ts = pd.to_datetime(event_date_str, errors="coerce")
                    if pd.notna(event_ts) and event_ts < ref_date:
                        relevant_edges.append((
                            u, 
                            data.get("relation", "related_to"), 
                            v, 
                            event_date_str
                        ))

        # Sort by date and return top 10
        relevant_edges.sort(key=lambda x: x[3], reverse=True)
        return relevant_edges[:10]

    @tool(
        "print_node_attributes",
        description=(
            "Get metadata for up to 5 events from the knowledge graph where both the specified actor and recipient are directly connected. "
            "For each event, provide the event date and a summary (up to 300 characters) of the event text. "
            "Use this tool to discover key events linking both parties and to review the context of their interaction."
        ),
    )
    def print_node_attributes_tool(actor: str, recipient: str, date: str):
        """
        Returns metadata for up to 5 event nodes where both actor and recipient are connected.
        Includes event date and a truncated event text field.
        """
        input_date_dt = pd.to_datetime(date)
        matching_events = []

        # Find event nodes that connect to both actor and recipient
        for node in graph.nodes:
            node_data = graph.nodes[node]
            if node_data.get("node_type") == "event":
                event_date_str = node_data.get("event_date")
                if event_date_str:
                    event_date = pd.to_datetime(event_date_str, errors="coerce")
                    
                    if pd.notna(event_date):
                        # Check if both actor and recipient are connected to this event
                        neighbors = list(graph.neighbors(node))
                        if actor in neighbors and recipient in neighbors:
                            time_diff = abs((event_date - input_date_dt).days)
                            matching_events.append((node, time_diff, node_data))

        # Sort by time difference and return top 5
        matching_events.sort(key=lambda x: x[1])
        top_events = matching_events[:5]

        result = {}
        for event in top_events:
            event_id, _, event_meta = event
            # Note: POLECAT data doesn't have 'event_text' field based on create_graph.py
            # Using available fields instead
            event_info = {
                "event_date": event_meta.get("event_date", "N/A"),
                "event_type": event_meta.get("event_type", "N/A"),
                "place": event_meta.get("place", "N/A"),
                "contexts": event_meta.get("contexts", "N/A"),
                "quad_code": event_meta.get("quad_code", "N/A")
            }
            result[event_id] = event_info
            
        return result

    @tool(
        "calculate_event_type_frequency",
        description=(
            "Analyze the knowledge graph and return the 10 most common event types that involve either the specified actor or recipient, before the given date. "
            "For each event type, return its frequency count. "
            "Use this tool to identify patterns and the most likely types of past events between the given parties."
        ),
    )
    def calculate_event_type_frequency_tool(actor: str, recipient: str, date: str):
        """
        Counts and returns up to 10 most frequent event types involving the actor or recipient before the given date.
        """
        ref_date = pd.to_datetime(date)
        event_counts = {}

        # Look for event nodes connected to actor or recipient
        for node in graph.nodes:
            node_data = graph.nodes[node]
            if node_data.get("node_type") == "event":
                event_date_str = node_data.get("event_date")
                if event_date_str:
                    event_ts = pd.to_datetime(event_date_str, errors="coerce")
                    
                    if pd.notna(event_ts) and event_ts < ref_date:
                        # Check if actor or recipient is connected to this event
                        neighbors = list(graph.neighbors(node))
                        if actor in neighbors or recipient in neighbors:
                            event_type = node_data.get("event_type")
                            if event_type and event_type.strip():
                                event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    @tool(
        "summarize_actor_recipient_history",
        description=(
            "Return a summary list of up to 10 events that directly connect the given actor and recipient in the knowledge graph. "
            "For each event, include the event ID, event date, event type, location, and contexts. "
            "Use this tool for a quick overview of the shared history between two parties."
        ),
    )
    def summarize_actor_recipient_history(actor: str, recipient: str):
        """
        Returns a list of up to 10 events connecting the actor and recipient.
        Each includes event ID, date, type, location, and contexts for quick summarization.
        """
        summaries = []

        # Find events that connect both actor and recipient
        for node in graph.nodes:
            node_data = graph.nodes[node]
            if node_data.get("node_type") == "event":
                neighbors = list(graph.neighbors(node))
                
                if actor in neighbors and recipient in neighbors:
                    event_date_str = node_data.get("event_date")
                    summaries.append({
                        "event_id": node,
                        "event_date": event_date_str if event_date_str else "N/A",
                        "event_type": node_data.get("event_type", "N/A"),
                        "location": node_data.get("place", "N/A"),
                        "contexts": node_data.get("contexts", "N/A"),
                        "quad_code": node_data.get("quad_code", "N/A")
                    })

        # Sort by date (handle None/empty dates)
        def sort_key(x):
            try:
                return pd.to_datetime(x["event_date"]) if x["event_date"] != "N/A" else pd.Timestamp.min
            except:
                return pd.Timestamp.min
                
        summaries.sort(key=sort_key, reverse=True)
        return summaries[:10]

    @tool(
        "get_actor_timeline",
        description=(
            "Get a chronological timeline of events for a specific actor before the given date. "
            "Returns up to 15 events in temporal order with event details. "
            "Use this tool to understand an actor's recent activity patterns."
        ),
    )
    def get_actor_timeline(actor: str, date: str):
        """
        Returns chronological timeline of events for the specified actor.
        """
        ref_date = pd.to_datetime(date)
        actor_events = []
        
        for node in graph.nodes:
            node_data = graph.nodes[node]
            if node_data.get("node_type") == "event":
                neighbors = list(graph.neighbors(node))
                
                if actor in neighbors:
                    event_date_str = node_data.get("event_date")
                    if event_date_str:
                        event_date = pd.to_datetime(event_date_str, errors="coerce")
                        if pd.notna(event_date) and event_date < ref_date:
                            actor_events.append({
                                "event_id": node,
                                "event_date": event_date_str,
                                "event_type": node_data.get("event_type", "N/A"),
                                "location": node_data.get("place", "N/A"),
                                "contexts": node_data.get("contexts", "N/A"),
                                "timestamp": event_date.timestamp()
                            })
        
        # Sort by timestamp (most recent first)
        actor_events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Remove timestamp from output
        for event in actor_events:
            del event["timestamp"]
            
        return actor_events[:15]

    @tool(
        "search-news",
        description=(
            "Search for up to 10 recent news articles about the specified actor and/or recipient, published in the week before the given date. "
            "For each article, return the title, up to 500 characters of content, URL, and publication date. "
            "Use this tool to find current events or public reporting related to the parties."
        ),
    )
    def search_and_extract_news(actor: str, recipient: str, date: str):
        """
        Searches for news articles about the actor and recipient from the week before the given date.
        """
        # Build the query
        if actor and recipient:
            query = f"{actor} AND {recipient}"
        elif actor:
            query = actor
        elif recipient:
            query = recipient
        else:
            return {"error": "At least one of actor or recipient must be provided"}

        # Calculate date range
        try:
            end_date = datetime.strptime(date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=7)
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
            
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return {"error": "NEWS_API_KEY not found in environment variables"}
            
        url = (
            f"https://api.thenewsapi.com/v1/news/all"
            f"?api_token={api_key}"
            f"&search={query}"
            f"&language=en"
            f"&limit=10"
            f"&published_after={start_date.strftime('%Y-%m-%d')}"
            f"&published_before={end_date.strftime('%Y-%m-%d')}"
        )

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                articles = response.json().get('data', [])
                results = []
                for article in articles:
                    article_url = article.get("url")
                    if not article_url:
                        continue
                    content = scrape_content(article_url)
                    if content:
                        results.append({
                            "title": article.get("title"),
                            "content": content[:500],
                            "url": article_url,
                            "published_at": article.get("published_at")
                        })
                return {"articles": results} if results else {"message": "No articles found for the specified criteria."}
            else:
                return {"error": f"Error fetching news articles: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def scrape_content(url):
        """
        Scrapes and cleans article content from the given URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs)
            content = ' '.join(content.split())  # Clean whitespace
            return content
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch the URL {url}: {e}")
            return None
        except Exception as e:
            print(f"Error parsing content from {url}: {e}")
            return None

    # Return tools as a list
    return [
        get_node_edge_connections_tool,
        print_node_attributes_tool,
        calculate_event_type_frequency_tool,
        summarize_actor_recipient_history,
        get_actor_timeline,
        search_and_extract_news,
    ]