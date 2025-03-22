from langchain.tools import tool
from pydantic import BaseModel, Field


from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os

from utils.load_graph import load_graph

from dotenv import load_dotenv
load_dotenv()

# Load graph as a global variable to be used across functions
G = load_graph("KG/new_graph.pkl")

class ToolsInput(BaseModel):
    actor: str = Field(..., description="Name of the actor.")
    recipient: str = Field(..., description="Name of the recipient.")
    date: str = Field(..., description="Date in YYYY-MM-DD format to filter events.")

@tool("node-edge-connections", args_schema=ToolsInput)
def get_node_edge_connections_tool(actor: str, recipient: str, date: str):
    """
    Returns up to 15 node-edge connections from the graph where either the actor or recipient is involved,
    and the event date is before the provided date.
    """
    relevant_edges = []
    for u, v, data in G.edges(data=True):
        # Ensure proper grouping of conditions with parentheses.
        if (actor in [u, v] or recipient in [u, v]) and (data.get('relation') == 'actor' or data.get('relation') == 'recipient'):
            event_date = G.nodes[u].get('event_date', G.nodes[v].get('event_date'))
            if event_date and datetime.strptime(event_date, "%Y-%m-%d") < datetime.strptime(date, "%Y-%m-%d"):
                relevant_edges.append((u, v, data))
    return relevant_edges[:15]

@tool("print-node-attributes", args_schema=ToolsInput)
def print_node_attributes_tool(actor: str, recipient: str, date: str):
    """
    Retrieves attributes for event nodes where both the actor and recipient are connected.
    Returns a dictionary of the top 5 events (sorted by time difference from the input date).
    """
    input_date_dt = datetime.strptime(date, "%Y-%m-%d")
    matching_events = []
    for node in G.nodes:
        if 'event_date' in G.nodes[node]:
            event_date = datetime.strptime(G.nodes[node]['event_date'], "%Y-%m-%d")
            actor_neighbors = list(G.neighbors(node))
            recipient_neighbors = list(G.neighbors(node))
            if actor in actor_neighbors and recipient in recipient_neighbors:
                time_diff = abs((event_date - input_date_dt).days)
                matching_events.append((node, time_diff, G.nodes[node]))
    matching_events.sort(key=lambda x: x[1])
    top_events = matching_events[:5]
    return {event[0]: event[2] for event in top_events}

@tool("event-type-frequency", args_schema=ToolsInput)
def calculate_event_type_frequency_tool(actor: str, recipient: str, date: str):
    """
    Calculates how many events of each type have occurred before the specified date,
    based on edges involving the given actor or recipient.
    """
    event_counts = {}
    for u, v, data in G.edges(data=True):
        if (actor in [u, v] or recipient in [u, v]):
            event_date = G.nodes[u].get('event_date', G.nodes[v].get('event_date'))
            if event_date and datetime.strptime(event_date, "%Y-%m-%d") < datetime.strptime(date, "%Y-%m-%d"):
                event_type = G.nodes[u].get('event_type', G.nodes[v].get('event_type'))
                if event_type:
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
    return event_counts

@tool("search-news", args_schema=ToolsInput)
def search_and_extract_news(actor: str, recipient: str, date: str):
    """
    Searches for news articles based on the actor, recipient, and date.
    """
    # Parse the date and create date range 
    target_date = datetime.strptime(date, "%Y-%m-%d")
    from_date = (target_date - timedelta(days=15)).strftime("%Y-%m-%d")
    to_date = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")
    
    if actor=="":
        query = f"{actor}"
    elif recipient=="":
        query = f"{recipient}"
    else:
        query=f"{actor} and {recipient}"
        
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        results = []
        for article in articles[:3]:  
            article_url = article.get("url")
            if not article_url:
                continue
            content = scrape_content(article_url)
            if content:
                results.append({
                    "title": article.get("title"),
                    "content": content[:500],  
                    "url": article_url
                })
        print(results)
        return {"articles": results} if results else {"message": "No articles found for the specified criteria."}
    else:
        return {"error": f"Error fetching news articles: {response.status_code}"}


def scrape_content(url):
    """
    Scrapes and cleans article content from the given URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)
        content = ' '.join(content.split())
        return content
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the URL {url}: {e}")
        return None