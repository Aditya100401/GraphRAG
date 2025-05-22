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

        for u, v, data in graph.edges(data=True):
            if actor in [u, v] or recipient in [u, v]:
                ts = pd.to_datetime(graph.nodes[u].get("event_date") or graph.nodes[v].get("event_date"), errors="coerce")
                if pd.notna(ts) and ts < ref_date:
                    relevant_edges.append((u, data.get("relation", "related_to"), v, ts.strftime("%Y-%m-%d")))

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

        for node in graph.nodes:
            event_meta = graph.nodes[node]
            event_date = pd.to_datetime(event_meta.get("event_date"), errors="coerce")

            if pd.notna(event_date):
                neighbors = list(graph.neighbors(node))
                if actor in neighbors and recipient in neighbors:
                    time_diff = abs((event_date - input_date_dt).days)
                    matching_events.append((node, time_diff, event_meta))

        matching_events.sort(key=lambda x: x[1])
        top_events = matching_events[:5]

        return {
            event[0]: {
                "event_date": str(event[2].get("event_date")),
                "event_text": (event[2].get("event_text") or "")[:300] + "..."
                if event[2].get("event_text") and len(event[2].get("event_text")) > 300 else event[2].get("event_text", "No text available")
            }
            for event in top_events
        }

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

        for u, v, data in graph.edges(data=True):
            if actor in [u, v] or recipient in [u, v]:
                ts = pd.to_datetime(graph.nodes[u].get("event_date") or graph.nodes[v].get("event_date"), errors="coerce")
                if pd.notna(ts) and ts < ref_date:
                    etype = graph.nodes[u].get("event_type") or graph.nodes[v].get("event_type")
                    if etype:
                        event_counts[etype] = event_counts.get(etype, 0) + 1
        return dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    @tool(
        "summarize_actor_recipient_history",
        description=(
            "Return a summary list of up to 10 events that directly connect the given actor and recipient in the knowledge graph. "
            "For each event, include the event ID, event date, and a truncated event text (up to 300 characters). "
            "Use this tool for a quick overview of the shared history between two parties."
        ),
    )
    def summarize_actor_recipient_history(actor: str, recipient: str):
        """
        Returns a list of up to 10 events connecting the actor and recipient.
        Each includes event ID, date, and a truncated event text for quick summarization.
        """
        summaries = []

        for node in graph.nodes:
            meta = graph.nodes[node]
            event_date = pd.to_datetime(meta.get("event_date"), errors="coerce")
            neighbors = list(graph.neighbors(node))

            if actor in neighbors and recipient in neighbors:
                summaries.append({
                    "event_id": node,
                    "event_date": str(event_date.date()) if pd.notna(event_date) else "N/A",
                    "event_text": (meta.get("event_text") or "")[:300] + "..."
                    if meta.get("event_text") and len(meta.get("event_text")) > 300 else meta.get("event_text", "No summary available")
                })

        summaries.sort(key=lambda x: x["event_date"])
        return summaries[:10]

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
            query = ""

        # Calculate date range
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=7)
        api_key = os.getenv("NEWS_API_KEY")
        url = (
            f"https://api.thenewsapi.com/v1/news/all"
            f"?api_token={api_key}"
            f"&search={query}"
            f"&language=en"
            f"&limit=10"
            f"&published_after={start_date.strftime('%Y-%m-%d')}"
            f"&published_before={end_date.strftime('%Y-%m-%d')}"
        )

        response = requests.get(url)
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

    # Return tools as a list
    return [
        get_node_edge_connections_tool,
        print_node_attributes_tool,
        calculate_event_type_frequency_tool,
        search_and_extract_news,
        summarize_actor_recipient_history,
    ]
