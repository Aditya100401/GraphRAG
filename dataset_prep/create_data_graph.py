# clean_data.py

import pandas as pd
import networkx as nx
import pickle
from collections import Counter

def clean_polecat_df(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop(columns=[
        'Feed', 'Headline', 'Version', 'Story People', 
        'Story Organizations', 'Story Locations', 'Publication Date'
    ], errors='ignore')

    df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")

    for col in ["Actor Name", "Recipient Name", "Raw Placename"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    df = df[~((df["Actor Name"] == "None") & (df["Recipient Name"] == "None"))]

    return df


def split_by_date(df, date_column="Event Date", min_count=1):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    pairs = list(zip(df["Actor Name"], df["Recipient Name"]))
    pair_counts = Counter(pairs)
    valid_pairs = {pair for pair, count in pair_counts.items() if count >= min_count}
    df = df[df.apply(lambda row: (row["Actor Name"], row["Recipient Name"]) in valid_pairs, axis=1)]

    train_data, test_data = [], []
    grouped = df.groupby(["Actor Name", "Recipient Name"])

    for (_, _), group in grouped:
        group = group.sort_values(by=date_column)
        test_data.append(group.tail(5))
        train_data.append(group.head(len(group) - 5))

    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)
    return train_df, test_df


def build_graph_from_dataframe(df: pd.DataFrame):
    G = nx.Graph()

    def add_event_to_graph(row):
        date = row['Event Date']
        event_id = row['Event ID']
        event_type = row['Event Type']
        place = row['Raw Placename']
        actor = row.get('Actor Name') or row.get('Actor Country')
        recipient = row.get('Recipient Name') or row.get('Recipient Country')
        event_text = row.get('Event Text')

        # Fallback if event_text is empty or null
        if pd.isna(event_text) or event_text in ["", "None"]:
            event_text = f"{actor} {event_type} {recipient} in {place} on {date.strftime('%Y-%m-%d') if pd.notna(date) else 'unknown date'}"

        G.add_node(event_id, event_date=date, event_text=event_text)

        if actor:
            G.add_node(actor)
            G.add_edge(event_id, actor, relation='actor')
        if recipient:
            G.add_node(recipient)
            G.add_edge(event_id, recipient, relation='recipient')
        if event_type:
            G.add_node(event_type)
            G.add_edge(event_id, event_type, relation='type')
        if place:
            G.add_node(place)
            G.add_edge(event_id, place, relation='location')

    df.apply(add_event_to_graph, axis=1)
    return G


def save_graph_pickle(G: nx.Graph, filename: str = "graph_cleaned.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(G, f)


def query_graph(G: nx.Graph, node: str):
    if node not in G:
        return f"No information found for node '{node}'"

    results = []
    for neighbor in G.neighbors(node):
        edge_data = G.get_edge_data(node, neighbor)
        relation = edge_data.get("relation", "related_to")
        timestamp = G.nodes[node].get("event_date") or G.nodes[neighbor].get("event_date")
        event_text = G.nodes[node].get("event_text") or G.nodes[neighbor].get("event_text")
        results.append((node, relation, neighbor, timestamp, event_text))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the raw POLECAT CSV file")
    parser.add_argument("--query_node", default=None, help="Optional node to query after building the graph")
    args = parser.parse_args()

    print("\n[INFO] Loading raw dataset...")
    raw_df = pd.read_csv(args.csv_path)

    print("[INFO] Cleaning dataset...")
    df = clean_polecat_df(raw_df)
    print(f"[INFO] Cleaned dataset contains {len(df)} rows.")

    print("[INFO] Splitting dataset by date...")
    train_df, test_df = split_by_date(df)
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    print(f"[INFO] Saved train ({len(train_df)} rows) and test ({len(test_df)} rows) datasets.")

    print("[INFO] Building graph from training data...")
    G = build_graph_from_dataframe(train_df)
    print(f"[INFO] Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    print("[INFO] Saving graph to 'graph_cleaned.pkl'...")
    save_graph_pickle(G)

    if args.query_node:
        print(f"\n[INFO] Querying graph for node '{args.query_node}':")
        for triple in query_graph(G, args.query_node):
            print(triple)

    print("[INFO] Done.")
