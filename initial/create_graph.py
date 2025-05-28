import os
import glob
import pickle
import pandas as pd
import networkx as nx

def build_temporal_kg(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
    actor_events = {}

    for _, row in df.iterrows():
        eid   = str(row['Event ID'])
        date  = row['Event Date']
        date_str = date.strftime('%Y-%m-%d') if pd.notna(date) else ''
        ts       = date.timestamp()          if pd.notna(date) else None

        # event properties
        G.add_node(eid,
                    node_type="event",
                    event_date=date_str,
                    event_ts=float(ts) if ts is not None else None,
                    event_type=str(row.get('Event Type', '')),
                    place=str(row.get('Raw Placename', '')),
                    quad_code=str(row['Quad Code']) if pd.notna(row.get('Quad Code')) else "",
                    contexts=";".join([c.strip() for c in str(row.get('Contexts','')).split(';') if c.strip()])
                    )

        # helper to add actor/recipient/eventType/location nodes
        def add_entity(name, role, extra_title=""):
            n = str(name).strip()
            if n and n.lower() not in ("none","nan"):
                G.add_node(n, node_type=role, title=str(extra_title))
                G.add_edge(eid, n, relation=role)

        add_entity(row.get('Actor Name')    or row.get('Actor Country'),    "actor",     row.get('Actor Title', ''))
        add_entity(row.get('Recipient Name') or row.get('Recipient Country'), "recipient", row.get('Recipient Title', ''))
        add_entity(row.get('Event Type'), "eventType")
        add_entity(row.get('Raw Placename'), "location")

        actor = row.get('Actor Name') or row.get('Actor Country')
        if actor and pd.notna(row['Event Date']):
            actor_events.setdefault(actor, []).append((row['Event Date'], eid))

    # link each actor’s events in temporal order
    for actor, evts in actor_events.items():
        timeline = sorted(evts, key=lambda x: x[0])
        for (_, e1), (_, e2) in zip(timeline, timeline[1:]):
            G.add_edge(e1, e2, relation="next_event", actor=str(actor))

    return G

def save_graph(G: nx.DiGraph, out_base: str):
    with open(out_base + ".pkl", "wb") as f:
        pickle.dump(G, f)
    nx.write_graphml(G, out_base + ".graphml")
    print(f"[INFO] Saved {out_base}.pkl and {out_base}.graphml")

if __name__ == "__main__":
    INPUT_DIR  = "./data/final_splits"
    OUTPUT_DIR = "./data/graphs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # process train data
    for csv_path in glob.glob(os.path.join(INPUT_DIR, "train*.csv")):
        df        = pd.read_csv(csv_path, low_memory=False)
        basename  = os.path.splitext(os.path.basename(csv_path))[0]
        country   = basename.replace("train", "")  # Extract country code after 'train'

        print(f"[INFO] Building KG for {country} (train)…")
        G = build_temporal_kg(df)
        print(f"  → Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        out_base = os.path.join(OUTPUT_DIR, f"graph_{country}_train")
        save_graph(G, out_base)

    print("[DONE] All train graphs built.")