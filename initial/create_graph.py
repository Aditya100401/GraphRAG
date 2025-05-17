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

        etype     = row.get('Event Type', '')
        place     = row.get('Raw Placename', '')
        intensity = row.get('Event Intensity')
        quad      = row.get('Quad Code')
        contexts  = row.get('Contexts')
        actor     = row.get('Actor Name')     or row.get('Actor Country')    or ''
        recipient = row.get('Recipient Name') or row.get('Recipient Country') or ''
        atitle    = row.get('Actor Title', '')
        rtitle    = row.get('Recipient Title', '')

        contexts_str = ";".join([c.strip() for c in str(contexts).split(';') if c.strip()]) if pd.notna(contexts) else ""

        G.add_node(eid,
            node_type     = "event",
            event_date    = date_str,
            event_ts      = float(ts) if ts is not None else None,
            event_type    = str(etype),
            place         = str(place),
            intensity     = float(intensity) if pd.notna(intensity) else None,
            quad_code     = str(quad) if pd.notna(quad) else "",
            contexts      = contexts_str
        )

        def add_entity(name, role, extra_title=""):
            name = str(name).strip()
            if name and name.lower() not in ("none", "nan"):
                G.add_node(name, node_type=role, title=str(extra_title))
                G.add_edge(eid, name, relation=role)

        add_entity(actor,    "actor",     atitle)
        add_entity(recipient,"recipient", rtitle)
        add_entity(etype, "eventType")
        add_entity(place,"location")

        if actor:
            actor_events.setdefault(actor, []).append((date, eid))

    for actor, evts in actor_events.items():
        timeline = sorted((d,e) for d,e in evts if pd.notna(d))
        for (_, e1), (_, e2) in zip(timeline, timeline[1:]):
            G.add_edge(e1, e2, relation="next_event", actor=str(actor))

    return G

def save_graph(G: nx.DiGraph, out_base: str):
    with open(out_base + ".pkl", "wb") as f:
        pickle.dump(G, f)
    nx.write_graphml(G, out_base + ".graphml")
    print(f"[INFO] Saved {out_base}.pkl and {out_base}.graphml")

if __name__ == "__main__":
    INPUT_DIR  = "datasets/country_sets"               
    OUTPUT_DIR = "datasets/country_specific_graphs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for csv_path in glob.glob(os.path.join(INPUT_DIR, "combined_data_*.csv")):
        df       = pd.read_csv(csv_path, low_memory=False)
        basename = os.path.basename(csv_path).replace(".csv","")
        print(f"[INFO] Building KG for {basename}...")

        G = build_temporal_kg(df)
        print(f"  → Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        out_base = os.path.join(OUTPUT_DIR, basename.replace("combined_data_", "graph_combined_"))
        save_graph(G, out_base)

    print("[DONE] All combined graphs built.")
