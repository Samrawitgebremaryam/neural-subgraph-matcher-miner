#!/usr/bin/env python3
"""
Read the bio graph from CSV/TXT and save as .pkl for SPMiner.

Expects:
- Nodes CSV: node_id, label_id, type_name (header row optional)
- Edges TXT: lines "source\ttarget" or edges CSV: source, target
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import networkx as nx


def load_graph_from_csv_txt(nodes_path, edges_path, directed=True):
    """Build nx.DiGraph from nodes CSV and edges TXT/CSV."""
    G = nx.DiGraph() if directed else nx.Graph()

    # Load nodes: CSV with optional header (node_id, label_id, type_name)
    with open(nodes_path, "r", newline="") as f:
        reader = csv.reader(f)
        row = next(reader, None)
        if row and row[0].strip().lower() in ("node_id", "id", "node"):
            row = next(reader, None)  # skip header
        while row:
            if not row or not row[0].strip():
                row = next(reader, None)
                continue
            try:
                nid = int(row[0])
            except ValueError:
                row = next(reader, None)
                continue
            label_id = int(row[1]) if len(row) > 1 else 0
            type_name = (row[2].strip() if len(row) > 2 else str(nid)) or str(nid)
            G.add_node(nid, label=type_name, label_id=label_id, type_name=type_name)
            row = next(reader, None)

    # Load edges (TXT: "source\ttarget" or CSV: source,target[,type])
    edges_path = Path(edges_path)
    with open(edges_path, "r") as f:
        if edges_path.suffix.lower() == ".csv":
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    u, v = int(row[0]), int(row[1])
                    attrs = {"type": row[2].strip()} if len(row) >= 3 and row[2].strip() else {}
                    G.add_edge(u, v, **attrs)
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", "\t").split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    attrs = {"type": parts[2].strip()} if len(parts) >= 3 and parts[2].strip() else {}
                    G.add_edge(u, v, **attrs)

    return G


def main():
    parser = argparse.ArgumentParser(description="Convert bio graph CSV/TXT to PKL for SPMiner.")
    parser.add_argument("--nodes", default="data/bio_nodes.csv", help="Nodes CSV path")
    parser.add_argument("--edges", default="data/bio_edges.txt", help="Edges TXT or CSV path")
    parser.add_argument("--output", "-o", default="data/bio_data.pkl", help="Output PKL path")
    parser.add_argument("--undirected", action="store_true", help="Build undirected graph")
    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    if not nodes_path.exists():
        print(f"Error: nodes file not found: {nodes_path}", file=sys.stderr)
        return 1
    if not edges_path.exists():
        print(f"Error: edges file not found: {edges_path}", file=sys.stderr)
        return 1

    G = load_graph_from_csv_txt(str(nodes_path), str(edges_path), directed=not args.undirected)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
