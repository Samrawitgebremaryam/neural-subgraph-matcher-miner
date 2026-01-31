#!/usr/bin/env python3
"""
Generate the balanced biological graph and write to CSV/TXT.
Then convert to .pkl for SPMiner.

- Nodes: 5 types (TF, Gene, mRNA, Enzyme, Metabolite), 500 each = 2,500 nodes.
- Golden motifs: 100 copies of the 5-node chain TF→Gene→mRNA→Enzyme→Metabolite.
- Balanced noise: random edges that follow biological order (t -> t+1 only).
"""

import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import networkx as nx

NODE_TYPES = ["TF", "Gene", "mRNA", "Enzyme", "Metabolite"]


def generate_balanced_bio_graph(num_motifs=100, nodes_per_type=500, num_noise_edges=1000, seed=42):
    """Build directed graph with balanced labels and planted golden motifs."""
    random.seed(seed)
    G = nx.DiGraph()

    # 1. Create nodes with balanced labels
    node_id = 0
    type_map = {t: [] for t in range(5)}

    for label_id in range(5):
        for _ in range(nodes_per_type):
            G.add_node(node_id, label=label_id, type_name=NODE_TYPES[label_id])
            type_map[label_id].append(node_id)
            node_id += 1

    # Edge types for biological flow (for visualization / checking)
    EDGE_TYPES = ["regulates", "transcribes", "translates", "catalyzes"]  # TF→Gene, Gene→mRNA, mRNA→Enzyme, Enzyme→Metabolite

    # 2. Plant the golden motifs (5-node chain per motif) with edge types
    for i in range(num_motifs):
        tf = type_map[0][i]
        gene = type_map[1][i]
        mrna = type_map[2][i]
        enzyme = type_map[3][i]
        metabolite = type_map[4][i]
        G.add_edge(tf, gene, type=EDGE_TYPES[0])
        G.add_edge(gene, mrna, type=EDGE_TYPES[1])
        G.add_edge(mrna, enzyme, type=EDGE_TYPES[2])
        G.add_edge(enzyme, metabolite, type=EDGE_TYPES[3])

    # 3. Balanced noise: edges that follow biological order (t -> t+1) with same edge types
    for _ in range(num_noise_edges):
        t1 = random.randint(0, 3)
        t2 = t1 + 1
        u = random.choice(type_map[t1])
        v = random.choice(type_map[t2])
        G.add_edge(u, v, type=EDGE_TYPES[t1])

    return G


def write_graph_csv_txt(G, nodes_path, edges_txt_path, edges_csv_path=None):
    """Write nodes and edges to CSV and plain TXT (edgelist)."""
    parent = Path(nodes_path).parent
    parent.mkdir(parents=True, exist_ok=True)

    # Nodes CSV: node_id, label_id, type_name
    with open(nodes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "label_id", "type_name"])
        for n in G.nodes():
            data = G.nodes[n]
            w.writerow([n, data.get("label", ""), data.get("type_name", "")])

    # Edges TXT (simple edgelist: one "source\ttarget" per line)
    with open(edges_txt_path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")

    # Edges CSV (optional): source, target, type
    if edges_csv_path is None:
        edges_csv_path = parent / "bio_edges.csv"
    with open(edges_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "type"])
        for u, v in G.edges():
            etype = G.edges[u, v].get("type", "")
            w.writerow([u, v, etype])

    return nodes_path, edges_txt_path, edges_csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate balanced bio graph and write CSV/TXT + optional PKL.")
    parser.add_argument("--num_motifs", type=int, default=100, help="Number of golden 5-node motifs")
    parser.add_argument("--nodes_per_type", type=int, default=500, help="Nodes per type (5 types)")
    parser.add_argument("--noise_edges", type=int, default=1000, help="Extra random edges (biological order)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data", help="Output directory")
    parser.add_argument("--nodes", default="bio_nodes.csv", help="Nodes filename (under out_dir)")
    parser.add_argument("--edges", default="bio_edges.txt", help="Edges filename (under out_dir)")
    parser.add_argument("--pkl", default="bio_data.pkl", help="PKL filename (under out_dir)")
    parser.add_argument("--no-pkl", action="store_true", help="Only write CSV/TXT; do not write pkl (use csv_to_pkl.py later)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    nodes_path = out_dir / args.nodes
    edges_txt_path = out_dir / args.edges
    edges_csv_path = out_dir / "bio_edges.csv"
    pkl_path = out_dir / args.pkl

    G = generate_balanced_bio_graph(
        num_motifs=args.num_motifs,
        nodes_per_type=args.nodes_per_type,
        num_noise_edges=args.noise_edges,
        seed=args.seed,
    )

    # Write CSV and TXT
    write_graph_csv_txt(G, str(nodes_path), str(edges_txt_path), str(edges_csv_path))
    print(f"Nodes: {nodes_path} ({G.number_of_nodes()} nodes)")
    print(f"Edges CSV: {edges_csv_path}")
    print(f"Edges TXT: {edges_txt_path} ({G.number_of_edges()} edges)")

    # Optionally write pkl (same graph, with 'label' as string for SPMiner/visualizer)
    if not args.no_pkl:
        for n in G.nodes():
            G.nodes[n]["label"] = G.nodes[n]["type_name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(G, f)
        print(f"PKL: {pkl_path}")
    else:
        print("Skipping PKL (use scripts/csv_to_pkl.py to convert CSV/TXT to pkl).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
