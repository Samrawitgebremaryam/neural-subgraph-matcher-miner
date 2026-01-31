"""
Recreate the bio graph and data/bio_data.pkl with edge labels (type).
Edge labels: regulates, transcribes, translates, catalyzes (TF→Gene→mRNA→Enzyme→Metabolite).
Writes: data/bio_nodes.csv, data/bio_edges.csv, data/bio_edges.txt, data/bio_data.pkl.
No networkx required.
"""
import csv
import pickle
from pathlib import Path

DATA = Path("data")
NODE_TYPES = ["TF", "Gene", "mRNA", "Enzyme", "Metabolite"]
EDGE_LABELS = ["regulates", "transcribes", "translates", "catalyzes"]  # TF→Gene, Gene→mRNA, etc.

type_map = {t: list(range(t * 500, (t + 1) * 500)) for t in range(5)}

# Motif edges with type (label)
motif_edges = []
for i in range(100):
    tf, gene = type_map[0][i], type_map[1][i]
    mrna, enzyme, metabolite = type_map[2][i], type_map[3][i], type_map[4][i]
    motif_edges.append((tf, gene, EDGE_LABELS[0]))
    motif_edges.append((gene, mrna, EDGE_LABELS[1]))
    motif_edges.append((mrna, enzyme, EDGE_LABELS[2]))
    motif_edges.append((enzyme, metabolite, EDGE_LABELS[3]))

# Noise edges with type (same deterministic set as before)
noise_edges = []
for i in range(1000):
    t1, t2 = i % 4, i % 4 + 1
    u = type_map[t1][(i * 31) % 500]
    v = type_map[t2][(i * 17) % 500]
    noise_edges.append((u, v, EDGE_LABELS[t1]))

all_edges = motif_edges + noise_edges

DATA.mkdir(parents=True, exist_ok=True)

# 1. bio_nodes.csv: node_id, label_id, type_name
with open(DATA / "bio_nodes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["node_id", "label_id", "type_name"])
    for n in range(2500):
        label_id = n // 500
        w.writerow([n, label_id, NODE_TYPES[label_id]])

# 2. bio_edges.csv: source, target, type (edge label)
with open(DATA / "bio_edges.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target", "type"])
    for u, v, etype in all_edges:
        w.writerow([u, v, etype])

# 3. bio_edges.txt: source\ttarget\ttype (edge label)
with open(DATA / "bio_edges.txt", "w") as f:
    for u, v, etype in all_edges:
        f.write(f"{u}\t{v}\t{etype}\n")

# 4. bio_data.pkl: dict with nodes and edges (edges have "type" for decoder/visualizer)
nodes = [
    (n, {"label": NODE_TYPES[n // 500], "label_id": n // 500, "type_name": NODE_TYPES[n // 500]})
    for n in range(2500)
]
edges_for_pkl = [(u, v, {"type": etype}) for u, v, etype in all_edges]
graph_dict = {"nodes": nodes, "edges": edges_for_pkl}

with open(DATA / "bio_data.pkl", "wb") as f:
    pickle.dump(graph_dict, f)

unique = len(set((u, v) for u, v, _ in all_edges))
print(f"Recreated graph with edge labels:")
print(f"  {DATA}/bio_nodes.csv  (2500 nodes)")
print(f"  {DATA}/bio_edges.csv  ({len(all_edges)} edges, type column)")
print(f"  {DATA}/bio_edges.txt  ({len(all_edges)} edges, type column)")
print(f"  {DATA}/bio_data.pkl   (2500 nodes, {unique} unique edges, edge type=regulates|transcribes|translates|catalyzes)")
