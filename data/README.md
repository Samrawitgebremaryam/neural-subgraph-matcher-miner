# Bio graph data (CSV/TXT and PKL)

## 1. Generate the graph (CSV + TXT + optional PKL)

```bash
# From repo root: create data/ with bio_nodes.csv, bio_edges.txt, bio_edges.csv, and bio_data.pkl
python scripts/generate_balanced_bio_graph.py

# Only CSV/TXT (no pkl); convert to pkl later with csv_to_pkl.py
python scripts/generate_balanced_bio_graph.py --no-pkl
```

**Output files (default `data/`):**

| File             | Format | Description                          |
|------------------|--------|--------------------------------------|
| `bio_nodes.csv`  | CSV    | node_id, label_id, type_name         |
| `bio_edges.txt`  | TXT    | One line per edge: `source\ttarget`  |
| `bio_edges.csv`  | CSV    | source, target                       |
| `bio_data.pkl`   | PKL    | NetworkX DiGraph for SPMiner          |

## 2. Convert CSV/TXT to PKL (if you only have CSV/TXT)

```bash
python scripts/csv_to_pkl.py --nodes data/bio_nodes.csv --edges data/bio_edges.txt -o data/bio_data.pkl
```

## 3. Run SPMiner (e.g. in GitHub Actions)

Use the generated `data/bio_data.pkl` as the dataset:

```bash
python -m subgraph_mining.decoder --dataset data/bio_data.pkl --graph_type directed ...
```
