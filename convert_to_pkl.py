import argparse
import pickle
import networkx as nx
from tqdm import tqdm
import re

def load_metadata(meta_file):
    """Load Amazon metadata from meta file, mapping ASINs to attributes."""
    metadata = {}
    print(f"Reading metadata from {meta_file}...")

    with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
        current_product = {}
        for line in tqdm(f, desc="Parsing metadata"):
            line = line.strip()
            if not line or line.startswith("#"):
                if current_product and "ASIN" in current_product:
                    try:
                        # Use ASIN as key, but convert to int if numeric for ID compatibility
                        asin = current_product["ASIN"]
                        node_id = int(asin) if asin.isdigit() else None
                        if node_id is not None and node_id < 548552:  # Limit to meta dataset size
                            metadata[node_id] = current_product
                    except ValueError:
                        continue  # Skip non-numeric ASINs
                    current_product = {}
                continue
            if line.startswith("Id:"):
                id_val = int(line.split("Id:")[1].strip())
                current_product["Id"] = id_val
            elif line.startswith("ASIN:"):
                current_product["ASIN"] = line.split("ASIN:")[1].strip()
            elif line.startswith("  title:"):
                current_product["title"] = line.split("title:")[1].strip()
            elif line.startswith("  group:"):
                current_product["group"] = line.split("group:")[1].strip()
            elif line.startswith("  salesrank:"):
                rank = line.split("salesrank:")[1].strip()
                current_product["salesrank"] = int(rank) if rank.isdigit() else None
            elif line.startswith("  similar:"):
                similar = line.split("similar:")[1].strip()
                current_product["similar_count"] = len(similar.split()) if similar else 0
            elif line.startswith("  categories:"):
                categories = []
                for cat_line in line.split("\n"):
                    if "|" in cat_line:
                        categories.append(cat_line.strip().split("|")[1:-1])  # Extract category path
                current_product["categories"] = categories
            elif line.startswith("  reviews:"):
                review_info = line.split("reviews:")[1].strip().split()
                current_product["reviews_total"] = int(review_info[1])
                current_product["reviews_avg_rating"] = float(review_info[5]) if review_info[5] else 0.0

    print(f"Loaded metadata for {len(metadata)} products")
    return metadata

def convert_to_pkl(input_file, output_file, meta_file=None, directed=True):
    """
    Convert edge list file to NetworkX pickle format with optional metadata.

    Args:
        input_file: Path to input edge list file (e.g., amazon0302.txt)
        output_file: Path to output pickle file (e.g., amazon0302.pkl)
        meta_file: Path to metadata file (e.g., amazon-meta.txt, optional)
        directed: Whether to create a directed graph (True for Amazon0302)
    """
    print(f"Converting {input_file} to {output_file}")
    print(f"Graph type: {'Directed' if directed else 'Undirected'}")
    if meta_file:
        print(f"Using metadata from: {meta_file}")

    # Create directed graph
    G = nx.DiGraph()

    # Load metadata if provided
    metadata = {}
    if meta_file:
        metadata = load_metadata(meta_file)

    # Read edges and build graph
    with open(input_file, "r") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} lines...")

    for line in tqdm(lines, desc="Processing edges"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            parts = line.split()
            if len(parts) >= 2:
                from_node = int(parts[0])
                to_node = int(parts[1])
                G.add_edge(from_node, to_node)
                # Add default attributes
                if from_node not in G.nodes:
                    G.nodes[from_node]["id"] = str(from_node)
                    G.nodes[from_node]["label"] = "product"
                if to_node not in G.nodes:
                    G.nodes[to_node]["id"] = str(to_node)
                    G.nodes[to_node]["label"] = "product"
        except ValueError:
            print(f"Skipping invalid line: {line}")
            continue

    # Enrich nodes with metadata where available
    for node_id in G.nodes:
        if node_id in metadata:
            meta = metadata[node_id]
            G.nodes[node_id].update({
                "asin": meta.get("ASIN", str(node_id)),
                "title": meta.get("title", f"Product {node_id}"),
                "group": meta.get("group", "Unknown"),
                "salesrank": meta.get("salesrank"),
                "similar_count": meta.get("similar_count", 0),
                "categories": meta.get("categories", []),
                "reviews_total": meta.get("reviews_total", 0),
                "reviews_avg_rating": meta.get("reviews_avg_rating", 0.0)
            })

    # Add edge attributes
    for u, v in G.edges():
        G.edges[u, v].update({"weight": 1.0, "type": "co_purchase"})

    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f)

    print(f"✅ Conversion complete!")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert edge list to NetworkX pickle for SPMiner")
    parser.add_argument("input", help="Input edge list file (e.g., amazon0302.txt)")
    parser.add_argument("-o", "--output", help="Output pickle file", default=None)
    parser.add_argument("-m", "--meta", help="Metadata file (e.g., amazon-meta.txt)", default=None)
    parser.add_argument(
        "--undirected", action="store_true", help="Create undirected graph (default: directed)"
    )

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        base_name = args.input.rsplit(".", 1)[0]
        graph_type = "undirected" if args.undirected else "directed"
        args.output = f"{base_name}_{graph_type}.pkl"

    convert_to_pkl(args.input, args.output, meta_file=args.meta, directed=not args.undirected)

if __name__ == "__main__":
    main()