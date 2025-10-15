import argparse
import pickle
import networkx as nx
from tqdm import tqdm
import re


def load_metadata(meta_file):
    """Load Amazon metadata from meta file."""
    metadata = {}
    print(f"Reading metadata from {meta_file}...")

    with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Parse metadata blocks
    blocks = re.split(r"\n\s*\n", content)

    for block in tqdm(blocks, desc="Parsing metadata"):
        if not block.strip():
            continue

        lines = block.strip().split("\n")
        if not lines:
            continue

        # Extract ASIN (node ID)
        asin = None
        for line in lines:
            if line.startswith("ASIN:"):
                asin = line.split("ASIN:", 1)[1].strip()
                break

        if not asin:
            continue

        # Extract other attributes
        attrs = {"asin": asin}

        for line in lines:
            if line.startswith("  title:"):
                attrs["title"] = line.split("title:", 1)[1].strip()
            elif line.startswith("  group:"):
                attrs["group"] = line.split("group:", 1)[1].strip()
            elif line.startswith("  salesrank:"):
                rank = line.split("salesrank:", 1)[1].strip()
                try:
                    attrs["salesrank"] = int(rank)
                except ValueError:
                    attrs["salesrank"] = None
            elif line.startswith("  similar:"):
                similar = line.split("similar:", 1)[1].strip()
                attrs["similar_count"] = len(similar.split()) if similar else 0

        # Use ASIN as key (this should match the node IDs in the edge file)
        try:
            node_id = int(asin)
            metadata[node_id] = attrs
        except ValueError:
            # If ASIN is not numeric, skip
            continue

    print(f"Loaded metadata for {len(metadata)} products")
    return metadata


def convert_to_pkl(input_file, output_file, meta_file=None, directed=True):
    """
    Convert edge list file to NetworkX pickle format with optional metadata.

    Args:
        input_file: Path to input edge list file
        output_file: Path to output pickle file
        meta_file: Path to metadata file (optional)
        directed: Whether to create directed or undirected graph
    """
    print(f"Converting {input_file} to {output_file}")
    print(f"Graph type: {'Directed' if directed else 'Undirected'}")
    if meta_file:
        print(f"Using metadata from: {meta_file}")

    # Create graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Load metadata if provided
    metadata = {}
    if meta_file:
        print("Loading metadata...")
        metadata = load_metadata(meta_file)

    # Read edges
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
        except ValueError:
            continue  # Skip invalid lines

    # Prepare data in the expected format
    # Extract nodes from edges and add attributes (with metadata if available)
    nodes_with_attrs = []
    for node_id in G.nodes():
        # Start with basic attributes
        node_attrs = {"id": str(node_id), "label": "product"}

        # Add metadata if available
        if node_id in metadata:
            meta = metadata[node_id]
            node_attrs.update(
                {
                    "asin": meta.get("asin", str(node_id)),
                    "title": meta.get("title", f"Product {node_id}"),
                    "group": meta.get("group", "Unknown"),
                    "salesrank": meta.get("salesrank"),
                    "similar_count": meta.get("similar_count", 0),
                }
            )

        nodes_with_attrs.append((node_id, node_attrs))

    # Add default attributes to edges
    edges_with_attrs = []
    for u, v in G.edges():
        edges_with_attrs.append((u, v, {"weight": 1.0, "type": "co_purchase"}))

    data_to_save = {
        "nodes": nodes_with_attrs,
        "edges": edges_with_attrs,
    }

    # Save graph in the expected format
    with open(output_file, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Conversion complete!")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert edge list to pickle")
    parser.add_argument("input", help="Input edge list file")
    parser.add_argument("-o", "--output", help="Output pickle file", default=None)
    parser.add_argument("-m", "--meta", help="Metadata file (optional)", default=None)
    parser.add_argument(
        "--undirected", action="store_true", help="Create undirected graph"
    )

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        base_name = args.input.rsplit(".", 1)[0]
        graph_type = "undirected" if args.undirected else "directed"
        args.output = f"{base_name}_{graph_type}.pkl"

    convert_to_pkl(
        args.input, args.output, meta_file=args.meta, directed=not args.undirected
    )


if __name__ == "__main__":
    main()
