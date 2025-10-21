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
        category_lines = False
        for line in tqdm(f, desc="Parsing metadata"):
            # Keep original line for indentation checks, but use a stripped
            # version for prefix matching so we correctly handle lines like
            # '  title:' that may have varying leading spaces.
            raw = line.rstrip("\n")
            s = raw.strip()

            if not s:
                # Blank line resets category parsing but otherwise skip
                category_lines = False
                continue

            # ID and ASIN lines have no indentation in the canonical meta file
            if s.startswith("Id:"):
                if current_product and "Id" in current_product:
                    metadata[current_product["Id"]] = current_product
                current_product = {}
                # support formats like 'Id:   12345'
                try:
                    current_product["Id"] = int(s.split("Id:", 1)[1].strip())
                except Exception:
                    current_product["Id"] = s.split("Id:", 1)[1].strip()
            elif s.startswith("ASIN:"):
                current_product["ASIN"] = s.split("ASIN:", 1)[1].strip()
            # The rest of the fields are indented in the meta file; use the
            # stripped version to match the field name regardless of spacing
            elif s.startswith("title:"):
                current_product["title"] = s.split("title:", 1)[1].strip()
            elif s.startswith("group:"):
                current_product["group"] = s.split("group:", 1)[1].strip()
            elif s.startswith("salesrank:"):
                rank = s.split("salesrank:", 1)[1].strip()
                current_product["salesrank"] = int(rank) if rank.isdigit() else None
            elif s.startswith("similar:"):
                similar = s.split("similar:", 1)[1].strip()
                similar_count = int(similar.split()[0]) if similar else 0
                current_product["similar_count"] = similar_count
                if similar_count > 0:
                    # similar list follows the count
                    current_product["similar"] = similar.split()[1:]
                else:
                    current_product["similar"] = []
            elif s.startswith("categories:"):
                category_lines = True
                current_product.setdefault("categories", [])
                # parse count if present, e.g. 'categories: 2'
                try:
                    current_product["categories_count"] = int(
                        s.split("categories:", 1)[1].strip()
                    )
                except Exception:
                    current_product["categories_count"] = None
            elif category_lines and s.startswith("|"):
                current_product.setdefault("categories", []).append(s)
            elif s.startswith("reviews:"):
                category_lines = False
                # extract total and avg rating using regex for robustness
                total_match = re.search(r"total:\s*(\d+)", s)
                avg_match = re.search(r"avg rating:\s*([0-9.]+)", s)
                try:
                    current_product["reviews_total"] = (
                        int(total_match.group(1)) if total_match else 0
                    )
                except Exception:
                    current_product["reviews_total"] = 0
                try:
                    current_product["reviews_avg_rating"] = (
                        float(avg_match.group(1)) if avg_match else 0.0
                    )
                except Exception:
                    current_product["reviews_avg_rating"] = 0.0
            else:
                # Other indented lines (individual review lines, etc.) are ignored
                pass

        if current_product and "Id" in current_product:
            # Ensure the Id is stored as an int key when possible
            try:
                meta_id = int(current_product["Id"])
            except Exception:
                meta_id = current_product["Id"]
            metadata[meta_id] = current_product

    print(f"Loaded metadata for {len(metadata)} products")
    return metadata


def convert_to_pkl(
    input_file, output_file, meta_file=None, directed=True, drop_zero=False
):
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

    # Create graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Load metadata if provided
    metadata = {}
    meta_asin_to_id = {}
    if meta_file:
        metadata = load_metadata(meta_file)
        # Create ASIN-to-Id mapping for metadata (ASIN values inside meta dict)
        meta_asin_to_id = {}
        for meta_id, meta in metadata.items():
            asin_val = meta.get("ASIN")
            if asin_val:
                meta_asin_to_id[str(asin_val)] = meta_id
        print(
            f"Metadata ASINs sample: {list(meta_asin_to_id.keys())[:10]}"
        )  # Debug ASINs

    # We'll decide whether to remove node 0 later based on metadata or flag

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
                # Initialize nodes with id and placeholder asin
                if from_node not in G.nodes:
                    G.nodes[from_node]["id"] = str(from_node)
                    G.nodes[from_node]["asin"] = str(
                        from_node
                    )  # Placeholder, to be updated
                if to_node not in G.nodes:
                    G.nodes[to_node]["id"] = str(to_node)
                    G.nodes[to_node]["asin"] = str(to_node)  # Placeholder
        except ValueError:
            print(f"Skipping invalid line: {line}")
            continue

    # Enrich nodes with metadata using ASIN mapping
    # Remove node 0 if requested by flag or if metadata is clearly 1-based
    if drop_zero and 0 in G.nodes:
        print("--drop-zero option enabled: removing node 0 from the graph")
        G.remove_node(0)
    else:
        if metadata:
            try:
                if min(int(k) for k in metadata.keys()) == 1 and 0 in G.nodes:
                    print(
                        "Detected 1-based metadata IDs and node 0 in graph -> removing node 0"
                    )
                    G.remove_node(0)
            except Exception:
                pass
    for node_id in list(G.nodes):
        # Ensure we operate on the actual node attribute dict
        node_attrs = G.nodes[node_id]
        # Always set the 'id' attribute as string
        node_attrs["id"] = str(node_id)

        # Set safe defaults for attributes that DeepSNAP will tensorize
        node_attrs.setdefault("asin", str(node_id))
        node_attrs.setdefault("title", f"Product {node_id}")
        node_attrs.setdefault("group", "Unknown")
        node_attrs.setdefault("salesrank", -1)
        node_attrs.setdefault("similar_count", 0)
        node_attrs.setdefault("similar", [])
        node_attrs.setdefault("categories", [])
        node_attrs.setdefault("reviews_total", 0)
        node_attrs.setdefault("reviews_avg_rating", 0.0)

        # Prefer metadata lookup by numeric Id (the file's "Id" field)
        meta = None
        if node_id in metadata:
            meta = metadata[node_id]
        else:
            # Fallback to ASIN-based lookup if an 'asin' attribute exists
            asin = node_attrs.get("asin")
            if asin and str(asin) in meta_asin_to_id:
                meta_id = meta_asin_to_id[str(asin)]
                meta = metadata.get(meta_id)
        if meta:
            # Use metadata values and ensure label comes from group when available
            label = meta.get("group", meta.get("title", f"Product {node_id}"))
            # coerce numeric fields to safe defaults (avoid None values)
            salesrank_val = meta.get("salesrank")
            salesrank_val = int(salesrank_val) if isinstance(salesrank_val, int) else -1
            similar_count_val = (
                int(meta.get("similar_count", 0))
                if meta.get("similar_count") is not None
                else 0
            )
            reviews_total_val = (
                int(meta.get("reviews_total", 0))
                if meta.get("reviews_total") is not None
                else 0
            )
            reviews_avg_val = (
                float(meta.get("reviews_avg_rating", 0.0))
                if meta.get("reviews_avg_rating") is not None
                else 0.0
            )
            node_attrs.update(
                {
                    "asin": meta.get("ASIN", node_attrs.get("asin", str(node_id))),
                    "title": meta.get("title", f"Product {node_id}"),
                    "group": meta.get("group", "Unknown"),
                    "salesrank": salesrank_val,
                    "similar_count": similar_count_val,
                    "similar": meta.get("similar", []),
                    "categories": meta.get("categories", []),
                    "reviews_total": reviews_total_val,
                    "reviews_avg_rating": reviews_avg_val,
                    "label": label,
                }
            )
        else:
            # Ensure a fallback label exists
            node_attrs.setdefault("label", f"Product {node_id}")

    # Add edge attributes
    for u, v in G.edges():
        G.edges[u, v].update({"weight": 1.0, "type": "co_purchase"})

    # Prepare raw data dict expected by CustomGraphDataset
    raw_data = {}
    # Nodes as (node_id, attr_dict) tuples
    raw_data["nodes"] = []
    for n in G.nodes():
        # make a plain dict copy of node attributes
        raw_data["nodes"].append((n, dict(G.nodes[n])))

    # Edges as (u, v) or (u, v, attr_dict)
    raw_data["edges"] = []
    for u, v, attrs in G.edges(data=True):
        # convert attr view to plain dict
        attr_dict = dict(attrs) if attrs else {}
        raw_data["edges"].append((u, v, attr_dict))

    # Save the raw_data dict to pickle (this matches the project's expectations)
    with open(output_file, "wb") as f:
        pickle.dump(raw_data, f)

    print(f"✅ Conversion complete!")
    print(f"Raw data: {len(raw_data['nodes'])} nodes, {len(raw_data['edges'])} edges")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert edge list + metadata to a raw-data pickle compatible with the project's CustomGraphDataset (produces a dict with 'nodes' and 'edges')"
    )
    parser.add_argument("input", help="Input edge list file (e.g., amazon0302.txt)")
    parser.add_argument("-o", "--output", help="Output pickle file", default=None)
    parser.add_argument(
        "-m", "--meta", help="Metadata file (e.g., amazon-meta.txt)", default=None
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Create undirected graph (default: directed)",
    )
    parser.add_argument(
        "--drop-zero",
        action="store_true",
        help="Remove node with id 0 from the graph (useful when metadata is 1-based)",
    )

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        base_name = args.input.rsplit(".", 1)[0]
        graph_type = "undirected" if args.undirected else "directed"
        args.output = f"{base_name}_{graph_type}.pkl"

    convert_to_pkl(
        args.input,
        args.output,
        meta_file=args.meta,
        directed=not args.undirected,
        drop_zero=args.drop_zero,
    )


if __name__ == "__main__":
    main()
