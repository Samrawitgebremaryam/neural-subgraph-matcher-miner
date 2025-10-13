import argparse
import pickle
import networkx as nx
from tqdm import tqdm


def convert_to_pkl(input_file, output_file, directed=True):
    """
    Convert edge list file to NetworkX pickle format.

    Args:
        input_file: Path to input edge list file
        output_file: Path to output pickle file
        directed: Whether to create directed or undirected graph
    """
    print(f"Converting {input_file} to {output_file}")
    print(f"Graph type: {'Directed' if directed else 'Undirected'}")

    # Create graph
    G = nx.DiGraph() if directed else nx.Graph()

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

    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f)

    print(f"Conversion complete!")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert edge list to pickle")
    parser.add_argument("input", help="Input edge list file")
    parser.add_argument("-o", "--output", help="Output pickle file", default=None)
    parser.add_argument(
        "--undirected", action="store_true", help="Create undirected graph"
    )

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        base_name = args.input.rsplit(".", 1)[0]
        graph_type = "undirected" if args.undirected else "directed"
        args.output = f"{base_name}_{graph_type}.pkl"

    convert_to_pkl(args.input, args.output, directed=not args.undirected)


if __name__ == "__main__":
    main()
