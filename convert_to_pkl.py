import argparse
import pickle
import networkx as nx
from tqdm import tqdm


def convert_protein_links(input_file, output_file):
    """
    Convert STRING protein links dataset (9606.protein.links.v12.0.txt)
    into the raw-data pickle format used by CustomGraphDataset.
    """

    print(f"Loading protein interaction data from: {input_file}")

    G = nx.Graph()  # STRING links are undirected

    with open(input_file, "r") as f:
        # Skip header line
        header = f.readline()

        for line in tqdm(f, desc="Processing edges"):
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            protein1, protein2, score = parts
            score = float(score)

            # Add edge
            G.add_edge(protein1, protein2, weight=score, type="protein_link")

            # Add node attributes (minimal)
            if protein1 not in G.nodes:
                G.nodes[protein1]["id"] = protein1
                G.nodes[protein1]["label"] = protein1

            if protein2 not in G.nodes:
                G.nodes[protein2]["id"] = protein2
                G.nodes[protein2]["label"] = protein2

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Prepare output dict
    raw_data = {
        "nodes": [],
        "edges": []
    }

    # Add nodes with attributes
    for n, attrs in G.nodes(data=True):
        raw_data["nodes"].append((n, dict(attrs)))

    # Add edges with attributes
    for u, v, attrs in G.edges(data=True):
        raw_data["edges"].append((u, v, dict(attrs)))

    # Save pickle
    with open(output_file, "wb") as f:
        pickle.dump(raw_data, f)

    print(f"✅ Saved PKL to: {output_file}")
    print(f"Nodes: {len(raw_data['nodes'])}, Edges: {len(raw_data['edges'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert STRING protein links dataset to PKL")
    parser.add_argument("input", help="Input file: 9606.protein.links.v12.0.txt")
    parser.add_argument("-o", "--output", default="protein_links.pkl", help="Output PKL file name")

    args = parser.parse_args()
    convert_protein_links(args.input, args.output)


if __name__ == "__main__":
    main()
