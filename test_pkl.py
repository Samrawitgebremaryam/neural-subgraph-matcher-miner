import pickle
import networkx as nx
from pprint import pprint

def test_pkl_file(pkl_file):
    """
    Test the structure and content of a NetworkX pickle file.

    Args:
        pkl_file (str): Path to the pickle file (e.g., 'amazon0302.pkl')
    """
    print(f"Testing pickle file: {pkl_file}")

    # Load the pickle file
    try:
        with open(pkl_file, "rb") as f:
            G = pickle.load(f)
        print(f"Successfully loaded graph from {pkl_file}")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # Verify graph type
    is_directed = isinstance(G, nx.DiGraph)
    print(f"Graph is {'directed' if is_directed else 'undirected'}")

    # Check basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

    # Inspect a sample of nodes
    print("\nSampling 5 nodes and their attributes:")
    node_samples = list(G.nodes(data=True))[:5]
    for node, attrs in node_samples:
        print(f"Node {node}:")
        pprint(attrs)
        expected_label = attrs.get("title", f"Product {node}")
        actual_label = attrs.get("label", "Missing")
        print(f"  Label check: Expected '{expected_label}', Got '{actual_label}'")
        if actual_label != expected_label:
            print(f"  Warning: Label mismatch for node {node}")

    # Inspect a sample of edges
    print("\nSampling 5 edges and their attributes:")
    edge_samples = list(G.edges(data=True))[:5]
    for u, v, attrs in edge_samples:
        print(f"Edge ({u}, {v}):")
        pprint(attrs)

    # Additional validation
    if num_nodes > 0 and num_edges > 0:
        print("\nValidation passed: Graph contains nodes and edges.")
    else:
        print("\nWarning: Graph is empty or malformed.")

    # Check for consistency in node attributes
    all_have_id = all("id" in attrs for _, attrs in G.nodes(data=True))
    all_have_label = all("label" in attrs for _, attrs in G.nodes(data=True))
    print(f"All nodes have 'id' attribute: {all_have_id}")
    print(f"All nodes have 'label' attribute: {all_have_label}")

    print("\nTest complete!")

def main():
    # Specify the path to your pickle file
    pkl_file = "amazon0302.pkl"
    test_pkl_file(pkl_file)

if __name__ == "__main__":
    main()