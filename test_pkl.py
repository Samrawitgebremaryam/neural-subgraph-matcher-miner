import argparse
import pickle
import sys
from pprint import pprint

try:
    import networkx as nx
except Exception:
    nx = None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_raw_data(obj):
    """Return (nodes, edges) normalized to expected shapes.

    nodes: list of (node_id, attr_dict)
    edges: list of (u, v) or (u, v, attr_dict)
    """
    # Preferred format
    if isinstance(obj, dict) and "nodes" in obj and "edges" in obj:
        return obj["nodes"], obj["edges"]

    # Support legacy: NetworkX Graph or DiGraph
    if nx is not None and isinstance(obj, (nx.Graph, nx.DiGraph)):
        G = obj
        nodes = [(n, dict(G.nodes[n])) for n in G.nodes]
        edges = []
        for u, v, attrs in G.edges(data=True):
            edges.append((u, v, dict(attrs) if attrs else {}))
        return nodes, edges

    raise ValueError(
        'Unsupported pickle format: expected dict with "nodes"/"edges" or a NetworkX Graph'
    )


def validate(pkl_path, expect_drop_zero=False, require_metadata=False, sample=5):
    print(f"Loading pickle: {pkl_path}")
    try:
        obj = load_pickle(pkl_path)
    except Exception as e:
        print("FATAL: failed to load pickle:", e)
        return 2

    try:
        nodes, edges = normalize_raw_data(obj)
    except Exception as e:
        print("FATAL:", e)
        return 2

    print(f"Loaded raw data: {len(nodes)} nodes, {len(edges)} edges")

    # Build node id set and check uniqueness
    node_ids = [n for n, _ in nodes]
    if len(set(node_ids)) != len(node_ids):
        print("FATAL: duplicate node ids found")
        return 2
    node_set = set(node_ids)

    # Check expect_drop_zero
    if expect_drop_zero:
        if 0 in node_set or "0" in node_set:
            print("FATAL: node 0 present but --expect-drop-zero was given")
            return 2
        else:
            print("OK: node 0 not present")

    # Validate edges point to existing nodes
    bad = 0
    for e in edges:
        if len(e) < 2:
            print("FATAL: invalid edge tuple length:", e)
            return 2
        u, v = e[0], e[1]
        if (u not in node_set) and (str(u) not in node_set):
            print("FATAL: edge references unknown node u=", u)
            bad += 1
        if (v not in node_set) and (str(v) not in node_set):
            print("FATAL: edge references unknown node v=", v)
            bad += 1
    if bad:
        return 2

    # Node attribute checks
    missing_id = 0
    missing_label = 0
    titles = 0
    mismatch = 0
    for nid, attrs in nodes:
        if not isinstance(attrs, dict):
            print(f"FATAL: node {nid} attrs not dict: {type(attrs)}")
            return 2
        if "id" not in attrs:
            missing_id += 1
        else:
            if str(attrs["id"]) != str(nid):
                print(f'FATAL: node {nid} has id attr {attrs["id"]} mismatch')
                return 2
        if "label" not in attrs:
            missing_label += 1
        if "title" in attrs:
            titles += 1
            if attrs.get("label") != attrs.get("title"):
                mismatch += 1

    if missing_id:
        print(f'FATAL: {missing_id} nodes missing "id" attribute')
        return 2
    if missing_label:
        print(f'WARNING: {missing_label} nodes missing "label" attribute')
    if require_metadata and titles == 0:
        print('FATAL: --require-metadata was set but no node has a "title" field')
        return 2
    if mismatch:
        print(f"WARNING: {mismatch} nodes had a title but label != title")

    # Print samples
    print("\nSample nodes:")
    for nid, attrs in nodes[:sample]:
        print("Node", nid)
        pprint(attrs)
        expected = attrs.get("title", f"Product {nid}")
        print("  label check -> expected:", expected, "got:", attrs.get("label"))

    print("\nSample edges:")
    for e in edges[:sample]:
        pprint(e)

    print("\nValidation completed.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset pickle produced by convert_to_pkl.py"
    )
    parser.add_argument("pkl", help="Path to pickle file")
    parser.add_argument(
        "--expect-drop-zero", action="store_true", help="Fail if node 0 is present"
    )
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="Fail if no node contains title metadata",
    )
    parser.add_argument(
        "--sample", type=int, default=5, help="Sample size for printing nodes/edges"
    )
    args = parser.parse_args()

    rc = validate(
        args.pkl,
        expect_drop_zero=args.expect_drop_zero,
        require_metadata=args.require_metadata,
        sample=args.sample,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
