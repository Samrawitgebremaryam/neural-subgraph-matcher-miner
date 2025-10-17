import argparse
import pickle
import sys
from typing import Any, Dict, List, Tuple

import torch

try:
    import networkx as nx
except Exception:
    nx = None

NUMERIC_KEYS = {"salesrank", "similar_count", "reviews_total", "reviews_avg_rating"}


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_raw_data(
    obj: Any,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple]]:
    """Return (nodes, edges) normalized to expected shapes.

    nodes: list of (node_id, attr_dict)
    edges: list of (u, v) or (u, v, attr_dict)
    """
    if isinstance(obj, dict) and "nodes" in obj and "edges" in obj:
        return obj["nodes"], obj["edges"]
    if nx is not None and isinstance(obj, (nx.Graph, nx.DiGraph)):
        G = obj
        nodes = [(n, dict(G.nodes[n])) for n in G.nodes]
        edges = []
        for u, v, attrs in G.edges(data=True):
            edges.append((u, v, dict(attrs) if attrs else {}))
        return nodes, edges
    raise ValueError(
        'Unsupported format: expected dict with "nodes"/"edges" or a NetworkX Graph'
    )


def check_nodes(
    nodes: List[Tuple[int, Dict[str, Any]]], sample: int, limit_errors: int
) -> int:
    print(f"🔹 Total nodes: {len(nodes)}")
    errors = 0
    warnings = 0

    none_attrs: List[Tuple[int, str]] = []
    bad_numeric: List[Tuple[int, str, Any]] = []
    tensor_fail: List[Tuple[int, str, Any, str]] = []
    missing_id = 0
    id_mismatch: List[Tuple[int, Any]] = []
    missing_label = 0

    for node_id, attrs in nodes:
        # id/label presence and id consistency
        if "id" not in attrs:
            missing_id += 1
        else:
            if str(attrs["id"]) != str(node_id):
                id_mismatch.append((node_id, attrs.get("id")))
        if "label" not in attrs:
            missing_label += 1

        for k, v in attrs.items():
            if v is None:
                none_attrs.append((node_id, k))
                continue

            # Only enforce numeric checks on known numeric keys or numeric types
            if k in NUMERIC_KEYS or isinstance(v, (int, float)):
                if not isinstance(v, (int, float)):
                    bad_numeric.append((node_id, k, v))
                else:
                    # Try tensor conversion for numeric scalars
                    try:
                        _ = torch.tensor([v])
                    except Exception as e:
                        tensor_fail.append((node_id, k, v, str(e)))

    if missing_id:
        errors += 1
        print(f"❌ {missing_id} nodes missing 'id' attribute")
    if id_mismatch:
        errors += 1
        print(f"❌ {len(id_mismatch)} nodes have id attribute mismatching tuple id")
        for nid, vid in id_mismatch[: min(limit_errors, len(id_mismatch))]:
            print(f"   Node {nid} has attrs.id={vid}")
    if missing_label:
        warnings += 1
        print(f"⚠️  {missing_label} nodes missing 'label' attribute")

    if none_attrs:
        errors += 1
        print(f"❌ Found {len(none_attrs)} None attributes on nodes")
        for nid, k in none_attrs[: min(limit_errors, len(none_attrs))]:
            print(f"   Node {nid} -> {k}=None")
    else:
        print("✅ No None attributes on nodes")

    if bad_numeric:
        errors += 1
        print(f"❌ {len(bad_numeric)} numeric-like attributes are not int/float")
        for nid, k, v in bad_numeric[: min(limit_errors, len(bad_numeric))]:
            print(f"   Node {nid} -> {k} has non-numeric type {type(v)}: {v}")
    if tensor_fail:
        errors += 1
        print(f"❌ {len(tensor_fail)} numeric attributes failed tensor conversion")
        for nid, k, v, e in tensor_fail[: min(limit_errors, len(tensor_fail))]:
            print(f"   Node {nid} -> {k}={v} failed: {e}")

    # Sample printout
    if sample > 0:
        print("\nSample nodes:")
        for nid, attrs in nodes[:sample]:
            title = attrs.get("title")
            lbl = attrs.get("label")
            print(f"Node {nid}: id={attrs.get('id')} label={lbl} title={title}")

    # Return status: 0 ok, 1 warnings only, 2 errors
    if errors:
        return 2
    return 1 if warnings and not errors else 0


def check_edges(edges: List[Tuple], limit_errors: int) -> int:
    print(f"\n🔹 Total edges: {len(edges)}")
    invalid_edges: List[Tuple[int, int, str]] = []
    for e in edges:
        if len(e) == 3:
            u, v, attrs = e
            for k, val in attrs.items():
                if val is None:
                    invalid_edges.append((u, v, k))
    if invalid_edges:
        print(f"❌ Found {len(invalid_edges)} edges with None attributes")
        for u, v, k in invalid_edges[: min(limit_errors, len(invalid_edges))]:
            print(f"   Edge {u}->{v} has {k}=None")
        return 2
    else:
        print("✅ All edge attributes valid")
        return 0


def attempt_dsgraph(nodes: List[Tuple[int, Dict[str, Any]]], edges: List[Tuple]) -> int:
    try:
        from deepsnap.graph import Graph as DSGraph  # type: ignore
    except Exception as e:
        print(f"⚠️  Skipping DSGraph construction (import failed): {e}")
        return 1

    if nx is None:
        print("⚠️  NetworkX not available; cannot attempt DSGraph construction")
        return 1

    G = nx.Graph()
    G.add_nodes_from(nodes)
    # Drop non-numeric edge attrs to mimic loader behavior
    cleaned_edges = []
    for e in edges:
        if len(e) == 3:
            u, v, attr = e
            cleaned_attr = {
                k: v for k, v in attr.items() if isinstance(v, (int, float))
            }
            cleaned_edges.append((u, v, cleaned_attr))
        else:
            cleaned_edges.append(e)
    G.add_edges_from(cleaned_edges)

    # Ensure node_feature exists so DSGraph doesn't complain
    for n in G.nodes:
        if "node_feature" not in G.nodes[n]:
            G.nodes[n]["node_feature"] = torch.tensor([1.0], dtype=torch.float)

    try:
        _ = DSGraph(G)
        print("✅ DSGraph construction succeeded")
        return 0
    except Exception as e:
        print(f"❌ DSGraph construction failed: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(
        description="Validate node attributes for DSGraph compatibility"
    )
    parser.add_argument(
        "pkl", help="Path to pickle file (raw_data dict or NetworkX graph)"
    )
    parser.add_argument(
        "--sample", type=int, default=5, help="Number of nodes to sample for printing"
    )
    parser.add_argument(
        "--limit-errors", type=int, default=20, help="Max errors to print per category"
    )
    parser.add_argument(
        "--attempt-dsgraph",
        action="store_true",
        help="Attempt DSGraph construction as a final check",
    )
    args = parser.parse_args()

    print(f"📦 Loading {args.pkl} ...")
    try:
        obj = load_pickle(args.pkl)
    except Exception as e:
        print(f"FATAL: failed to load pickle: {e}")
        sys.exit(2)

    try:
        nodes, edges = normalize_raw_data(obj)
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(2)

    rc_nodes = check_nodes(nodes, sample=args.sample, limit_errors=args.limit_errors)
    rc_edges = check_edges(edges, limit_errors=args.limit_errors)

    rc_ds = 0
    if args.attempt_dsgraph:
        rc_ds = attempt_dsgraph(nodes, edges)

    # Final exit code prioritizes errors (2). If only warnings (1), return 1.
    final_rc = max(rc_nodes, rc_edges, rc_ds)
    if final_rc == 0:
        print("\nNode validation completed: OK")
    elif final_rc == 1:
        print("\nNode validation completed: WARNINGS present")
    else:
        print("\nNode validation completed: ERRORS detected")
    sys.exit(final_rc)


if __name__ == "__main__":
    main()
