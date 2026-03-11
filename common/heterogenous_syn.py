"""
Phase 2: Heterogeneous Synthetic Graph Generator

Extends combined_syn.py to produce graphs with node and edge type labels.

Design principles:
  - Fully generic: zero hardcoded domain vocabulary anywhere
  - Caller supplies node_types and edge_types at runtime
  - Graceful degradation: if no types supplied, behaves exactly
    like the original combined_syn.py (structure-only)
  - valid_edges is always optional
  - Backward compatible: drop-in replacement for combined_syn equivalents

Usage examples:

  # 1. Structure-only — identical to original combined_syn.py
  gen = get_heterogeneous_generator(sizes)

  # 2. Generic — caller defines the vocabulary
  gen = get_heterogeneous_generator(
      sizes,
      node_types=["A", "B", "C"],
      edge_types=["x", "y", "z"],
  )

  # 3. With edge constraints — any domain
  gen = get_heterogeneous_generator(
      sizes,
      node_types=["Gene", "Protein", "Drug", "Disease"],
      edge_types=["regulates", "binds", "treats", "encodes"],
      valid_edges={
          ("Gene",    "Protein"): ["encodes"],
          ("Drug",    "Disease"): ["treats"],
          ("Protein", "Protein"): ["binds"],
          ("Gene",    "Gene"):    ["regulates"],
      }
  )

  # 4. Any other domain — finance, social, knowledge graphs, etc.
  gen = get_heterogeneous_generator(
      sizes,
      node_types=["Account", "Bank", "Transaction"],
      edge_types=["transfers", "owns", "audits"],
  )
"""

import numpy as np
import deepsnap.dataset as dataset

from common.combined_syn import (
    ERGenerator,
    WSGenerator,
    BAGenerator,
    PowerLawClusterGenerator,
)


# ---------------------------------------------------------------------------
# Core labelling helpers
# ---------------------------------------------------------------------------

def assign_node_types(graph, node_types):
    """
    Randomly assign a type label to every node.

    Args:
        graph:      NetworkX graph (modified in place)
        node_types: non-empty list of type strings supplied by caller

    Returns:
        graph with graph.nodes[n]['label'] set on every node
    """
    for node in graph.nodes():
        graph.nodes[node]["label"] = np.random.choice(node_types)
    return graph


def assign_edge_types(graph, edge_types, valid_edges=None):
    """
    Randomly assign a type to every edge.

    Args:
        graph:       NetworkX graph (modified in place)
        edge_types:  non-empty list of edge type strings
        valid_edges: optional dict {(node_type_A, node_type_B): [edge_types]}
                     Both orderings of the pair are checked automatically.
                     Edges whose pair has no entry fall back to edge_types.

    Returns:
        graph with graph.edges[u, v]['type'] set on every edge
    """
    for u, v in graph.edges():
        if valid_edges:
            u_label = graph.nodes[u].get("label", "")
            v_label = graph.nodes[v].get("label", "")
            allowed = (
                valid_edges.get((u_label, v_label))
                or valid_edges.get((v_label, u_label))
                or edge_types
            )
        else:
            allowed = edge_types
        graph.edges[u, v]["type"] = np.random.choice(allowed)
    return graph


def make_heterogeneous(graph, node_types=None,
                       edge_types=None, valid_edges=None):
    """
    Add node and edge type labels to any NetworkX graph.

    Graceful degradation:
      - node_types=None → nodes get no 'label' attribute (structure-only)
      - edge_types=None → edges get no 'type'  attribute (structure-only)
      - valid_edges=None → all edge types equally likely on all edges

    Args:
        graph:       a connected NetworkX graph
        node_types:  list of type strings, or None
        edge_types:  list of type strings, or None
        valid_edges: dict or None

    Returns:
        graph, possibly enriched with 'label' and 'type' attributes
    """
    if node_types:
        graph = assign_node_types(graph, node_types)
    if edge_types:
        graph = assign_edge_types(graph, edge_types, valid_edges)
    return graph


# ---------------------------------------------------------------------------
# Heterogeneous generators — one per topology, mirroring combined_syn.py
# ---------------------------------------------------------------------------

class HeterogeneousERGenerator(ERGenerator):
    """
    Erdos-Renyi random graph with optional node/edge type labels.
    Falls back to structure-only when node_types or edge_types is None.
    """

    def __init__(self, sizes, node_types=None, edge_types=None,
                 valid_edges=None, p_alpha=1.3, **kwargs):
        super().__init__(sizes, p_alpha=p_alpha, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(
            graph, self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousWSGenerator(WSGenerator):
    """
    Watts-Strogatz graph with optional node/edge type labels.
    Falls back to structure-only when node_types or edge_types is None.
    """

    def __init__(self, sizes, node_types=None, edge_types=None,
                 valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(
            graph, self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousBAGenerator(BAGenerator):
    """
    Barabasi-Albert graph with optional node/edge type labels.
    Falls back to structure-only when node_types or edge_types is None.
    """

    def __init__(self, sizes, node_types=None, edge_types=None,
                 valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(
            graph, self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousPowerLawGenerator(PowerLawClusterGenerator):
    """
    PowerLaw cluster graph with optional node/edge type labels.
    Falls back to structure-only when node_types or edge_types is None.
    """

    def __init__(self, sizes, node_types=None, edge_types=None,
                 valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(
            graph, self.node_types, self.edge_types, self.valid_edges)


# ---------------------------------------------------------------------------
# Ensemble builder and dataset factory — drop-in replacements
# ---------------------------------------------------------------------------

def get_heterogeneous_generator(sizes, size_prob=None, dataset_len=None,
                                node_types=None, edge_types=None,
                                valid_edges=None):
    """
    Ensemble of all 4 heterogeneous generators.
    Drop-in replacement for combined_syn.get_generator().

    Args:
        sizes:       array of possible graph sizes
        size_prob:   probability weights per size (optional)
        dataset_len: number of graphs to generate (optional)
        node_types:  list of node type strings, or None (structure-only)
        edge_types:  list of edge type strings, or None (structure-only)
        valid_edges: optional dict constraining edge types per node pair

    Returns:
        deepsnap.dataset.EnsembleGenerator
    """
    shared = dict(
        node_types=node_types,
        edge_types=edge_types,
        valid_edges=valid_edges,
        size_prob=size_prob,
    )
    return dataset.EnsembleGenerator(
        [
            HeterogeneousERGenerator(sizes, **shared),
            HeterogeneousWSGenerator(sizes, **shared),
            HeterogeneousBAGenerator(sizes, **shared),
            HeterogeneousPowerLawGenerator(sizes, **shared),
        ],
        dataset_len=dataset_len,
    )


def get_heterogeneous_dataset(task, dataset_len, sizes,
                               size_prob=None,
                               node_types=None,
                               edge_types=None,
                               valid_edges=None,
                               **kwargs):
    """
    GraphDataset using heterogeneous generators.
    Drop-in replacement for combined_syn.get_dataset().

    Args:
        task:        deepsnap task string e.g. "graph"
        dataset_len: number of graphs
        sizes:       array of possible graph sizes
        node_types:  list of node type strings, or None (structure-only)
        edge_types:  list of edge type strings, or None (structure-only)
        valid_edges: optional dict constraining edge types per node pair
        **kwargs:    passed through to GraphDataset

    Returns:
        deepsnap.dataset.GraphDataset
    """
    generator = get_heterogeneous_generator(
        sizes,
        size_prob=size_prob,
        dataset_len=dataset_len,
        node_types=node_types,
        edge_types=edge_types,
        valid_edges=valid_edges,
    )
    return dataset.GraphDataset(
        None, task=task, generator=generator, **kwargs)


# ---------------------------------------------------------------------------
# Demo — run with: python common/heterogeneous_syn.py
# Same style as combined_syn.py main()
# ---------------------------------------------------------------------------

def main():
    sizes = np.arange(6, 31)

    def show(g, label=""):
        print(f"\n  {label}")
        for n, d in list(g.nodes(data=True))[:2]:
            print(f"    node {n}: label={d.get('label', 'NONE')}")
        for u, v, d in list(g.edges(data=True))[:2]:
            print(f"    edge {u}->{v}: type={d.get('type', 'NONE')}")

    # 1. Structure-only — no types passed, identical to combined_syn.py
    print("=== 1. Structure-only (graceful degradation) ===")
    ds = get_heterogeneous_dataset("graph", dataset_len=2, sizes=sizes)
    show(ds[0].G, "no node/edge types")

    # 2. Generic caller-defined vocabulary
    print("\n=== 2. Generic caller-defined types ===")
    ds2 = get_heterogeneous_dataset(
        "graph", dataset_len=2, sizes=sizes,
        node_types=["A", "B", "C"],
        edge_types=["x", "y", "z"],
    )
    show(ds2[0].G, "generic types")

    # 3. Caller-supplied domain with edge constraints
    print("\n=== 3. Domain with edge constraints (caller-supplied) ===")
    ds3 = get_heterogeneous_dataset(
        "graph", dataset_len=2, sizes=sizes,
        node_types=["Gene", "Protein", "Disease", "Drug"],
        edge_types=["regulates", "binds", "treats", "encodes"],
        valid_edges={
            ("Gene",    "Protein"):  ["encodes"],
            ("Drug",    "Disease"):  ["treats"],
            ("Protein", "Protein"):  ["binds"],
            ("Gene",    "Gene"):     ["regulates"],
        }
    )
    show(ds3[0].G, "domain with constraints")

    print("\nDone.")


if __name__ == "__main__":
    main()