"""Compatibility wrapper for optional rustworkx usage.

This wrapper provides a small set of functions used in the miner/decoder
to allow swapping in rustworkx later. Right now it prefers rustworkx when
available but falls back to NetworkX implementations so changes are safe.

Keep this file minimal and extend it with rustworkx-specific implementations
if you want to fully leverage rustworkx performance later.
"""
import logging
import os
try:
    import rustworkx as rx
    RX_AVAILABLE = True
except Exception:
    rx = None
    RX_AVAILABLE = False

if os.environ.get("NSMM_DISABLE_RX", "0") == "1":
    RX_AVAILABLE = False
    rx = None

import networkx as nx
from typing import Iterable, List, Any, Dict
import weakref

logger = logging.getLogger(__name__)
if RX_AVAILABLE:
    logger.info("rustworkx available: using rx for structural ops")
else:
    logger.info("rustworkx not available: falling back to networkx")

# One-time debug flag to avoid spamming logs when neighbors() is called repeatedly
_rx_neighbors_debug_logged = False

# Cache converted graphs: WeakKeyDictionary keyed by original NetworkX graph
# value is tuple: (rx_graph, node_to_index, index_to_node, adjacency_dict)
_nx_to_rx_cache = weakref.WeakKeyDictionary()


def is_rx_graph(G: Any) -> bool:
    if not RX_AVAILABLE:
        return False
    # best-effort type check; keep tolerant if API changes
    return hasattr(rx, 'PyGraph') and isinstance(G, getattr(rx, 'PyGraph', object)) or (
        hasattr(rx, 'PyDiGraph') and isinstance(G, getattr(rx, 'PyDiGraph', object)))


def convert_nx_to_rx_cache(G: Any) -> bool:
    """Ensure a NetworkX graph `G` has a cached rustworkx conversion.

    Returns True if conversion/caching succeeded, False otherwise.
    """
    if not RX_AVAILABLE:
        return False
    try:
        if not isinstance(G, nx.Graph):
            return False
        if G in _nx_to_rx_cache:
            return True
        nodes = list(G.nodes())
        node_to_index = {n: i for i, n in enumerate(nodes)}
        index_to_node = nodes
        rx_graph = rx.PyGraph() if not G.is_directed() else rx.PyDiGraph()
        for _ in nodes:
            rx_graph.add_node(None)
        for u, v in G.edges():
            ui = node_to_index[u]
            vi = node_to_index[v]
            try:
                rx_graph.add_edge(ui, vi, None)
            except Exception:
                pass
        # build adjacency
        adj = {i: [] for i in range(len(nodes))}
        for u, v in G.edges():
            ui = node_to_index[u]
            vi = node_to_index[v]
            adj[ui].append(vi)
            if not G.is_directed():
                adj[vi].append(ui)

        _nx_to_rx_cache[G] = (rx_graph, node_to_index, index_to_node, adj)
        return True
    except Exception as e:
        logger.debug(f"rx_utils.convert_nx_to_rx_cache failed: {e}")
        return False


def convert_graphs(graphs: Iterable[Any]) -> int:
    """Attempt to convert an iterable of graphs to rustworkx-backed cache.

    Returns the number of graphs successfully cached.
    """
    n = 0
    if not RX_AVAILABLE:
        return 0
    for G in graphs:
        try:
            if isinstance(G, nx.Graph):
                if convert_nx_to_rx_cache(G):
                    n += 1
        except Exception:
            continue
    if n > 0:
        logger.info(f"rx_utils: converted {n} NetworkX graphs to rustworkx-backed cache")
    return n


class CompactGraph:
    """Lightweight graph wrapper storing only structural data.

    Provides a minimal NetworkX-like interface used by the search code:
    - attribute `nodes` as a list of node ids
    - `neighbors(node)` method returning neighbor ids
    - `subgraph(nodes_iterable)` method returning a NetworkX Graph of those nodes
    - `degree(node)` method
    """
    __slots__ = ('index_to_node', 'node_to_index', 'adj', 'nodes')

    def __init__(self, index_to_node: List[Any], adj: List[List[int]]):
        self.index_to_node = list(index_to_node)
        self.node_to_index = {n: i for i, n in enumerate(self.index_to_node)}
        # adjacency uses indices -> list of indices
        self.adj = [list(row) for row in adj]
        # expose nodes as a list attribute (NetworkX has .nodes view)
        self.nodes = list(self.index_to_node)

    def neighbors(self, node: Any) -> List[Any]:
        idx = self.node_to_index.get(node)
        if idx is None:
            return []
        return [self.index_to_node[i] for i in self.adj[idx]]

    def __len__(self) -> int:
        return len(self.index_to_node)

    def subgraph(self, nodes_iterable: Iterable[Any]) -> nx.Graph:
        NG = nx.Graph()
        nodes = list(nodes_iterable)
        NG.add_nodes_from(nodes)
        for u in nodes:
            for v in self.neighbors(u):
                if v in NG and u in NG:
                    NG.add_edge(u, v)
        return NG

    def degree(self, node: Any) -> int:
        return len(self.neighbors(node))


def build_compact_graphs(graphs: Iterable[nx.Graph]) -> List[CompactGraph]:
    """Create compact adjacency-only wrappers for a list of NetworkX graphs.

    Each CompactGraph stores index_to_node (original node ids) and an
    adjacency list of integer indices. This is much smaller than keeping
    full NetworkX objects in worker processes.
    """
    out = []
    for G in graphs:
        try:
            nodes = list(G.nodes())
            node_to_index = {n: i for i, n in enumerate(nodes)}
            adj = [[] for _ in nodes]
            for u, v in G.edges():
                ui = node_to_index[u]
                vi = node_to_index[v]
                adj[ui].append(vi)
                if not G.is_directed():
                    adj[vi].append(ui)
            out.append(CompactGraph(nodes, adj))
        except Exception:
            # on failure, fall back to an empty compact graph
            out.append(CompactGraph([], []))
    return out


def neighbors(G: Any, node: Any) -> List[Any]:
    """Return list of neighbors for `node` in graph `G`.

    Works for NetworkX graphs and for objects that implement a `neighbors` method.
    For rustworkx graphs the current implementation will fall back to converting
    to networkx if necessary (conservative but safe).
    """
    # NetworkX path (most common)
    if hasattr(G, 'neighbors'):
        return list(G.neighbors(node))

    # rustworkx path (best-effort): try to access adjacency via public API
    global _rx_neighbors_debug_logged
    if RX_AVAILABLE and is_rx_graph(G):
        # one-time log to indicate rustworkx path is active
        if not _rx_neighbors_debug_logged:
            logger.info('rx_utils: rustworkx neighbors path active')
            _rx_neighbors_debug_logged = True
        try:
            # rustworkx currently doesn't hold attribute dicts in the same way
            # so we extract adjacency lists via method if present
            adj = getattr(G, 'adj', None)
            if adj is not None and node in adj:
                return list(adj[node].keys())
        except Exception:
            pass

    # If we have a NetworkX graph but rustworkx is available, try to convert
    # the NetworkX graph to a rustworkx graph once and cache the result. If
    # conversion fails, fall back to NetworkX neighbors.
    try:
        if RX_AVAILABLE and isinstance(G, nx.Graph):
            # ensure conversion exists in cache
            if G not in _nx_to_rx_cache:
                try:
                    nodes = list(G.nodes())
                    node_to_index = {n: i for i, n in enumerate(nodes)}
                    index_to_node = nodes
                    # create rx graph
                    rx_graph = rx.PyGraph() if not G.is_directed() else rx.PyDiGraph()
                    # add placeholder nodes
                    for _ in nodes:
                        rx_graph.add_node(None)
                    # add edges
                    for u, v in G.edges():
                        ui = node_to_index[u]
                        vi = node_to_index[v]
                        try:
                            rx_graph.add_edge(ui, vi, None)
                        except Exception:
                            # ignore individual edge failures
                            pass
                    # build adjacency from rx_graph if possible, else from NX
                    adj = {}
                    try:
                        # try to use rx_graph.adj if available
                        raw_adj = getattr(rx_graph, 'adj', None)
                        if raw_adj is not None:
                            for idx, nbrs in enumerate(raw_adj):
                                if nbrs is None:
                                    adj[idx] = []
                                else:
                                    try:
                                        adj[idx] = list(nbrs.keys())
                                    except Exception:
                                        adj[idx] = []
                        else:
                            # fallback to reading edges from NX
                            adj = {i: [] for i in range(len(nodes))}
                            for u, v in G.edges():
                                ui = node_to_index[u]
                                vi = node_to_index[v]
                                adj[ui].append(vi)
                                # for undirected graphs, add symmetric
                                if not G.is_directed():
                                    adj[vi].append(ui)
                    except Exception:
                        # final fallback: adjacency from NX
                        adj = {i: [] for i in range(len(nodes))}
                        for u, v in G.edges():
                            ui = node_to_index[u]
                            vi = node_to_index[v]
                            adj[ui].append(vi)
                            if not G.is_directed():
                                adj[vi].append(ui)

                    _nx_to_rx_cache[G] = (rx_graph, node_to_index, index_to_node, adj)
                except Exception as e:
                    logger.warning(f"rx_utils: failed to convert NX->rx graph: {e}")

            # if conversion succeeded, use cached rx adjacency
            if G in _nx_to_rx_cache:
                rx_graph, node_to_index, index_to_node, adj = _nx_to_rx_cache[G]
                if not _rx_neighbors_debug_logged:
                    logger.info('rx_utils: using cached rustworkx-backed adjacency for neighbors')
                    _rx_neighbors_debug_logged = True
                idx = node_to_index.get(node)
                if idx is None:
                    return []
                nbrs_idx = adj.get(idx, [])
                return [index_to_node[i] for i in nbrs_idx]
    except Exception:
        # conversion attempt failed, continue to NetworkX fallback
        pass

    # as a final fallback, try iteration over edges
    try:
        return [v for u, v in G.edges() if u == node]  # type: ignore
    except Exception:
        return []

def subgraph_from_nodes(G: Any, nodes: Iterable[Any], attrs: Dict[Any, Dict] = None, directed: bool = False) -> nx.Graph:
    """Construct a NetworkX subgraph from structural graph `G` using the provided
    `nodes` iterable. Reattach attributes from `attrs` mapping (node -> attr dict)
    if provided.
    """
    if directed:
        NG = nx.DiGraph()
    else:
        NG = nx.Graph()

    nodes = list(nodes)
    NG.add_nodes_from(nodes)

    # add edges by querying neighbors
    for u in nodes:
        for v in neighbors(G, u):
            if v in NG and u in NG:
                NG.add_edge(u, v)

    # attach attrs if present
    if attrs:
        for n, d in attrs.items():
            if n in NG:
                NG.nodes[n].update(d)

    return NG

def connected_components(G: Any):
    if hasattr(nx, 'connected_components') and not (RX_AVAILABLE and is_rx_graph(G)):
        return nx.connected_components(G)
    # fallback: compute with networkx conversion
    NG = subgraph_from_nodes(G, G.nodes())
    return nx.connected_components(NG)


def degree(G: Any, node: Any) -> int:
    try:
        return G.degree(node)
    except Exception:
        # fall back to counting neighbors
        return len(neighbors(G, node))


def is_directed(G: Any) -> bool:
    try:
        return G.is_directed()
    except Exception:
        return False
