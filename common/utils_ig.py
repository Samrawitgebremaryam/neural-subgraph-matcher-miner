"""
igraph-native implementation of common/utils.py

This module replaces NetworkX with igraph (python-igraph) for all graph operations.
igraph is orders of magnitude more resource-efficient than NetworkX.

All functions work directly with igraph.Graph objects.
No converters or compatibility layers - pure igraph performance.
"""

from collections import defaultdict, Counter
import igraph as ig
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm
import warnings
import torch
import torch.optim as optim


def sample_neigh(graphs, size, graph_type="undirected"):
    """
    Sample a random neighborhood of specified size from igraph graphs.
    
    Selects a random graph weighted by size, then samples a connected
    neighborhood starting from a random node using BFS.
    
    Args:
        graphs: List of igraph.Graph objects
        size: Target neighborhood size
        graph_type: 'undirected' or 'directed'
        
    Returns:
        Tuple of (igraph.Graph subgraph, list of vertex indices in neighborhood)
    """
    if not graphs:
        raise ValueError("graphs must be a non-empty list")

    # Compatibility path: preserve original behavior for NetworkX inputs.
    if isinstance(graphs[0], nx.Graph):
        ps = np.array([len(g) for g in graphs], dtype=float)
        ps /= np.sum(ps)
        dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

        while True:
            idx = dist.rvs()
            graph = graphs[idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            if graph_type == "undirected":
                frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            elif graph_type == "directed":
                frontier = list(set(graph.successors(start_node)) - set(neigh))
            else:
                raise ValueError("graph_type must be 'undirected' or 'directed'")
            visited = set([start_node])
            while len(neigh) < size and frontier:
                new_node = random.choice(list(frontier))
                neigh.append(new_node)
                visited.add(new_node)
                if graph_type == "undirected":
                    frontier += list(graph.neighbors(new_node))
                else:
                    frontier += list(graph.successors(new_node))
                frontier = [x for x in frontier if x not in visited]
            if len(neigh) == size:
                return graph, neigh

    # Native igraph path.
    # Weight selection by graph size
    ps = np.array([len(g.vs) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    
    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        
        # Start from random vertex
        start_vertex = random.randint(0, len(graph.vs) - 1)
        neigh = [start_vertex]
        
        # Get neighbors based on graph type
        if graph_type == "undirected" or not graph.is_directed():
            frontier = list(set(graph.neighbors(start_vertex)) - set(neigh))
        else:  # directed
            frontier = list(set(graph.neighbors(start_vertex, mode=ig.OUT)) - set(neigh))
        
        visited = set([start_vertex])
        
        # Grow neighborhood via BFS
        while len(neigh) < size and frontier:
            new_vertex = random.choice(list(frontier))
            assert new_vertex not in neigh
            neigh.append(new_vertex)
            visited.add(new_vertex)
            
            # Expand frontier
            if graph_type == "undirected" or not graph.is_directed():
                frontier += graph.neighbors(new_vertex)
            else:
                frontier += graph.neighbors(new_vertex, mode=ig.OUT)
            
            frontier = [x for x in frontier if x not in visited]
        
        if len(neigh) == size:
            # Return induced subgraph
            subgraph = graph.induced_subgraph(neigh)
            return subgraph, neigh


cached_masks = None

def vec_hash(v):
    """Hash function for vectors."""
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    return v


def wl_hash(g, dim=64, node_anchored=False):
    """
    Weisfeiler-Lehman hash of an igraph graph.
    
    Computes WL hash by iteratively updating node features based on neighbor aggregates.
    
    Args:
        g: igraph.Graph object
        dim: Hash dimension
        node_anchored: If True, use anchor node attribute to seed computation
        
    Returns:
        Tuple of hash values (one per dimension) for the graph
    """
    n_vertices = len(g.vs)
    vecs = np.zeros((n_vertices, dim), dtype=int)
    
    # Initialize with anchor if provided
    if node_anchored:
        for v_id in range(n_vertices):
            if 'anchor' in g.vs[v_id].attributes():
                if g.vs[v_id]['anchor'] == 1:
                    vecs[v_id] = 1
                    break
    
    # WL iterations
    for iteration in range(n_vertices):
        newvecs = np.zeros((n_vertices, dim), dtype=int)
        
        for v_id in range(n_vertices):
            # Aggregate over vertex and neighbors
            neighbor_ids = [v_id] + g.neighbors(v_id)
            neighbor_vecs = vecs[neighbor_ids, :]
            aggregated = np.sum(neighbor_vecs, axis=0)
            newvecs[v_id] = vec_hash(aggregated)
        
        vecs = newvecs
    
    return tuple(np.sum(vecs, axis=0))


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    """
    Enumerate all subgraphs of size up to k in igraph graph G.
    
    Uses adaptive sampled ESU to reduce runtime for large graphs.
    
    Args:
        G: igraph.Graph object
        k: Maximum subgraph size
        progress_bar: Whether to show tqdm progress
        node_anchored: Whether to anchor subgraphs at starting node
        
    Returns:
        Dictionary mapping (size, wl_hash) -> list of igraph subgraphs
    """
    ps = np.arange(1.0, 0.0, -1.0 / (k + 1)) ** 1.5
    motif_counts = defaultdict(list)
    
    vertices = list(range(len(G.vs)))
    vertex_iter = tqdm(vertices) if progress_bar else vertices
    
    for node in vertex_iter:
        sg = set([node])
        v_ext = set()
        
        # Get neighbors with ID > node (for ordering)
        neighbors = [nbr for nbr in G.neighbors(node) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, min(n_samples, len(neighbors)))
        
        for nbr in neighbors:
            v_ext.add(nbr)
        
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    """
    Recursive subgraph extension for ESU enumeration.
    
    Args:
        G: igraph.Graph
        k: Max subgraph size
        sg: Current subgraph nodes (set)
        v_ext: Extension frontier (set)
        node_id: Anchor node ID
        motif_counts: Accumulator dictionary
        ps: Probability schedule
        node_anchored: Whether to anchor subgraph
    """
    # Base case: record current subgraph
    sg_vertex_ids = sorted(list(sg))
    sg_G = G.induced_subgraph(sg_vertex_ids)
    
    # Set anchor attribute if needed
    if node_anchored:
        for i, v_id in enumerate(sg_vertex_ids):
            if v_id == node_id:
                sg_G.vs[i]['anchor'] = 1
            else:
                sg_G.vs[i]['anchor'] = 0
    
    # Compute hash and record
    hash_val = wl_hash(sg_G, node_anchored=node_anchored)
    motif_counts[len(sg), hash_val].append(sg_G)
    
    if len(sg) == k:
        return
    
    # Recursive step
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        
        # Get valid neighbors of w
        neighbors = [nbr for nbr in G.neighbors(w) 
                     if nbr > node_id and nbr not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, min(n_samples, len(neighbors)))
        
        for nbr in neighbors:
            new_v_ext.add(nbr)
        
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps, node_anchored)
        sg.remove(w)


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    """
    Generate baseline query graphs via random ESU sampling.
    
    Args:
        queries: List of igraph.Graph query graphs
        targets: List of igraph.Graph target graphs
        node_anchored: Whether to mark anchor nodes
        
    Returns:
        List of sampled igraph subgraph queries
    """
    sizes = Counter([len(q.vs) for q in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
            progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size:
                total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    
    return out


def gen_baseline_queries_mfinder(queries, targets, n_samples=10000, node_anchored=False):
    """
    Generate baseline query graphs via random neighborhood sampling (mfinder-style).
    
    Args:
        queries: List of igraph.Graph query graphs
        targets: List of igraph.Graph target graphs
        n_samples: Number of random samples to draw per query size
        node_anchored: Whether to mark anchor nodes
        
    Returns:
        List of sampled igraph subgraph queries
    """
    sizes = Counter([len(q.vs) for q in queries])
    out = []
    
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        
        for i in tqdm(range(n_samples)):
            subgraph, neigh = sample_neigh(targets, size, graph_type="undirected")
            
            # Anchor at first node of neighborhood
            anchor_vertex_id = neigh[0]
            
            # Set anchor attribute
            for v_id in subgraph.vs:
                v_id['anchor'] = 0
            subgraph.vs[0]['anchor'] = 1  # First vertex in subgraph
            
            # Remove self-loops
            selfloops = [e.index for e in subgraph.es if e.source == e.target]
            subgraph.delete_edges(selfloops)
            
            hash_val = wl_hash(subgraph, node_anchored=node_anchored)
            counts[hash_val].append(subgraph)
        
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    
    return out


def parse_optimizer(parser):
    """Add optimizer-related arguments to argument parser."""
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
            help='Optimizer weight decay.')


def build_optimizer(args, params):
    """Build PyTorch optimizer from args."""
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    
    return scheduler, optimizer


def standardize_graph_ig(graph, anchor=None):
    """
    Standardize igraph attributes for processing.
    
    Args:
        graph: igraph.Graph object
        anchor: Optional anchor vertex index
        
    Returns:
        igraph.Graph with standardized attributes
    """
    # Copy graph to avoid modifying original
    g = graph.copy()
    
    # Ensure vertex attributes
    for v_id in g.vs:
        v_id['node_feature'] = torch.tensor([1.0])
        if 'label' not in v_id.attributes():
            v_id['label'] = str(v_id.index)
        if 'id' not in v_id.attributes():
            v_id['id'] = str(v_id.index)
    
    # Ensure edge attributes
    for e in g.es:
        if 'weight' not in e.attributes():
            e['weight'] = 1.0
        else:
            try:
                e['weight'] = float(e['weight'])
            except (ValueError, TypeError):
                e['weight'] = 1.0
    
    return g


def batch_nx_graphs(graphs, anchors=None):
    """
    Batch process igraph graphs for PyTorch training.
    
    Note: This function is kept for backward compatibility.
    It converts igraph to NetworkX for DeepSnap processing.
    
    Args:
        graphs: List of igraph.Graph objects
        anchors: Optional list of anchor vertex IDs per graph
        
    Returns:
        PyTorch Batch object on appropriate device
    """
    from deepsnap.graph import Graph as DSGraph
    from deepsnap.batch import Batch
    from common import feature_preprocess
    
    import networkx as nx
    
    # Convert igraph to NetworkX for DeepSnap compatibility.
    # If caller already passes NetworkX graphs, keep them as-is.
    nx_graphs = []
    for graph in graphs:
        if isinstance(graph, nx.Graph):
            nx_graphs.append(graph)
            continue

        ig_graph = graph
        if ig_graph.is_directed():
            g_nx = nx.DiGraph()
        else:
            g_nx = nx.Graph()

        for v in ig_graph.vs:
            v_attrs = dict(v.attributes())
            g_nx.add_node(v.index, **v_attrs)

        for e in ig_graph.es:
            e_attrs = dict(e.attributes())
            g_nx.add_edge(e.source, e.target, **e_attrs)

        nx_graphs.append(g_nx)
    
    # Initialize feature augmenter
    augmenter = feature_preprocess.FeatureAugment()
    
    # Process graphs with proper attribute handling
    processed_graphs = []
    for i, graph in enumerate(nx_graphs):
        anchor = anchors[i] if anchors is not None else None
        try:
            # Ensure node features
            for node in graph.nodes():
                if 'node_feature' not in graph.nodes[node]:
                    graph.nodes[node]['node_feature'] = torch.tensor([1.0])
            
            # Convert to DeepSnap format
            ds_graph = DSGraph(graph)
            processed_graphs.append(ds_graph)
            
        except Exception as e:
            print(f"Warning: Error processing graph {i}: {str(e)}")
            # Create minimal graph with basic features if conversion fails
            minimal_graph = nx.Graph()
            minimal_graph.add_nodes_from(graph.nodes())
            minimal_graph.add_edges_from(graph.edges())
            for node in minimal_graph.nodes():
                minimal_graph.nodes[node]['node_feature'] = torch.tensor([1.0])
            processed_graphs.append(DSGraph(minimal_graph))
    
    # Create batch
    batch = Batch.from_data_list(processed_graphs)
    
    # Suppress warnings during augmentation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown type of key*')
        batch = augmenter.augment(batch)
    
    return batch.to(get_device())


def get_device():
    """Get PyTorch device (GPU if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_gpu_memory():
    """Utility function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2 
    return 0
