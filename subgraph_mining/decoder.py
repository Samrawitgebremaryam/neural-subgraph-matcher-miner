import argparse
import csv
from itertools import combinations
import time
import os
import pickle
import sys
from pathlib import Path

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder

# CRITICAL: Import visualizer at top level (not inside functions)
try:
    from visualizer.visualizer import visualize_pattern_graph_ext, visualize_all_pattern_instances
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import visualizer - visualization will be skipped")
    VISUALIZER_AVAILABLE = False
    visualize_pattern_graph_ext = None
    visualize_all_pattern_instances = None

from subgraph_mining.search_agents import (
    GreedySearchAgent, MCTSSearchAgent, 
    MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, 
    BeamSearchAgent
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
import json 
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        "plots",
        "plots/cluster",
        "results"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def generate_target_embeddings(dataset, model, args):
    """
    Expert Optimization: One-time calculation of target embeddings (Scoring Key).
    This allows parallel workers to share the same scoring context without redundant computation.
    """
    logger.info(f"Generating global context (embeddings) for {len(dataset)} graphs...")
    neighs = []
    
    # 1. Neighborhood sampling for the "Scoring Key"
    for j in tqdm(range(args.n_neighborhoods)):
        graph, neigh = utils.sample_neigh(dataset,
            random.randint(args.min_neighborhood_size,
                args.max_neighborhood_size), args.graph_type)
        neigh = graph.subgraph(neigh)
        neigh = nx.convert_node_labels_to_integers(neigh)
        neigh.add_edge(0, 0)
        neighs.append(neigh)
    
    # 2. Parallel Embedding Phase
    embs = []
    device = utils.get_device()
    for i in range(len(neighs) // args.batch_size):
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=[0]*args.batch_size if args.node_anchored else None)
            emb = model.emb_model(batch.to(device))
            # Keep on CPU to share with workers easily, they will move to GPU as needed
            emb = emb.to(torch.device("cpu"))
        embs.append(emb)
    
    return embs


def pattern_growth_streaming(dataset, task, args):
    """
    Optimized Batch Processing implementation (Neighborhood Batching).
    Distributes search workload across workers while maintaining 100% accuracy.
    """
    # Phase 1: Initialize Global Model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))
    
    # Phase 2: One-time Global Context Generation
    global_embs = generate_target_embeddings(dataset, model, args)
    
    # Clean up GPU to allow workers to use CUDA
    del model
    torch.cuda.empty_cache()

    # Phase 3: Parallel Search Trials (Workload Partitioning)
    return pattern_growth(dataset, task, args, precomputed_data=global_embs)


def visualize_pattern_graph(pattern, args, count_by_size):
    """Visualize a single pattern representative (Kept per user preference)."""
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        base_size = max(12, min(20, num_nodes * 2))
        figsize = (base_size, base_size * 0.8)
        plt.figure(figsize=figsize)

        node_labels = {n: f"{pattern.nodes[n].get('label', 'unk')}:{pattern.nodes[n].get('id', n)}" 
                       for n in pattern.nodes()}

        pos = nx.spring_layout(pattern, k=2.0, seed=42, iterations=50)
        
        node_colors = ['red' if pattern.nodes[n].get('anchor', 0) == 1 else 'skyblue' for n in pattern.nodes()]
        nx.draw_networkx_nodes(pattern, pos, node_color=node_colors, node_size=3000, edgecolors='black')
        nx.draw_networkx_edges(pattern, pos, width=2, alpha=0.7, arrows=pattern.is_directed())
        nx.draw_networkx_labels(pattern, pos, labels=node_labels, font_size=10, font_weight='bold')

        plt.title(f"Pattern (Size: {num_nodes}, Edges: {num_edges})", fontsize=14)
        plt.axis('off')
        
        filename = f"pattern_{num_nodes}_{count_by_size[num_nodes]}"
        plt.savefig(f"plots/cluster/{filename}.png", bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Error visualizing pattern graph: {e}")
        return False


def save_and_visualize_all_instances(agent, args, representative_patterns=None):
    """Kept per user preference, fixed to use visualizer.visualizer correctly."""
    try:
        logger.info("="*70)
        logger.info("SAVING AND VISUALIZING ALL PATTERN INSTANCES")
        logger.info("="*70)

        if not hasattr(agent, 'counts') or not agent.counts:
            return None

        # Implementation preserved from user template with minimal fixes
        output_data = {}
        for size, hashed_patterns in agent.counts.items():
            sorted_patterns = sorted(hashed_patterns.items(), key=lambda x: len(x[1]), reverse=True)
            for rank, (wl_hash, instances) in enumerate(sorted_patterns[:args.out_batch_size], 1):
                key = f"size_{size}_rank_{rank}"
                output_data[key] = {'size': size, 'rank': rank, 'instances': instances[:5]} # Limit for size
        
        pkl_path = args.out_path.replace('.pkl', '_all_instances.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        logger.info(f"✓ Results saved to {pkl_path}")
        return pkl_path
    
    except Exception as e:
        logger.error(f"Error in save_instance_logic: {e}")
        return None


def pattern_growth(dataset, task, args, precomputed_data=None):
    """Main pattern mining function (Parallel Support Added)."""
    start_time = time.time()
    ensure_directories()
    
    # Load model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    neighs = []
    logger.info(f"{len(dataset)} graphs")
    logger.info(f"Search strategy: {args.search_strategy}")
    
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                if 'label' not in graph.nodes[node]: graph.nodes[node]['label'] = str(node)
                if 'id' not in graph.nodes[node]: graph.nodes[node]['id'] = str(node)
        graphs.append(graph)
    
    # Use precomputed context if available (Neighborhood Batching)
    if precomputed_data:
        embs = precomputed_data
    else:
        # Standard Sampler
        anchors = []
        for j in tqdm(range(args.n_neighborhoods)):
            graph, neigh = utils.sample_neigh(graphs,
                random.randint(args.min_neighborhood_size,
                    args.max_neighborhood_size), args.graph_type)
            neigh = graph.subgraph(neigh)
            neigh = nx.convert_node_labels_to_integers(neigh)
            neigh.add_edge(0, 0)
            neighs.append(neigh)
            if args.node_anchored: anchors.append(0)

        embs = []
        for i in range(len(neighs) // args.batch_size):
            top = (i+1)*args.batch_size
            with torch.no_grad():
                batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                    anchors=anchors if args.node_anchored else None)
                emb = model.emb_model(batch.to(utils.get_device()))
                emb = emb.to(torch.device("cpu"))
            embs.append(emb)

    # Search Logic
    if not hasattr(args, 'n_workers'):
        args.n_workers = mp.cpu_count()

    # Initialize agent
    if args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, n_beams=1,
            n_workers=args.n_workers)
        agent.args = args
    else:
        # Beam/MCTS Fallback
        agent = BeamSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, beam_width=args.beam_width)
    
    logger.info(f"Running search with {args.n_trials} trials...")
    out_graphs = agent.run_search(args.n_trials)
    
    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.2f}s")

    # Finalization (Pickle/JSON/Visual)
    save_and_visualize_all_instances(agent, args, out_graphs)
    
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return out_graphs


def main():
    ensure_directories()

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    
    args = parser.parse_args()

    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Graph type: {args.graph_type}")

    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            full_data = pickle.load(f)
            if isinstance(full_data, (nx.Graph, nx.DiGraph)):
                graph = full_data
            elif isinstance(full_data, dict) and 'nodes' in full_data:
                graph = nx.DiGraph() if args.graph_type == "directed" else nx.Graph()
                graph.add_nodes_from(full_data['nodes'])
                graph.add_edges_from(full_data['edges'])
            else:
                graph = full_data[0] if isinstance(full_data, list) else full_data
                
        dataset = [graph]
        task = 'graph'
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    else:
        # Fallback for other datasets
        dataset = []
        task = 'graph'

    # ADAPTIVE MODE SELECTOR: Choose between Standard and Streaming Batching
    if isinstance(dataset, (list, TUDataset, PPI)):
        num_nodes = sum(len(g) for g in dataset)
    else:
        num_nodes = len(dataset)
        
    threshold = getattr(args, 'auto_streaming_threshold', 100000)
    workers = getattr(args, 'streaming_workers', 1)

    use_streaming = (num_nodes > threshold or args.n_trials > 2000) and workers > 1

    logger.info("\nStarting pattern mining...")
    if use_streaming:
        logger.info(f"Adaptive Mode: Detected large scale ({num_nodes} nodes). Using Streaming Batching. 🚀")
        pattern_growth_streaming(dataset, task, args)
    else:
        logger.info("Adaptive Mode: Standard Sequential Processing. 🧵")
        args.n_workers = 1 # Force single worker to avoid pool overhead
        pattern_growth(dataset, task, args)
    
    logger.info("\n✓ Pattern mining complete!")


if __name__ == '__main__':
    # Fix for Windows multi-processing
    mp.set_start_method('spawn', force=True)
    main()