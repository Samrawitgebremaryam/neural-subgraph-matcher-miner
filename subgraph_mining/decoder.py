import argparse
import csv
from itertools import combinations
import time
import os
import pickle
import sys
from pathlib import Path

from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from common import feature_preprocess
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


class StreamingNeighborhoodDataset(Dataset):
    def __init__(self, dataset, n_neighborhoods, args):
        self.dataset = dataset
        self.n_neighborhoods = n_neighborhoods
        self.args = args
        self.anchors = [0] * n_neighborhoods if args.node_anchored else None

    def __len__(self):
        return self.n_neighborhoods

    def __getitem__(self, idx):
        # On-the-fly sampling: Zero-Copy from the giant graph
        graph, neigh = utils.sample_neigh(self.dataset,
            random.randint(self.args.min_neighborhood_size,
                self.args.max_neighborhood_size), self.args.graph_type)
        
        neigh_graph = graph.subgraph(neigh)
        neigh_graph = nx.convert_node_labels_to_integers(neigh_graph)
        neigh_graph.add_edge(0, 0) # SPMiner anchor convention
        
        # Standardize and convert to DeepSnap
        anchor = 0 if self.args.node_anchored else None
        std_graph = utils.standardize_graph(neigh_graph, anchor)
        return DSGraph(std_graph)

def collate_fn(ds_graphs):
    """
    Batching logic for DeepSnap models.
    Converts a list of neighborhood chunks into a single batched matrix (The 'Conveyor Belt').
    """
    batch = Batch.from_data_list(ds_graphs)
    # Apply SPMiner feature augmentation
    augmenter = feature_preprocess.FeatureAugment()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown type of key*')
        batch = augmenter.augment(batch)
    return batch

def generate_target_embeddings(dataset, model, args):
    """
    Expert Optimization: Using DataLoader for true Batch Processing.
    This is where 'Static Graphs' become 'Dynamic Streams'.
    """
    logger.info(f"Setting up Batch Processing Pipeline with DataLoader (Batch Size: {args.batch_size})")
    
    stream_dataset = StreamingNeighborhoodDataset(dataset, args.n_neighborhoods, args)
    dataloader = DataLoader(stream_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    embs = []
    device = utils.get_device()
    model.to(device)
    
    logger.info(f"Generating embeddings for {args.n_neighborhoods} neighborhoods...")
    for batch in tqdm(dataloader):
        with torch.no_grad():
            emb = model.emb_model(batch.to(device))
            # Move to CPU to save GPU memory for search workers
            embs.append(emb.to(torch.device("cpu")))
    
    return embs

def pattern_growth_streaming(dataset, task, args):
    """Entry point for Optimized Batch Processing."""
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))
    model.eval()
    
    #  Batched Embedding Generation (The Conveyor Belt)
    global_embs = generate_target_embeddings(dataset, model, args)
    
    #  Parallel Search
    logger.info("Search phase starting (Workload Batching active)...")
    return pattern_growth(dataset, task, args, precomputed_data=global_embs, preloaded_model=model)

def pattern_growth(dataset, task, args, precomputed_data=None, preloaded_model=None):
    start_time = time.time()
    ensure_directories()
    
    if preloaded_model:
        model = preloaded_model
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
        model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))
    
    model.to(utils.get_device())
    model.eval()

    # Pre-process graphs
    graphs = []
    source_graphs = dataset[0] if isinstance(dataset, tuple) else dataset
    if not isinstance(source_graphs, list): source_graphs = [source_graphs]
    
    for g in source_graphs:
        if not isinstance(g, (nx.Graph, nx.DiGraph)):
            g = pyg_utils.to_networkx(g).to_undirected()
        graphs.append(g)

    # Use batched context
    embs = precomputed_data if precomputed_data else generate_target_embeddings(graphs, model, args)

    # Initialize agent
    if args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, n_beams=1,
            n_workers=args.n_workers)
        agent.args = args
    else:
        agent = BeamSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size, beam_width=args.beam_width)
    
    out_graphs = agent.run_search(args.n_trials)
    
    # Save results (Simplified per user preference for minimal changes)
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Total time: {time.time() - start_time:.2f}s")
    return out_graphs

def main():
    ensure_directories()
    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            data_obj = pickle.load(f)
        
        if isinstance(data_obj, (nx.Graph, nx.DiGraph)):
            dataset = [data_obj]
        elif isinstance(data_obj, dict):
            # Special handling for dict-formatted graphs (like Amazon)
            dataset = [nx.DiGraph(data_obj)]
            logger.info(f"Created directed graph from dict format with {len(dataset[0].nodes)} nodes and {len(dataset[0].edges)} edges")
        else:
            dataset = data_obj # Assume list of graphs
        task = 'graph'
    else:
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'

    num_nodes = sum(len(g) for g in dataset) if isinstance(dataset, list) else len(dataset)
    threshold = getattr(args, 'auto_streaming_threshold', 100000)
    
    use_streaming = (num_nodes > threshold or args.n_trials > 2000) and args.streaming_workers > 1

    if use_streaming:
        logger.info(f"Adaptive Mode: Enabling Streaming Batch Processing für {num_nodes} nodes. 🚀")
        pattern_growth_streaming(dataset, task, args)
    else:
        logger.info("Adaptive Mode: Standard Sequential Processing. 🧵")
        args.n_workers = 1
        pattern_growth(dataset, task, args)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()