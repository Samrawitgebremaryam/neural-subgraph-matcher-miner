"""igraph version of alignment.py - Subgraph matching alignment using igraph graphs."""

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import igraph as ig
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
import matplotlib.pyplot as plt

from common import data
from common import models
from common import utils_ig as utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.test import validation
from subgraph_matching.train import build_model

def gen_alignment_matrix(model, query, target, method_type="order"):
    """Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: the subgraph matching model. Must have been trained with
            node anchored setting (--node_anchored, default)
        query: the query graph (igraph.Graph)
        target: the target graph (igraph.Graph)
        method_type: the method used for the model.
            "order" for order embedding or "mlp" for MLP model
    """

    mat = np.zeros((len(query.vs), len(target.vs)))
    for i, u in enumerate(query.vs):
        for j, v in enumerate(target.vs):
            # Use igraph vertex objects, convert to indices for anchors
            u_idx = u.index
            v_idx = v.index
            batch = utils.batch_nx_graphs([query, target], anchors=[u_idx, v_idx])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)
            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]
            mat[i][j] = raw_pred.item()
    return mat

def main():
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    parser = argparse.ArgumentParser(description='Alignment arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    args = parser.parse_args()
    args.test = True
    
    if args.query_path:
        with open(args.query_path, "rb") as f:
            query = pickle.load(f)
    else:
        # Generate random igraph instead of NetworkX
        query = ig.Graph.Erdos_Renyi(8, 0.25)
    
    if args.target_path:
        with open(args.target_path, "rb") as f:
            target = pickle.load(f)
    else:
        # Generate random igraph instead of NetworkX
        target = ig.Graph.Erdos_Renyi(16, 0.25)

    model = build_model(args)
    mat = gen_alignment_matrix(model, query, target,
        method_type=args.method_type)

    np.save("results/alignment.npy", mat)
    print("Saved alignment matrix in results/alignment.npy")

    plt.imshow(mat, interpolation="nearest")
    plt.savefig("plots/alignment.png")
    print("Saved alignment matrix plot in plots/alignment.png")

if __name__ == '__main__':
    main()
