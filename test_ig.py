"""igraph version of test.py - Test all imports including igraph."""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy
import torch
import igraph
import deepsnap
import matplotlib
import seaborn
import scipy
import sklearn 
import torch_geometric
import test_tube
import tqdm

print("All imports successful!")
print(f"igraph version: {igraph.__version__}")
