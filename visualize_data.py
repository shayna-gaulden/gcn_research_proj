# Install required packages.
import argparse
import dgl
import dgl.sparse as dglsp
import torch.nn.functional as F
import torch.nn as nn
import os, torch, sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

# Create an argument parser
parser = argparse.ArgumentParser(
    prog='visualize_data',
    description='A program to visualize data sets from the dgl data loader',
    epilog=':)')

# Add an argument to accept the file name
parser.add_argument("dataset", type=str, help="The name of the datast")
# Parse the arguments
args = parser.parse_args()
dataset = args.dataset.upper()
# chck if allowed
DATASETS = ["CORA"]
if dataset not in DATASETS:
    print("Please enter a valid dataset from one of the following:.")
    for d in DATASETS:
        print(" * "+d)
    sys.exit()

print("Please wait...")

# Load graph from the existing dataset.
match dataset:
    # Nodes mean paper and edges mean citation relationships.
    # Row normalized
    case "CORA": 
        data = dgl.data.CoraGraphDataset()

g = data[0]
label_counts = dict(sorted(dict(Counter(g.ndata['label'].tolist())).items()))

# find in degres and out degrees
in_degree_counts = dict(sorted(dict(Counter(data[0].in_degrees().tolist())).items()))
out_degree_counts = dict(sorted(dict(Counter(data[0].out_degrees().tolist())).items()))
features = g.ndata['feat'].tolist()

plt.figure()
plt.bar(label_counts.keys(), label_counts.values())
plt.title(f"{dataset}: Number of nodes with each label")
plt.ylabel("Number of nodes")
plt.xlabel("Labels")
plt.savefig(f"figures/{dataset}_label_bar_plot.png")
print(f"Created graph: figures/{dataset}_label_bar_plot.png")
plt.close()

plt.figure()
plt.bar(in_degree_counts.keys(), in_degree_counts.values())
plt.title(f"{dataset}: Number of in-degrees with each label")
plt.ylabel("Number of nodes")
plt.xlabel("In-Degree")
plt.savefig(f"figures/{dataset}_indegree_bar_plot.png")
print(f"Created graph: figures/{dataset}_indegree_bar_plot.png")
plt.close()

plt.figure()
plt.bar(out_degree_counts.keys(), out_degree_counts.values())
plt.title(f"{dataset}: Number of out-degrees with each label")
plt.ylabel("Number of nodes")
plt.xlabel("Out-Degree")
plt.savefig(f"figures/{dataset}_outdegree_bar_plot.png")
print(f"Created graph: figures/{dataset}_outdegree_bar_plot.png")
plt.close()