# Install required packages.
import dgl
import dgl.sparse as dglsp
import torch.nn.functional as F
import torch.nn as nn
import os, sys
import torch
import argparse
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

# Uncomment below to install required packages. If the CUDA version is not 11.8,
# check the https://www.dgl.ai/pages/start.html to find the supported CUDA
# version and corresponding command to install DGL.
#!pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html > /dev/null

# try:
#     import dgl
#     installed = True
# except ImportError:
#     installed = False
# print("DGL installed!" if installed else "DGL not found!")

# Create an argument parser
parser = argparse.ArgumentParser(
    prog='train_gcn',
    description='A program to train GCNs with different data',
    epilog=':)')

# Add an argument to accept the file name
parser.add_argument("--dataset", type=str, help="The name of the dataset", default="CORA")
parser.add_argument("--preproc", type=str, help="The normmalization for the adjacency and degree matrix", default="SYM")
# Parse the arguments
args = parser.parse_args()
dataset = args.dataset.upper()
preproc = args.preproc.upper()
# chck if allowed
DATASETS = ["CORA"]
if dataset not in DATASETS:
    print("Please enter a valid dataset from one of the following:")
    for d in DATASETS:
        print(" * "+d)
    sys.exit()
if preproc not in ["SYM", "RW"]:
    print("Please enter a valid preprocessing technique:")
    for p in ["SYM", "RW"]:
        print(" * "+p)
    sys.exit()

class GCNLayer(nn.Module):
    def __init__(self, in_size, out_size, preproc="SYM"):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_size, out_size)
        self.preproc = preproc.upper()

    def forward(self, A, X):
        ########################################################################
        # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
        # Sparse Matrix API
        ########################################################################
        match self.preproc:
            case "SYM":
                I = dglsp.identity(A.shape)
                A_hat = A + I
                D_hat = dglsp.diag(A_hat.sum(0))
                D_hat_invsqrt = D_hat ** -0.5
                return D_hat_invsqrt @ A_hat @ D_hat_invsqrt @ self.W(X)
            case "RW":
                I = dglsp.identity(A.shape)
                A_hat = A + I
                D_hat = dglsp.diag(A_hat.sum(0))
                D_hat_invsqrt = D_hat ** -1
                return D_hat_invsqrt @ A_hat @ self.W(X)
            case _:
                I = dglsp.identity(A.shape)
                A_hat = A + I
                D_hat = dglsp.diag(A_hat.sum(0))
                D_hat_invsqrt = D_hat ** -0.5
                return D_hat_invsqrt @ A_hat @ D_hat_invsqrt @ self.W(X)

# Create a GCN with the GCN layer.


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, preproc="SYM"):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_size, hidden_size, preproc)
        self.conv2 = GCNLayer(hidden_size, out_size, preproc)
        self.preproc = preproc

    def forward(self, A, X):
        X = self.conv1(A, X)
        X = F.relu(X)
        return self.conv2(A, X)


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g):
    features = g.ndata["feat"]
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    for epoch in range(100):
        model.train()

        # Forward.
        logits = model(A, features)

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}"
                f", test acc: {test_acc:.3f}"
            )

# Load graph from the existing dataset.
match dataset:
    # Nodes mean paper and edges mean citation relationships.
    # Row normalized
    case "CORA": 
        dataset = dgl.data.CoraGraphDataset()
print(dataset[0])

g = dataset[0]

print(g.ndata['feat'])
# Create model.
feature = g.ndata['feat']
in_size = feature.shape[1]
out_size = dataset.num_classes
gcn_model = GCN(in_size, out_size, 16, preproc)

# Kick off training.
train(gcn_model, g)
