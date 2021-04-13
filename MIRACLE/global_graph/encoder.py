import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from .graph_conv import SparseGraphConvolution, GraphConvolution, AttnGraphConvolution


class GCNEncoder(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False,
                 sparse: bool = True):
        super(GCNEncoder, self).__init__()
        self.input_dim = num_features
        self.features_nonzero = features_nonzero if sparse else None
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse
        GC = SparseGraphConvolution if sparse else GraphConvolution
        self.gc1 = GC(in_features=num_features, out_features=args.hidden1, bias=bias)
        self.gc2 = GC(in_features=args.hidden1, out_features=args.hidden2, bias=bias)

    def forward(self, features: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        if not self.sparse:
            features = self.dropout(features)
        hidden1 = F.relu(self.gc1(features, adj))
        hidden1 = self.dropout(hidden1)
        embeddings = self.gc2(hidden1, adj)
        return embeddings


class GCNEncoderWithFeatures(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False,
                 sparse: bool = True):
        super(GCNEncoderWithFeatures, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse
        GC = SparseGraphConvolution if sparse else GraphConvolution
        self.gc_input = GC(in_features=num_features, out_features=args.gcn_hidden1, bias=bias)
        self.gc_hidden1 = GC(in_features=args.gcn_hidden1, out_features=args.gcn_hidden2, bias=bias)
        self.gc_hidden2 = GC(in_features=args.gcn_hidden2, out_features=args.gcn_hidden3, bias=bias)
        self.trans_h = nn.Linear(args.gcn_hidden1 + num_features, args.gcn_hidden1, bias=True)
        self.trans_h1 = nn.Linear(args.gcn_hidden2 + num_features, args.gcn_hidden2, bias=True)
        self.trans_h2 = nn.Linear(args.gcn_hidden3 + num_features, args.gcn_hidden3, bias=True)

    def forward(self, features: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        if not self.sparse:
            features = self.dropout(features)
        hidden1 = F.relu(self.trans_h(torch.cat([self.gc_input(features, adj), features], dim=1)))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.trans_h1(torch.cat([self.gc_hidden1(hidden1, adj), features], dim=1)))
        hidden2 = self.dropout(hidden2)
        embeddings = F.relu(self.trans_h2(torch.cat([self.gc_hidden2(hidden2, adj), features], dim=1)))
        return embeddings