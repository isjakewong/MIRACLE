import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Tuple, List, Dict
from .encoder import GCNEncoder, GCNEncoderWithFeatures
from .decoder import InnerProductDecoder
from features import BatchMolGraph
from models.mpn import MPN
from data.mol_tree import Vocab
from argparse import Namespace
import numpy as np
from sklearn.svm import SVC, SVR
import joblib

from models.mpn import PairMPN
from models.ggnn import GGNN
from models.smiles import SmilesNN
from models.jtnn_enc import JTNNEncoder
from models.jt_mpn import JunctionTreeGraphNN
from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights
from models.feature_loader import Mol2vecLoader
from models.pooling import *
from data.data import mol2sentence
from .model_info import shortcut
from .deepinfomax import GcnInfomax

class HierGlobalGCN(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(HierGlobalGCN, self).__init__()
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.bias = bias
        self.sparse = sparse
        self.args = args
        self.create_encoder(args)       
        self.global_enc = self.select_encoder(args)
        self.dec_local = InnerProductDecoder(args.hidden_size)
        self.dec_global = InnerProductDecoder(args.hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.DGI_setup()
        self.create_ffn(args)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        if not args.smiles_based:
            if args.graph_encoder == 'ggnn':
                self.encoder = GGNN(args)
            else:
                self.encoder = MPN(args)
        else:
            self.encoder = SmilesNN(args)

        if args.jt:
            self.encoder = JTNNEncoder(vocab, args.hidden_size) if args.jt_encoder == 'tree' else \
                JunctionTreeGraphNN(args)

        return self.encoder

    def select_encoder(self, args: Namespace):
        return GCNEncoderWithFeatures(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)

    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        self.fusion_ffn_local = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fusion_ffn_global = nn.Linear(args.gcn_hidden3, args.ffn_hidden_size)
        ffn = []
        # after fusion layer
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout

    def DGI_setup(self):
        self.DGI_model = GcnInfomax(self.args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        smiles_batch = batch
        features_batch = None
        feat_orig = self.encoder(smiles_batch, features_batch)

        feat = self.dropout(feat_orig)
        fused_feat = self.fusion_ffn_local(feat)
        output = self.ffn(fused_feat)
        outputs = self.sigmoid(output)       
        outputs_l = outputs.view(-1)
        embeddings = self.global_enc(feat_orig, adj)

        feat_g = self.dropout(embeddings)
        fused_feat_g = self.fusion_ffn_global(feat_g)
        output_g = self.ffn(fused_feat_g)
        outputs_ = self.sigmoid(output_g)
        outputs_g = outputs_.view(-1)
        local_embed = feat_orig

        DGI_loss = self.DGI_model(embeddings, local_embed, adj_tensor, drug_nums)
        if return_embeddings:
            return outputs_, embeddings
        return outputs_g, outputs_l, DGI_loss