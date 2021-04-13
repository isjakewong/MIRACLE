from argparse import Namespace

import torch
import numpy as np
import torch.nn as nn
from sklearn.svm import SVC, SVR
from typing import List
from sklearn.externals import joblib

from .mpn import MPN
from .mpn import PairMPN
from .ggnn import GGNN
from .smiles import SmilesNN
from .jtnn_enc import JTNNEncoder
from .jt_mpn import JunctionTreeGraphNN
from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights
from models.feature_loader import Mol2vecLoader, ElmoLoader, BertLoader
from models.pooling import *
from data.data import mol2sentence
from pretrain.bert.dataset import WordVocab

class DDIModel(nn.Module):

    def __init__(self):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(DDIModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        if not args.smiles_based:
            if args.graph_encoder == 'ggnn':
                self.encoder = GGNN(args)
            else:
                self.encoder = MPN(args)
        else:
            self.encoder = SmilesNN(args)

        if args.jt:
            # self.encoder = JTNNEncoder(vocab, args.hidden_size)
            # self.encoder = JunctionTreeGraphNN(args)
            self.encoder = JTNNEncoder(vocab, args.hidden_size) if args.jt_encoder == 'tree' else \
                JunctionTreeGraphNN(args)

    def forward(self, input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """

        smiles_batch = input 
        features_batch = None
        encoder_name = self.encoder.__class__.__name__
        feat = self.encoder(smiles_batch, features_batch)
        feat = self.dropout(feat)
        output = self.sigmoid(feat)
        return output