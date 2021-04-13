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
from models.feature_loader import Mol2vecLoader
from models.pooling import *
from data.data import mol2sentence


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network followed by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

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

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        if args.pooling == 'lstm':
            first_linear_dim *= (1 * 2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


FEATURES_FUSIONER_REGISTRY = {}


def register_features_fusioner(features_fusioner_name: str):
    def decorator(features_fusioner):
        FEATURES_FUSIONER_REGISTRY[features_fusioner_name] = features_fusioner
        return features_fusioner

    return decorator


def get_features_fusioner(features_fusioner_name):
    if features_fusioner_name not in FEATURES_FUSIONER_REGISTRY:
        raise ValueError(f'Features fusioner "{features_fusioner_name}" could not be found. '
                         f'If this fusioner relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_FUSIONER_REGISTRY[features_fusioner_name]


def get_available_features_fusioners():
    """Returns the names of available features generators."""
    return list(FEATURES_FUSIONER_REGISTRY.keys())


@register_features_fusioner('nn')
class NN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(NN, self).__init__()
        self.feat_size = feat_size
        self.output_size = output_size
        self.layer = nn.Linear(feat_size * 2, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2), dim=-1))
        return out


@register_features_fusioner('add_nn')
class AddNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(AddNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 3, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, feat_1 + feat_2), dim=-1))
        return out


@register_features_fusioner('sub_nn')
class SubNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(SubNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 3, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, feat_1 - feat_2), dim=-1))
        return out


@register_features_fusioner('abs_sub_nn')
class AbsSubNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(AbsSubNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 3, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, torch.abs(feat_1 - feat_2)), dim=-1))
        return out


@register_features_fusioner('mul_nn')
class MulNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(MulNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 3, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, feat_1 * feat_2), dim=-1))
        return out


@register_features_fusioner('sub_mul_nn')
class SubMulNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(SubMulNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 4, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, feat_1 - feat_2, feat_1 * feat_2), dim=-1))
        return out


@register_features_fusioner('abs_sub_mul_nn')
class AbsSubMulNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(AbsSubMulNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Linear(feat_size * 4, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        out = self.layer(torch.cat((feat_1, feat_2, torch.abs(feat_1 - feat_2), feat_1 * feat_2), dim=-1))
        return out


@register_features_fusioner('bilinear')
class BilinearNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(BilinearNN, self).__init__()
        self.feat_size = feat_size
        self.layer = nn.Bilinear(feat_size, feat_size, output_size)

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
        out = self.layer(feat_1, feat_2)
        return out


@register_features_fusioner('hole')
class HoleNN(nn.Module):
    def __init__(self, feat_size: int, output_size: int):
        super(HoleNN, self).__init__()
        self.feat_size = feat_size
        self.layer = None

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
        return self.circular_correlation(feat_1, feat_2)

    def circular_correlation(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
        """
        :param feat_1: (batch_size, feat_size)
        :param feat_2: (batch_size, feat_size)
        :return: (batch_size, feat_size)
        Computes the circular correlation of two vectors a and b via their fast fourier transforms
        In python code, ifft(np.conj(fft(a)) * fft(b)).real
        (a - j * b) * (c + j * d) = (ac + bd) + j * (ad - bc)
        """
        # (batch_size, feat_size, 2)
        feat_1_fft = torch.rfft(feat_1, 1, normalized=True, onesided=False)
        feat_2_fft = torch.rfft(feat_2, 1, normalized=True, onesided=False)
        prod_fft = torch.zeros_like(feat_1_fft)
        prod_fft[:, :, 0] = feat_1_fft[:, :, 0] * feat_2_fft[:, :, 0] + feat_1_fft[:, :, 1] * feat_2_fft[:, :, 1]
        prod_fft[:, :, 1] = feat_1_fft[:, :, 0] * feat_2_fft[:, :, 1] - feat_1_fft[:, :, 1] * feat_2_fft[:, :, 0]
        cc = torch.irfft(prod_fft, 1, normalized=True, onesided=False)
        return cc


# @register_features_fusioner('mlb_naive')
# class NaiveMLBNN(nn.Module):
#     def __init__(self, feat_size, output_size, rank_size=):
#         super(NaiveMLBNN, self).__init__()
#         self.feat_size = feat_size
#         self.output_size = output_size
#         self.fir_layer = nn.Linear()


class DDIModel(nn.Module):

    def __init__(self, classification: bool, multiclass: bool, multilabel: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(DDIModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.multilabel = multilabel
        if self.multilabel:
            self.sigmoid = nn.Sigmoid()
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        if not args.with_pair:
            # self.encoder = SmilesNN(args) if args.smiles_based else MPN(args)

            if not args.smiles_based:
                if args.graph_encoder == 'ggnn':
                    self.encoder = GGNN(args)
                else:
                    self.encoder = MPN(args)
            else:
                self.encoder = SmilesNN(args)
        else:
            self.encoder = PairMPN(args)

        if args.jt:
            # self.encoder = JTNNEncoder(vocab, args.hidden_size)
            # self.encoder = JunctionTreeGraphNN(args)
            self.encoder = JTNNEncoder(vocab, args.hidden_size) if args.jt_encoder == 'tree' else \
                JunctionTreeGraphNN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        self.multilabel = args.dataset_type == 'multilabel'
        if self.multilabel:
            self.num_labels = args.num_labels
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        if args.smiles_based and args.pooling == 'lstm':
            first_linear_dim = 2 * first_linear_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        fusioner = FEATURES_FUSIONER_REGISTRY[args.fusioner]

        # Create FFN layers
        if args.ffn_num_layers == 1:
            # ffn = [
            #     # dropout,
            #     fusioner(first_linear_dim, args.output_size)
            # ]
            self.fusion_ffn = fusioner(first_linear_dim, args.output_size)
            self.ffn = None
        else:
            # ffn = [
            #     # dropout,
            #     fusioner(first_linear_dim, args.ffn_hidden_size)
            # ]
            self.fusion_ffn = fusioner(first_linear_dim, args.ffn_hidden_size)
            ffn = []
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])
            # Create FFN model
            self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """

        # INSERT our modification code
        batch, features_pair_batch = input
        smiles_1_batch, smiles_2_batch = zip(*batch)
        if features_pair_batch is not None:
            features_1_batch, features_2_batch = zip(*features_pair_batch)
        else:
            features_1_batch, features_2_batch = None, None

        encoder_name = self.encoder.__class__.__name__
        if not encoder_name.startswith('Pair'):
            # we also use a more general class PairEnocder
            feat_1 = self.encoder(smiles_1_batch, features_1_batch)
            feat_2 = self.encoder(smiles_2_batch, features_2_batch)
        else:
            feat_1, feat_2 = self.encoder(smiles_1_batch, smiles_2_batch, features_1_batch, features_2_batch)

        # output = self.ffn(self.encoder(*input))
        # output = self.ffn(torch.cat([feat_1, feat_2], dim=-1))
        feat_1 = self.dropout(feat_1)
        feat_2 = self.dropout(feat_2)
        fused_feat = self.fusion_ffn(feat_1, feat_2)
        if self.ffn is not None:
            output = self.ffn(fused_feat)
        else:
            output = fused_feat
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        if self.multilabel:
            output = self.sigmoid(output)
        return output


# add other types of downstream models such as SVM, Random Forest, GBDT
def build_model(args: Namespace, ddi:bool = False) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes
    if args.dataset_type == 'multilabel':
        args.output_size = args.num_labels

    if not ddi:
        model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    else:
        model = DDIModel(classification=args.dataset_type == 'classification',
                         multiclass=args.dataset_type == 'multiclass',
                         multilabel=args.dataset_type == 'multilabel')
    if args.jt and args.jt_vocab_file is not None:
        vocab = [x.strip("\r\n ") for x in open(args.jt_vocab_file, 'r')]
        vocab = Vocab(vocab)
    else:
        vocab = None
    model.create_encoder(args, vocab=vocab)
    model.create_ffn(args)

    initialize_weights(model)
    return model


class SklearnModel(object):
    def __init__(self, args, pretrain_method, model_name):
        self.args = args
        self.pretrain_method = pretrain_method
        self.model_name = model_name

        self.loader = self.get_loader()

        # pooling and vocab
        self.pooling = self.get_pooling()
        self.vocab = None

        # model
        self.model = self.get_downstream_model()

    def fit(self,
            smiles_list: List[str],
            target_list: List[List[float]]):
        features = self.get_features(smiles_list)
        features = features.detach().cpu().numpy()
        targets = np.array(target_list)
        self.model.fit(features, targets)

    def predict(self,
                smiles_list: List[str]):
        features = self.get_features(smiles_list)
        features = features.detach().cpu().numpy()
        targets = self.model.predict(features)
        targets = np.expand_dims(targets, axis=1)
        return targets

    def get_features(self, smiles_batch: List[str],
                     features_batch: List[np.ndarray] = None):
        args = self.args
        assert self.vocab is None and self.pooling is not None
        emb_batch, length_batch = self.loader(smiles_batch)
        if emb_batch.dim() > 2:
            out = self.pooling.forward(emb_batch, length_batch, features_batch)
        else:
            out = emb_batch
        if torch.cuda.is_available():
            out = out.cuda()
        return out

    def get_loader(self):
        args = self.args
        if self.pretrain_method == 'mol2vec':
            return Mol2vecLoader(embed_dim=args.emb_size)
        else:
            raise ValueError('No such pretrain loader named {}'.format(args.pretrain))

    def get_pooling(self):
        args = self.args
        if args.pooling == 'sum':
            return SumPooling(args)
        elif args.pooling == 'max':
            return MaxPooling(args)
        elif args.pooling == 'lstm':
            return LSTMPooling(args, emb_size=args.emb_size,
                               hidden_size=args.hidden_size,
                               depth=args.depth,
                               bidirectional=True,
                               dropout=args.dropout)
        else:
            raise ValueError('No such encoder named {}'.format(args.pooling))

    def get_downstream_model(self):
        args = self.args
        if self.model_name == 'svm':
            model = SVC() if args.dataset_type == 'classification' else SVR()
        else:
            raise ValueError('No such model named {}'.format(self.model_name))
        return model

    def save_model(self, path):
        joblib.dump(self.model, path)



