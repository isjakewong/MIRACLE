
from argparse import Namespace
from typing import List, Union, Tuple
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from data.data import mol2sentence

MAPPING = None
POOLING_REGISTRY = {}


def register_pooling(pooling_name: str):
    def decorator(pooling_obj):
        POOLING_REGISTRY[pooling_name] = pooling_obj
        return pooling_obj

    return decorator


def get_pooling(pooling_name):
    if pooling_name not in POOLING_REGISTRY:
        raise ValueError(f'Pooling "{pooling_name}" could not be found. '
                         f'If this pooling relies on rdkit features, you may need to install descriptastorus.')

    return POOLING_REGISTRY[pooling_name]


def get_available_poolings():
    """Returns the names of available features generators."""
    return list(POOLING_REGISTRY.keys())


class Mol2vecLoader(nn.Module):
    def __init__(self, embed_dim: int = None):
        super(Mol2vecLoader, self).__init__()
        self.embed_dim = embed_dim

        # dict
        try:
            mol2vec = word2vec.Word2Vec.load(MOL2VEC_FILEPATH, mmap='r')
        except AttributeError:
            with open(MOL2VEC_FILEPATH, 'rb') as reader:
                mol2vec = pickle.load(reader)
        self.mol2vec = mol2vec
        try:
            mol2vec_embed_dim = mol2vec.wv.word_vec(list(mol2vec.wv.vocab.keys())[0]).shape[0]
        except AttributeError:
            mol2vec_embed_dim = list(mol2vec.values())[0].shape[0]
        self.mol2vec_embed_dim = mol2vec_embed_dim
        if mol2vec_embed_dim != embed_dim:
            ffn = [
                nn.Linear(mol2vec_embed_dim, embed_dim),
                nn.ReLU(),
            ]
            self.ffn = nn.Sequential(*ffn)
        else:
            self.ffn = None

        self.mapping = {}
        self.unk_emb = np.random.uniform(-1.0, 1.0, size=(mol2vec_embed_dim))

    def forward(self, smiles_batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = []
        lengths = []
        max_seq_len = 0
        batch_size = len(smiles_batch)
        for smiles in smiles_batch:
            try_emb = self.mapping.get(smiles, None)
            if try_emb is None:
                # try:
                mol = Chem.MolFromSmiles(smiles)
                sentence = mol2alt_sentence(mol, radius=1)
                emb = []
                for word in sentence:
                    try:
                        try:
                            vec = self.mol2vec.wv.word_vec(word)
                        except AttributeError:
                            vec = self.mol2vec[word]
                    except KeyError:
                        vec = self.unk_emb
                    emb.append(vec)
                # (seq_len, embed_dim)
                emb = np.array(emb, dtype=np.float)
                seq_len = len(sentence)
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
                embs.append(emb)
                lengths.append(seq_len)
            # except:
            # print('Failed smiles {}'.format(smiles))
        # embs: List[np.ndarray]
        emb_data = np.zeros((batch_size, max_seq_len, self.mol2vec_embed_dim), dtype=np.float)
        for emb_no, emb in enumerate(embs):
            emb_data[emb_no, :lengths[emb_no]] = emb
        emb_tensor = torch.Tensor(emb_data)
        length_data = np.array(lengths, dtype=np.int)
        length_tensor = torch.LongTensor(length_data)

        if torch.cuda.is_available():
            emb_tensor = emb_tensor.cuda()
            length_tensor = length_tensor.cuda()

        if self.ffn is not None:
            emb_tensor = self.ffn(emb_tensor)

        return emb_tensor, length_tensor


class SmilesEncoder(nn.Module):
    def __init__(self):
        super(SmilesEncoder, self).__init__()

    def forward(self,
                emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        return None


@register_pooling('sum')
class SumPooling(nn.Module):
    def __init__(self, args: Namespace):
        super(SumPooling, self).__init__()
        self.args = args

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        return torch.sum(emb_batch, dim=1)


@register_pooling('max')
class MaxPooling(nn.Module):
    def __init__(self, args: Namespace):
        super(MaxPooling, self).__init__()
        self.args = args

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        return torch.max(emb_batch, dim=1)[0]


class LSTMPooling(nn.Module):
    def __init__(self, args, emb_size, hidden_size,
                 depth=1, bidirectional=True, dropout=0.0):
        super(LSTMPooling, self).__init__()
        self.args = args
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size,
                            num_layers=depth,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        batch_size = emb_batch.size(0)
        emb_batch = self.dropout(emb_batch)
        if torch.cuda.is_available():
            emb_batch = emb_batch.cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(emb_batch)
        final_hidden_state = final_hidden_state.view(batch_size, -1)
        return torch.mean(output, dim=1)


class SmilesNN(nn.Module):
    def __init__(self, args: Namespace):
        super(SmilesNN, self).__init__()
        self.args = args
        self.loader = self.get_loader()
        self.pooling = self.get_pooling()
        self.vocab = None

        if args.cuda:
            self.loader = self.loader.cuda()
            if self.pooling is not None:
                self.pooling = self.pooling.cuda()

    def forward(self,
                smiles_batch: List[str],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
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
        if args.pretrain == 'mol2vec':
            return Mol2vecLoader(args.emb_size)
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
