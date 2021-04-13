
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace
from typing import List, Tuple
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from data.data import mol2sentence

class Mol2vecLoader(nn.Module):
    def __init__(self, embed_dim: int = None):
        super(Mol2vecLoader, self).__init__()
        self.embed_dim = embed_dim

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