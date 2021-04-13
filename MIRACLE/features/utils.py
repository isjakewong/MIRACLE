import csv
import os
import pickle
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")
    - .npz (assumes features are saved with name "features")
    - .npy
    - .csv/.txt (assumes comma-separated features with a header and with one line per molecule)
    - .pkl/.pckl/.pickle containing a sparse numpy array (TODO: remove this option once we are no longer dependent on it)

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class MolFeatureExtractionError(Exception):
    pass


def type_check_num_atoms(mol, num_max_atoms=-1):
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


def construct_atomic_number_array(mol, out_size=-1):
    atom_list = [a.GetAtomicNum() for a in  mol.GetAtoms()]
    n_atoms = len(atom_list)

    if out_size < 0:
        return np.array(atom_list, dtype=np.int32)
    elif out_size >= n_atoms:
        atom_array = np.zeros(out_size, dtype=np.int32)
        atom_array[:n_atoms] = atom_list
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atoms))


def construct_adj_matrix(mol, out_size=-1, self_connections=True):
    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connections:
        adj += np.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(np.float32)
    elif out_size >= 0:
        adj_array = np.zeros((out_size, out_size), dtype=np.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))

    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1):
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))

    adjs = np.zeros((4, size, size), dtype=np.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0

    return adjs