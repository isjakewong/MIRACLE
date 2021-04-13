from typing import List, Tuple, Dict, Union
import rdkit
from rdkit import Chem
import copy
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, \
    decode_stereo
from features.featurization import atom_features, bond_features, get_atom_fdim, get_bond_fdim

# Memoization
TREE_TO_GRAPH = {}

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    UNK_ID = 0

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        vmap = {'UNK': self.UNK_ID}
        self.vmap = {x: i + len(vmap) for i, x in enumerate(self.vocab)}
        self.vmap.update(vmap)
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        return self.vmap.get(smiles, self.UNK_ID)

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vmap)


class RDKitBond(object):
    def __init__(self, bond_name):
        super(RDKitBond, self).__init__()
        self.bond_name = bond_name


class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique]  # copy
        self.neighbors = []
        self.neighbor_bonds = []

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def add_neighbor_bond(self, nei_node, entire_mol):
        bonds = []
        for self_atom_idx in self.clique:
            for nei_atom_idx in nei_node.clique:
                try:
                    # 这里的mol有错，只是部分的，我们应当传入整体的mol
                    # try to get bond
                    bond = entire_mol.GetBondBetweenAtoms(self_atom_idx, nei_atom_idx)
                    if bond is not None:
                        bonds.append(bond)
                except:
                    continue

        if len(bonds) == 0:
            return RDKitBond('virtual bond')
        elif len(bonds) >= 1:
            self.neighbor_bonds.append(bonds[0])
            return bonds[0]

    def recover(self, original_mol):
        # TODO: What's the meaning of recover function?
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

    def node_features(self, atoms):
        total_features = []
        for atom_idx in self.clique:
            atom = atoms[atom_idx]
            atom_wise_features = atom_features(atom)
            total_features.append(atom_wise_features)
        total_features = np.array(total_features)
        avg_features = np.mean(total_features, axis=0)
        return avg_features


class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        self.node_pair2bond = {}

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        self.n_edges = 0
        self.n_virtual_edges = 0
        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
            xy_bond = self.nodes[x].add_neighbor_bond(self.nodes[y], self.mol)
            yx_bond = self.nodes[y].add_neighbor_bond(self.nodes[x], self.mol)
            self.node_pair2bond[(x, y)] = xy_bond
            self.node_pair2bond[(y, x)] = yx_bond
            if isinstance(xy_bond, RDKitBond) or isinstance(yx_bond, RDKitBond):
                self.n_virtual_edges += 1
            self.n_edges += 1

        # change
        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def num_of_edges(self):
        return self.n_edges

    def num_of_virtual_edges(self):
        return self.n_virtual_edges

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

    def has_root(self):
        return True if len(self.nodes) > 0 else False

    def get_nodes(self):
        return self.nodes

    def get_bond_between_node_pair(self, node_idx_1, node_idx_2):
        try:
            return self.node_pair2bond.get((node_idx_1, node_idx_2), None)
        except AttributeError:
            temp = 1


class MolGraph(object):
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, moltree: MolTree, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.moltree = moltree
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = self.moltree.size()

        # Get atom features
        for i, atom in enumerate(moltree.get_nodes()):
            atoms = moltree.mol.GetAtoms()
            self.f_atoms.append(atom.node_features(atoms))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                # bond = mol.GetBondBetweenAtoms(a1, a2)
                bond = moltree.get_bond_between_node_pair(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                if args.atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph(object):
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, moltree_graphs: List[MolGraph], args: Namespace):
        self.n_mols = len(moltree_graphs)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in moltree_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def moltree2graph(moltree_batch: List[MolTree],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for moltree in moltree_batch:
        if moltree in TREE_TO_GRAPH:
            mol_graph = TREE_TO_GRAPH[moltree]
        else:
            mol_graph = MolGraph(moltree, args)
            if not args.no_cache:
                TREE_TO_GRAPH[moltree] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--num_data', type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    cset = set()
    lines = open(args.input_file, 'r').readlines()
    for i, line in enumerate(lines[:args.num_data]):
        smiles = line.split()[0]
        print("processing smiles {}".format(smiles))
        mol = MolTree(smiles)
        for c in mol.nodes:
            cset.add(c.smiles)
    for x in cset:
        print(x)
    print('total number: {}'.format(len(cset)))
