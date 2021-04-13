import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace
from typing import Union, List
from features.featurization import BatchMolGraph, mol2graph


class GGNNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, mol_fdim: int):
        super(GGNNEncoder, self).__init__()
        # predefined for GGNNEncoder
        self.num_atom_types = 117
        self.num_edge_types = 4
        num_layers = args.depth
        weight_tying = False
        concat_hidden = False
        num_message_layers = 1 if weight_tying else num_layers
        num_readout_layers = num_layers if concat_hidden else 1
        self.embed = nn.Embedding(self.num_atom_types, atom_fdim)
        self.message_layers = nn.ModuleList([nn.Linear(atom_fdim, self.num_edge_types * atom_fdim) for _ in range(num_message_layers)])
        self.update_layer = nn.GRUCell(atom_fdim, atom_fdim)
        self.i_layers = nn.ModuleList([nn.Linear(2 * atom_fdim, mol_fdim) for _ in range(num_readout_layers)])
        self.j_layers = nn.ModuleList([nn.Linear(atom_fdim, mol_fdim) for _ in range(num_readout_layers)])

        self.args = args
        self.atom_fdim = atom_fdim
        self.mol_fdim = mol_fdim
        self.num_layers = num_layers
        self.atom_fdim = atom_fdim
        self.weight_tying = weight_tying
        self.concat_hidden = concat_hidden
        self.use_input_features = args.use_input_features

    def forward(self, mol_graph: BatchMolGraph, features_batch: List[np.ndarray] = None):
        args = self.args
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        atom_array, adj = mol_graph.get_ggnn_features()

        if args.cuda:
            atom_array = atom_array.cuda()
            adj = adj.cuda()

        # self.update_layer.reset_parameters()

        # embedding layer
        h = self.embed(atom_array)
        h0 = h.clone()

        # message passing
        for step in range(self.num_layers):
            h = self.update(h, adj, step)

        g = self.readout(h, h0, step=0)
        return g

    def update(self, h: torch.Tensor, adj: torch.Tensor, step: int = 0):
        mb, atom, _ = h.shape
        message_layer_index = 0 if self.weight_tying else step
        # h: (mb, atom, atom_fdim) -> (mb, atom, num_edge_types * atom_fdim)
        # m: (mb, atom, num_edge_types, atom_fdim)
        m = self.message_layers[message_layer_index](h).view(
            mb, atom, self.num_edge_types, self.atom_fdim
        )
        # m: (mb, num_edge_types, atom, atom_fdim)
        m = m.permute(0, 2, 1, 3)
        m = m.contiguous()
        # m: (mb * num_edge_types, atom, atom_fdim)
        m = m.view(mb * self.num_edge_types, atom, self.atom_fdim)

        # adj: (mb * num_edge_types, atom, atom)
        adj = adj.view(mb * self.num_edge_types, atom, atom)

        # m: (mb * num_edge_types, atom, atom_fdim)
        m = torch.bmm(adj, m)
        # m: (mb, num_edge_types, atom, atom_fdim)
        m = m.view(mb, self.num_edge_types, atom, self.atom_fdim)
        # m: (mb, atom, atom_fdim)
        m = torch.sum(m, dim=1)

        # update via GRUCell
        # m: (mb * atom, atom_fdim)
        m = m.view(mb * atom, -1)
        # h: (mb * atom, atom_fdim)
        h = h.view(mb * atom, -1)
        # out_h: (mb * atom, atom_fdim)
        out_h = self.update_layer(m, h if step > 0 else None)
        out_h = out_h.view(mb, atom, self.atom_fdim)
        return out_h

    def readout(self, h, h0, step):
        index = step if self.concat_hidden else 0
        return torch.sum(torch.sigmoid(self.i_layers[index](torch.cat([h, h0], dim=2))) * self.j_layers[index](h), dim=1)


class GGNN(nn.Module):
    def __init__(self, args: Namespace, graph_input: bool = False):
        super(GGNN, self).__init__()
        self.args = args
        self.graph_input = graph_input
        self.encoder = GGNNEncoder(args, atom_fdim=args.hidden_size, mol_fdim=args.hidden_size)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> Union[torch.FloatTensor, torch.Tensor]:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output
