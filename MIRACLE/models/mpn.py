from argparse import Namespace
from typing import List, Union, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

from features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from nn_utils import index_select_ND, get_activation_function
from model_utils import convert_to_2D, convert_to_3D, compute_max_atoms
from .alignment import Alignment


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.weight_tying = self.args.weight_tying
        n_message_layer = 1 if self.weight_tying else self.depth - 1
        # self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_h = nn.ModuleList([nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
                                  for _ in range(n_message_layer)])

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # TODO: parameters for attention
        self.attn_num_d = self.args.attn_num_d
        self.attn_num_r = self.args.attn_num_r
        self.W_s1 = Parameter(torch.FloatTensor(self.hidden_size, self.attn_num_d))
        self.W_s2 = Parameter(torch.FloatTensor(self.attn_num_d, self.attn_num_r))
        self.softmax = nn.Softmax(dim=1)

        self.i_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.j_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> Union[torch.FloatTensor, torch.Tensor]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            # message = self.W_h(message)
            # shared or not shared
            step = 0 if self.weight_tying else depth
            message = self.W_h[step](message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.args.attn_output:
            mol_vecs = self.attention(atom_hiddens, a_scope)
            return mol_vecs
        # print(mol_vecs.size(), mol_vecs)
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs_ = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)
            return mol_vecs_, mol_vecs
        else:
            return mol_vecs  # num_molecules x hidden

    # TODO: add self-attention layer like https://arxiv.org/abs/1904.05003
    # TODO: add option 'attn_output', 'attn_num_d' and 'attn_num_r' in args: Namespace
    def attention(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        """
        :param atom_hiddens: (num_atoms, hidden_size)
        :param a_scope: list of tuple (int, int)
        :return: (num_atoms, hidden_size * attn_num_r)
        """
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        max_atoms = compute_max_atoms(a_scope)
        # batch_hidden: (batch_size, max_atoms, hidden_size)
        # batch_mask: (batch_size, max_atoms, max_atoms)
        batch_hidden, batch_mask = convert_to_3D(atom_hiddens, a_scope, max_atoms, device=device, self_attn=True)
        batch_size = batch_hidden.size(0)

        # # https://arxiv.org/abs/1904.05003
        # # W_s1: (batch_size, hidden_size, attn_num_d)
        # W_s1 = self.W_s1.unsqueeze(dim=0).repeat([batch_size, 1, 1])
        # # W_s2: (batch_size, attn_num_d, attn_num_r)
        # W_s2 = self.W_s2.unsqueeze(dim=0).repeat([batch_size, 1, 1])
        # # s: (batch_size, max_atoms, attn_num_r)
        # # s = torch.mm(torch.tanh(torch.mm(batch_hidden, W_s1)), W_s2)
        # s = self.softmax(torch.matmul(torch.tanh(torch.matmul(batch_hidden, W_s1)), W_s2))
        # # e: (batch_size, attn_num_r, hidden_size)
        # # e = s.permute((0, 2, 1)) * batch_hidden
        # e = torch.matmul(s.permute((0, 2, 1)), batch_hidden)
        # # this is the final graph representation
        # e = e.view([batch_size, -1])
        # # print(e.size)
        # self-contained attention mechanism like GGNN
        e = torch.sum(torch.sigmoid(self.j_layer(batch_hidden)) * self.i_layer(batch_hidden), dim=1)
        return e


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
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


class Mixture(nn.Module):
    def __init__(self, feat_size, output_size):
        super(Mixture, self).__init__()
        self.feat_size = feat_size
        self.output_size = output_size
        ffn = [
            nn.Linear(feat_size * 2, output_size),
            nn.ReLU(),
        ]
        self.ffn = nn.Sequential(*ffn)

    def forward(self, feat_1, feat_2):
        if torch.cuda.is_available():
            feat_1, feat_2 = feat_1.cuda(), feat_2.cuda()
        return self.ffn(torch.cat((feat_1, feat_2), dim=-1))


class PairMPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(PairMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # alignment
        if self.args.align:
            self.align = nn.ModuleList([Alignment(args) for _ in range(self.depth - 1)])
            self.mix = nn.ModuleList([Mixture(self.hidden_size, self.hidden_size) for _ in range(self.depth - 1)])

    def forward(self,
                mol_graph: BatchMolGraph,
                ano_mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                ano_features_batch: List[np.ndarray] = None) -> [torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            ano_features_batch = torch.from_numpy(np.stack(ano_features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()
                ano_features_batch = ano_features_batch.cuda()

            if self.features_only:
                return features_batch, ano_features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        ano_f_atoms, ano_f_bonds, ano_a2b, ano_b2a, ano_b2revb, ano_a_scope, ano_b_scope = ano_mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()
            ano_a2a = ano_mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()
            ano_f_atoms, ano_f_bonds, ano_a2b, ano_b2a, ano_b2revb = ano_f_atoms.cuda(), ano_f_bonds.cuda(), ano_a2b.cuda(), ano_b2a.cuda(), ano_b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()
                ano_a2a = ano_a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
            ano_input = self.W_i(ano_f_atoms)
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
            ano_input = self.W_i(ano_f_bonds)
        message = self.act_func(input)  # num_bonds x hidden_size
        ano_message = self.act_func(ano_input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2
                ano_message = (ano_message + ano_message[ano_b2revb]) / 2

            # messaging process
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim

                ano_nei_a_message = index_select_ND(ano_message, ano_a2a)
                ano_nei_f_bonds = index_select_ND(f_bonds, ano_a2a)
                ano_nei_message = torch.cat((ano_nei_a_message, ano_nei_f_bonds), dim=2)
                ano_message = ano_nei_message.sum(dim=1)
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                # neighboring incoming bonds' features
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                # summing the incoming bond features to get the atom representation
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

                ano_nei_a_message = index_select_ND(ano_message, ano_a2b)  # num_atoms x max_num_bonds x hidden
                ano_a_message = ano_nei_a_message.sum(dim=1)  # num_atoms x hidden
                ano_rev_message = ano_message[ano_b2revb]  # num_bonds x hidden
                ano_message = ano_a_message[ano_b2a] - ano_rev_message  # num_bonds x hidden

            # update process
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

            ano_message = self.W_h(ano_message)
            ano_message = self.act_func(ano_input + ano_message)  # num_bonds x hidden_size
            ano_message = self.dropout_layer(ano_message)  # num_bonds x hidden

            # align
            if self.args.align:
                # (batch_size, max_seq_len, hidden_size)
                bt_message = self.batching(b_scope, message)
                bt_ano_message = self.batching(ano_b_scope, ano_message)
                zero_tensor = torch.zeros_like(bt_message)
                mask = torch.ne(bt_message, zero_tensor)
                ano_zero_tensor = torch.zeros_like(bt_ano_message)
                ano_mask = torch.ne(bt_ano_message, ano_zero_tensor)
                bt_align_message, bt_align_ano_message = self.align[depth](bt_message, bt_ano_message, mask, ano_mask)
                align_message = self.reverse_batching(b_scope, bt_align_message)
                align_ano_message = self.reverse_batching(ano_b_scope, bt_align_ano_message)
                message = self.mix[depth](message, align_message)
                ano_message = self.mix[depth](ano_message, align_ano_message)
                temp = 1

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        ano_a2x = ano_a2a if self.atom_messages else ano_a2b
        ano_nei_a_message = index_select_ND(ano_message, ano_a2x)  # num_atoms x max_num_bonds x hidden
        ano_a_message = ano_nei_a_message.sum(dim=1)  # num_atoms x hidden
        ano_a_input = torch.cat([ano_f_atoms, ano_a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        ano_atom_hiddens = self.act_func(self.W_o(ano_a_input))  # num_atoms x hidden
        ano_atom_hiddens = self.dropout_layer(ano_atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        # Readout
        ano_mol_vecs = []
        for i, (a_start, a_size) in enumerate(ano_a_scope):
            if a_size == 0:
                ano_mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = ano_atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                ano_mol_vecs.append(mol_vec)

        ano_mol_vecs = torch.stack(ano_mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_input_features:
            ano_features_batch = ano_features_batch.to(ano_mol_vecs)
            if len(ano_features_batch.shape) == 1:
                ano_features_batch = ano_features_batch.view([1, ano_features_batch.shape[0]])
            ano_mol_vecs = torch.cat([ano_mol_vecs, ano_features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs, ano_mol_vecs  # num_molecules x hidden

    def batching(self, scope, message):
        """
        :param scope:
        :param message: (num_bonds, hidden)
        :return:
        """
        # Readout
        mol_messages = []
        max_num_bonds = 0
        batch_size = len(scope)
        hidden_size = message.size(1)
        num_bonds_list = []
        # for every molecule
        for i, (start, size) in enumerate(scope):
            if size == 0:
                # mol_messages.append(self.cached_zero_vector)
                continue
            else:
                # (atom_num_bonds, hidden)
                cur_hiddens = message.narrow(0, start, size)
                num_bonds = cur_hiddens.size(0)
                num_bonds_list.append(num_bonds)
                if num_bonds > max_num_bonds:
                    max_num_bonds = num_bonds
                mol_message = cur_hiddens
                mol_messages.append(mol_message)

        mol_messages_tensor = torch.zeros(batch_size, max_num_bonds, hidden_size)
        for i in range(batch_size):
            mol_messages_tensor[i, :num_bonds_list[i], :] = mol_messages[i]
        # mol_messages = torch.stack(mol_messages, dim=0)  # (num_molecules, hidden_size)
        if torch.cuda.is_available():
            mol_messages_tensor = mol_messages_tensor.cuda()
        return mol_messages_tensor

    def reverse_batching(self, scope, mol_messages):
        """
        :param scope:
        :param mol_messages: (batch_size, max_num_bonds, hidden)
        :return:
        """
        if torch.cuda.is_available():
            mol_messages = mol_messages.cuda()
        messages = []
        messages.append(self.cached_zero_vector.data)
        for i, (start, size) in enumerate(scope):
            num_bonds = size
            # (num_bonds, hidden)
            message = mol_messages[i, :num_bonds, :]
            message = torch.split(message, 1, dim=0)
            message = [torch.squeeze(m, dim=0) for m in message]
            # list of (hidden, )
            messages.extend(message)
        messages = torch.stack(messages, dim=0)
        return messages


class PairMPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(PairMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        # self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        self.encoder = PairMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                ano_batch,
                features_batch = None,
                ano_features_batch: List[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)
            ano_batch = mol2graph(ano_batch, self.args)

        # output = self.encoder.forward(batch, features_batch)
        # ano_output = self.encoder.forward(ano_batch, ano_features_batch)
        output, ano_output = self.encoder(batch, ano_batch, features_batch, ano_features_batch)

        return output, ano_output