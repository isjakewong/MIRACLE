import torch
import torch.nn as nn
import numpy as np
from collections import deque
from argparse import Namespace
from typing import List, Tuple, Dict
from data.mol_tree import MolTreeNode, MolTree, Vocab

MAX_NB = 8


def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    # h_nei is m_{ki}
    # s_ij = \sum_{k \in N(i)\\ j} (4)
    sum_h = h_nei.sum(dim=1).float()
    # formula (5)
    z_input = torch.cat([x, sum_h], dim=1)
    z = nn.Sigmoid()(W_z(z_input))

    # formula (6)
    r_1 = W_r(x).view(-1, 1, hidden_size)
    h_nei = h_nei.float()
    r_2 = U_r(h_nei)
    # r_{ki}
    r = nn.Sigmoid()(r_1 + r_2)

    # r_{ki} \odot m_{ki}
    gated_h = r * h_nei
    # sum_{k \in N(i)\\j} r_{ki} \doot m_{ki}
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x, sum_gated_h], dim=1)
    pre_h = nn.Tanh()(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h


"""
Helper functions
"""


def get_prop_order(root: MolTreeNode) -> List[List[Tuple[MolTreeNode, MolTreeNode]]]:
    """
    BFS(Breadth-First Search) algorithm
    :param root: MolTreeNode object.
    :return: order, list of tuple (atom_1, atom_2)
    """
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1, order2 = [], []
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                # order1: from root to leaf
                order1[y.depth - 1].append((x, y))
                # order2: from leaf to bottom
                order2[y.depth - 1].append((y, x))
    # order2: from bottom to leaf
    order = order2[::-1] + order1
    return order


def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = torch.zeros(hidden_size, requires_grad=False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
        pad_len = MAX_NB - len(nei)

        # nei.extend([padding] * pad_len)

        if pad_len > 0:
            nei.extend([padding] * pad_len)
        else:
            nei = nei[:MAX_NB]
        assert len(nei) == MAX_NB

        h_nei.extend(nei)

    h_nei = [h.cuda() for h in h_nei]
    h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NB, hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = torch.LongTensor(x_idx)
    if torch.cuda.is_available(): x_vec = x_vec.cuda()
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


# import line_profiler
class JTNNEncoder(nn.Module):

    def __init__(self, vocab: Vocab, hidden_size: int, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    # @profile
    def forward(self, moltree_batch: List[MolTree],
                features_batch: List[np.ndarray] = None) -> Tuple[Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
        # this loop is time-consuming... accounting for 69.5%
        # for mol_tree in moltree_batch:
        #     mol_tree.recover()
        set_batch_nodeID(moltree_batch, self.vocab)
        root_batch = [mol_tree.nodes[0] for mol_tree in moltree_batch]

        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)

        h = {}
        max_depth = max([len(x) for x in orders])
        padding = torch.zeros(self.hidden_size, requires_grad=False)

        # BFS (Breadth-First Search)
        for t in range(max_depth):
            prop_list = []
            for order in orders:
                # ensure that the depth is less than the max depth
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x, node_y in prop_list:
                x, y = node_x.idx, node_y.idx
                # TODO: node_x.wid, word_id
                cur_x.append(node_x.wid)

                # h_nei: (seq_len, hidden_size)
                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z, x)])

                pad_len = MAX_NB - len(h_nei)

                if pad_len > 0:
                    h_nei.extend([padding] * pad_len)
                else:
                    h_nei = h_nei[:MAX_NB]
                assert len(h_nei) == MAX_NB

                # if pad_len > 0:
                #     # h_nei: (seq_len, hidden_size)
                #     if len(h_nei) > 0:
                #         h_nei = torch.stack(h_nei, dim=0)
                #         # h_nei: (MAX_NB, hidden_size)
                #         h_nei = torch.cat((h_nei, torch.zeros([pad_len, self.hidden_size], device=device)), dim=0)
                #     else:
                #         h_nei = torch.zeros([pad_len, self.hidden_size], device=device)
                # else:
                #     h_nei = torch.stack(h_nei, dim=0)
                #     h_nei = h_nei[:MAX_NB]

                cur_h_nei.extend(h_nei)

            cur_x = torch.LongTensor(cur_x)
            if torch.cuda.is_available():
                cur_x = cur_x.cuda()

            cur_x = self.embedding(cur_x)
            # time consuming accounting for 14%
            # (len(prop_list), MAX_NB, hidden_size)
            cur_h_nei = [h.cuda() for h in cur_h_nei]
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)

            if torch.cuda.is_available():
                cur_h_nei = cur_h_nei.cuda()

            # cur_x: one-hot encoding representing the cluster's label type
            # cur_h_nei: message from neighboring atoms
            # time consuming accounting for 5%
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i, m in enumerate(prop_list):
                x, y = m[0].idx, m[1].idx
                h[(x, y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        # return h, root_vecs
        return root_vecs



