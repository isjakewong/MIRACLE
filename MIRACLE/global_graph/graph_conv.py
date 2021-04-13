import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, inputs: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(SparseGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttnGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 dropout: float = 0.3, alpha: float = 0.2, act = F.elu):
        super(AttnGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.act = act

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param input: (num_nodes, in_features)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        h = torch.mm(input, self.W)
        if self.bias is not None:
            h = h + self.bias
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        adj_at = adj.to_dense()
        attention = torch.where(adj_at > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
