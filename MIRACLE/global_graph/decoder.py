import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        rec = torch.mm(inputs_row, inputs_col)
        outputs = self.act(rec)
        return outputs


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0,
                 act = lambda x: x):
        super(BilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.relation.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        intermediate_product = torch.mm(inputs_row, self.relation)
        rec = torch.mm(intermediate_product, inputs_col)
        outputs = self.act(rec)
        return outputs


class NNBilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0,
                 act = lambda x: x):
        super(NNBilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        ffn = [
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        ]
        self.ffn = nn.Sequential(*ffn)
        self.reset_parameter()

    def reset_parameter(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() == 0:
                nn.init.constant_(param, 1.)
            else:
                nn.init.xavier_normal_(param)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs
        inputs_row = self.ffn(self.dropout(inputs_row))
        inputs_col = self.ffn(self.dropout(inputs_col)).transpose(0, 1)
        intermediate_product = torch.mm(inputs_row, self.relation)
        rec = torch.mm(intermediate_product, inputs_col)
        outputs = self.act(rec)
        return outputs


class DistMultDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0,
                 act = lambda x: x):
        super(DistMultDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.tmp = Parameter(torch.FloatTensor(input_dim, 1))
        self.reset_parameter()
        self.relation = self.tmp.view(-1)

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.tmp.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        relation = torch.diag(self.relation)
        relation = relation.cuda()
        intermediate_product = torch.mm(inputs_row, relation)
        rec = torch.mm(intermediate_product, inputs_col)
        outputs = self.act(rec)
        return outputs


class CosDecoder(InnerProductDecoder):
    def __init__(self, input_dim: int, dropout: float = 0.0,
                 act = lambda x: x):
        super(CosDecoder, self).__init__(input_dim, dropout, act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        normed_x1 = inputs_row / torch.norm(inputs_row, dim=1, keepdim=True)
        normed_x2 = inputs_col / torch.norm(inputs_col, dim=0, keepdim=True)
        outputs = torch.mm(normed_x1, normed_x2)
        outputs = self.act(outputs)
        outputs = outputs.view(-1)
        return outputs
