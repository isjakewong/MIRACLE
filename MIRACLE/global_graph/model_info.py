import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace

class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)
        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF_local(nn.Module):
    def __init__(self, args:Namespace, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, args.FF_hidden1),
            nn.ReLU(),
            nn.Linear(args.FF_hidden1, args.FF_hidden2),
            nn.ReLU(),
            nn.Linear(args.FF_hidden2, args.FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, args.FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class FF_global(nn.Module):
    def __init__(self, args:Namespace, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, args.FF_hidden1),
            nn.ReLU(),
            nn.Linear(args.FF_hidden1, args.FF_hidden2),
            nn.ReLU(),
            nn.Linear(args.FF_hidden2, args.FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, args.FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class shortcut(nn.Module):
    def __init__(self, input_dim_1, input_dim_2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim_1, input_dim_2),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim_2, input_dim_2)

    def forward(self, x1, x2):
        return self.block(x1) + self.linear_shortcut(x2)