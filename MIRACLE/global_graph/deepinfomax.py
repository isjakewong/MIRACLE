import os.path as osp
import torch

import torch.nn as nn
import torch.nn.functional as F
import json
from argparse import Namespace

from .losses_info import local_global_drug_loss_
from .model_info import PriorDiscriminator, FF_local, FF_global

class GcnInfomax(nn.Module):
  def __init__(self, args: Namespace, gamma=.1):
    super(GcnInfomax, self).__init__()
    self.args = args
    self.gamma = gamma
    self.prior = args.prior
    self.features_dim = args.hidden_size
    self.embedding_dim = args.gcn_hidden3
    self.local_d = FF_local(args, self.features_dim)
    self.global_d = FF_global(args, self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)


  def forward(self, embeddings, features, adj_tensor, num_drugs):
    
    g_enc = self.global_d(embeddings)
    l_enc = self.local_d(features)
    measure='JSD' # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    local_global_loss = local_global_drug_loss_(self.args, l_enc, g_enc, adj_tensor, num_drugs, measure)
    eps = 1e-5
    if self.prior:
        prior = torch.rand_like(embeddings)
        term_a = torch.log(self.prior_d(prior) + eps).mean()
        term_b = torch.log(1.0 - self.prior_d(embeddings) + eps).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR
