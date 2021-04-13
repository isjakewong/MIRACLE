import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_losses import get_positive_expectation, get_negative_expectation
from argparse import Namespace

def local_global_drug_loss_(args: Namespace, l_enc, g_enc, adj_tensor, num_drugs, measure):
    if args.cuda:
        pos_mask = adj_tensor.cuda() + torch.eye(num_drugs).cuda()
        neg_mask = torch.ones((num_drugs, num_drugs)).cuda() - pos_mask
    else:
        pos_mask = adj_tensor + torch.eye(num_drugs)
        neg_mask = torch.ones((num_drugs, num_drugs)) - pos_mask

    res = torch.mm(l_enc, g_enc.t())
    num_edges = args.num_edges_w + num_drugs
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_edges
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_drugs ** 2 - 2 * num_edges)

    return (E_neg - E_pos)