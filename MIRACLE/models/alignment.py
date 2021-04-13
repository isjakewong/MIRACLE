from typing import Tuple
from argparse import Namespace
import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class Alignment(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)), requires_grad=True)

    def _attention(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        :param a: (batch_size, num_len_a, hidden_size)
        :param b: (batch_size, num_len_b, hidden_size)
        :return: (batch_size, num_len_a, num_len_b)
        """
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                mask_a: torch.Tensor, mask_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param a: (batch_size, num_len_a, hidden_size)
        :param b: (batch_size, num_len_b, hidden_size)
        :param mask_a: (batch_size, num_len_a, hidden_size)
        :param mask_b: (batch_size, num_len_b, hidden_size)
        :return:
        """
        # (batch_size, num_len_a, num_len_b)
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        mask = mask.bool()
        attn.masked_fill_(~mask, -1e7)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b


