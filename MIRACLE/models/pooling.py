import torch
import torch.nn as nn
from argparse import Namespace
from typing import List
import numpy as np


class SumPooling(nn.Module):
    def __init__(self, args: Namespace):
        super(SumPooling, self).__init__()
        self.args = args

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        return torch.sum(emb_batch, dim=1)


class MaxPooling(nn.Module):
    def __init__(self, args: Namespace):
        super(MaxPooling, self).__init__()
        self.args = args

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        return torch.max(emb_batch, dim=1)[0]


class LSTMPooling(nn.Module):
    def __init__(self, args, emb_size, hidden_size,
                 depth=1, bidirectional=True, dropout=0.0):
        super(LSTMPooling, self).__init__()
        self.args = args
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size,
                            num_layers=depth,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        # output_size = (2 if bidirectional else 1) * depth * hidden_size
        # if output_size != hidden_size:
        #     ffn = [
        #         nn.Linear(output_size, hidden_size),
        #         nn.ReLU(),
        #     ]
        #     self.ffn = nn.Sequential(*ffn)
        # else:
        #     self.ffn = None

    def forward(self, emb_batch: torch.FloatTensor,
                length_batch: torch.LongTensor,
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        :param emb_batch: (batch_size, seq_len, embed_size)
        :param length_batch: (batch_size, )
        :param features_batch: (batch_size, feat_size)
        :return:
        """
        batch_size = emb_batch.size(0)
        emb_batch = self.dropout(emb_batch)
        if torch.cuda.is_available():
            emb_batch = emb_batch.cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(emb_batch)
        # (batch_size, depth * bidirectional * hidden_size)
        final_hidden_state = final_hidden_state.view(batch_size, -1)
        # if self.ffn is not None:
        #     final_hidden_state = self.ffn(final_hidden_state)
        # return final_hidden_state
        return torch.mean(output, dim=1)


