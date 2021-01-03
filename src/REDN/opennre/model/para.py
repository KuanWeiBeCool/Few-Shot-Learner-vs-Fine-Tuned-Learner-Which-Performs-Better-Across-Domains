# -*- coding: utf-8 -*-
"""framework_modified.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AcFuYE4QxvKX0NVMIvcLTKiUG_oP2DwV
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base_model import SentenceRE


class PARA(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, num_token_labels, subject_1=False, use_cls=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.use_cls = use_cls
        self.subject_1 = subject_1
        self.num_token_labels = num_token_labels
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        hidden_size = self.sentence_encoder.hidden_size

        # self.lstm = nn.LSTM(hidden_size, hidden_size//2, 64, bidirectional=True, batch_first=True)
        # self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

        self.subject_output_fc = nn.Linear(hidden_size, self.num_token_labels)
        # self.bias = Parameter(torch.zeros((num_class, seq_len, seq_len)))

        self.attn_score = MultiHeadAttention(input_size=hidden_size,
                                             output_size=num_class * hidden_size,
                                             num_heads=num_class)

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, token, att_mask):
        bs = token.shape[0]
        sl = token.shape[1]

        rep, hs, atts = self.sentence_encoder(token, att_mask)  # (B, H)
        # print(self.sentence_encoder(token, att_mask))
        # print("hs : {}".format(hs))
        if self.subject_1:
            subject_output = hs[-1]  # BS * SL * HS
        else:
            subject_output = hs[-2]  # BS * SL * HS

        # # Added lstm layer
        # packed_output, (hidden, cell) = self.lstm(subject_output)
        # cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # rel = self.relu(cat)

        subject_output_logits = self.subject_output_fc(subject_output)  # BS * SL * NTL

        if self.use_cls:
            subject_output = subject_output + rep.view(-1, 1, rep.shape[-1])
        score = self.attn_score(hs[-1], subject_output)  # BS * NTL * SL * SL
        score = score.sigmoid()
        return score, subject_output_logits, atts[-1]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = input_size

        self.depth = int(output_size / self.num_heads)

        self.relu = nn.ReLU()
        self.LSTMq = nn.LSTM(input_size, input_size//2, 64, bidirectional=True, batch_first=True)
        self.LSTMk = nn.LSTM(input_size, input_size//2, 64, bidirectional=True, batch_first=True)

        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H
        return x.permute([0, 2, 1, 3])  # BS * NH * SL * H

    def forward(self, k, q):  # BS * SL * HS
        batch_size = q.shape[0]
        # # Add LSTM
        # packed_output_q, (hidden_q, cell_q) = self.LSTMq(q)
        # cat_q = torch.cat((hidden_q[-2, :, :], hidden_q[-1, :, :]), dim=1)
        # lstm_q = self.relu(cat_q)

        q = self.Wq(q)  # BS * SL * OUT
        # # Add LSTM
        # packed_output_k, (hidden_k, cell_k) = self.LSTMq(k)
        # cat_k = torch.cat((hidden_k[-2, :, :], hidden_k[-1, :, :]), dim=1)
        # lstm_k = self.relu(cat_k)

        k = self.Wk(k)  # BS * SL * OUT

        # q = F.dropout(q, 0.8, training=self.training)
        # k = F.dropout(k, 0.8, training=self.training)

        q = self.split_into_heads(q, batch_size)  # BS * NH * SL * H
        k = self.split_into_heads(k, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = attn_score / np.sqrt(k.shape[-1])

        # scaled_attention = output[0].permute([0, 2, 1, 3])
        # attn = output[1]
        # original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        # output = self.dense(original_size_attention)

        return attn_score
