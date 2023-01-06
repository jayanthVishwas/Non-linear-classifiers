# author - Jayanth

import torch
from torch import nn


def getWordPosArc(x):
    word_ind = [0, 1, 2, 6, 7, 8, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    pos_ind = [3, 4, 5, 9, 10, 11, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]
    arc_ind = [14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47]

    return x[:, word_ind], x[:, pos_ind], x[:, arc_ind]


class ArcEagerParser(nn.Module):
    def __init__(self, hidden_units, words_size, pos_size, arc_size, trans_size):
        super().__init__()

        self.word_emb = nn.Embedding(words_size, 50)
        self.pos_emb = nn.Embedding(pos_size, 50)
        self.arc_emb = nn.Embedding(arc_size, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.hidden1 = nn.Linear(2400, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, 500)
        self.out_trans = nn.Linear(500, trans_size)
        self.out_arcs = nn.Linear(500, arc_size)

    def forward(self, x):
        word = x[:, :18]
        pos = x[:, 18:36]
        arc = x[:, 36:48]

        word_embedding = self.word_emb(word)
        pos_embedding = self.pos_emb(pos)
        arc_embedding = self.arc_emb(arc)

        embedding = torch.cat([word_embedding.reshape(-1, 900), pos_embedding.reshape(-1, 900), arc_embedding.reshape(-1, 600)], dim=1)
        # embedding = self.dropout(embedding)

        hidden1 = self.hidden1(embedding)

        hidden1 = torch.tanh(hidden1)
        hidden2 = torch.tanh(self.hidden2(hidden1))

        output_trans  = self.out_trans(hidden2)
        output_arcs  =  self.out_arcs(hidden2)
        return output_trans, output_arcs


