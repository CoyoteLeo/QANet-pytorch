import torch
from torch import nn
from torch.nn import functional as F

import config


class CQAttention(nn.Module):
    def __init__(self):
        super(CQAttention, self).__init__()

    def forward(self, *input):
        pass


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass


class Highway(nn.Module):
    """
    input:
        x: concat(word_embedding, character_embedding) , shape [batch_size, embedding length, context max length] => [8, 500, 400]
    output:
        emb: embedding result, shape [batch_size, embedding length, context max length] => [8, 500, 400]
    """

    def __init__(self, layer_number, output_length):
        super().__init__()
        self.n = layer_number
        self.linear = nn.ModuleList([nn.Linear(output_length, output_length) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(output_length, output_length) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length] => [8, 500, 400]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            nonlinear = F.dropout(nonlinear, p=config.LAYERS_DROPOUT, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class Embedding(nn.Module):
    """
    input:
        cemb: character embedding, shape [batch_size, context max length, word length, character embedding length] => [8, 400, 16, 200]
        wemb word embedding, shape [batch_size, context max length, word embedding length] => [8, 400, 300]
    output:
        emb: embedding result, shape [batch_size, embedding length, context max length] => [8, 500, 400]
1    """

    def __init__(self, wemb_dim, cemb_dim):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, cemb_dim, kernel_size=(1, 5), padding=0, bias=True)
        self.highway = Highway(2, wemb_dim + cemb_dim)

    def forward(self, cemb: torch.Tensor, wemb: torch.Tensor):
        cemb = cemb.permute((0, 3, 1, 2))
        cemb = F.dropout(cemb, p=config.WORD_EMBEDDING_DROPOUT, training=self.training)
        cemb = self.conv2d(cemb)
        cemb = F.relu(cemb)
        cemb, _ = torch.max(cemb, dim=3)
        cemb = cemb.squeeze()
        wemb = F.dropout(wemb, p=config.WORD_EMBEDDING_DROPOUT, training=self.training)
        wemb = wemb.transpose(1, 2)
        emb = torch.cat((cemb, wemb), dim=1)
        emb = self.highway(emb)
        return emb


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, activation=F.relu):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels,
                                               bias=bias)
        self.pointwise_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x):
        y = self.pointwise_convolution(self.depthwise_convolution(x))
        if self.activation:
            y = self.activation(y)
        return y


class QANet(nn.Module):
    """
    input:
        Cwid: context word id, shape [batch_size, context max length] => [8, 400]
        Ccid: context word id, shape [batch_size, context max length, word length] => [8, 400, 16]
        Qwid: context word id, shape [batch_size, Question max length] => [8, 50]
        Qcid: context word id, shape [batch_size, Question max length, word length] => [8, 50, 16]
    output:
        pass
    """

    def __init__(self, word_mat, char_mat):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.char_embedding = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.embedding = Embedding(word_mat.shape[1], char_mat.shape[1])

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        # cmask = (torch.zeros_like(Cwid) == Cwid).float()
        # qmask = (torch.zeros_like(Qwid) == Qwid).float()
        # maskC = (torch.ones_like(Cwid) *self.PAD != Cwid).float()
        # maskQ = (torch.ones_like(Qwid) *self.PAD != Qwid).float()
        Cw, Cc = self.word_embedding(Cwid), self.char_embedding(Ccid)
        Qw, Qc = self.word_embedding(Qwid), self.char_embedding(Qcid)
        C, Q = self.embedding(Cc, Cw), self.embedding(Qc, Qw)
