import math
import torch
from torch import nn
from torch.nn import functional as F

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            nonlinear = F.dropout(nonlinear, p=config.LAYERS_DROPOUT, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
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


class PositionEncoder(nn.Module):
    """
    input:
        x: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    output:
        x: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    """

    def __init__(self, max_length, hidden_size):
        super(PositionEncoder, self).__init__()
        position = torch.arange(0, max_length).unsqueeze(1).float().to(device)
        div_term = torch.tensor([10000 ** (2 * i / hidden_size) for i in range(0, hidden_size // 2)]).to(device)
        self.position_encoding = torch.zeros(max_length, hidden_size, requires_grad=False).to(device)
        self.position_encoding[:, 0::2] = torch.sin(position[:, ] * div_term)
        self.position_encoding[:, 1::2] = torch.cos(position[:, ] * div_term)
        self.position_encoding = self.position_encoding.transpose(0, 1)

    def forward(self, x):
        x = x + self.position_encoding
        return x


class MultiHeadAttention(nn.Module):
    """
    input:
        x: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    output:
        attention: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    """

    def __init__(self, hidden_size, head_number=config.ATTENTION_HEAD_NUMBER):
        super().__init__()
        self.head_number = head_number
        self.dim_per_head = hidden_size // head_number
        self.q_linear = nn.Linear(hidden_size, hidden_size)  # 8 * Wq
        self.v_linear = nn.Linear(hidden_size, hidden_size)  # 8 * Wv
        self.k_linear = nn.Linear(hidden_size, hidden_size)  # 8 * Wk
        self.dropout = nn.Dropout(config.LAYERS_DROPOUT)
        self.linear_project = nn.Linear(hidden_size, hidden_size)
        self.dim_sqrt_invert = 1 / math.sqrt(self.dim_per_head)

    # TODO trace mask
    @staticmethod
    def mask_logits(target, mask):
        return target * (1 - mask) + mask * (-1e30)

    def forward(self, x, mask):
        batch_size, dim, length = x.size()
        x = x.transpose(1, 2)
        q = self.q_linear(x).view(batch_size, length, self.head_number, self.dim_per_head)  # project to eight multihead matrix
        k = self.k_linear(x).view(batch_size, length, self.head_number, self.dim_per_head)  # project to eight multihead matrix
        v = self.v_linear(x).view(batch_size, length, self.head_number, self.dim_per_head)  # project to eight multihead matrix
        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.head_number, length, self.dim_per_head)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.head_number, length, self.dim_per_head)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.head_number, length, self.dim_per_head)
        mask = mask.unsqueeze(1).expand(-1, length, -1).repeat(self.head_number, 1, 1)
        attention = torch.bmm(q, k.transpose(1, 2)) * self.dim_sqrt_invert
        attention = self.mask_logits(attention, mask)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        attention = torch.bmm(attention, v)
        attention = attention.view((self.head_number, batch_size, length, self.dim_per_head)) \
            .permute((1, 2, 0, 3)).contiguous().view((batch_size, length, self.dim_per_head * self.head_number))
        attention = self.linear_project(attention)
        attention = self.dropout(attention)
        return attention.transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, convolution_number, max_length, hidden_size):
        super(EncoderBlock, self).__init__()
        self.convolution_number = convolution_number
        self.conv1d = nn.Conv1d(in_channels=config.GLOVE_WORD_REPRESENTATION_DIM + config.CHAR_REPRESENTATION_DIM,
                                out_channels=config.HIDDEN_SIZE, kernel_size=1)
        self.position_encoder = PositionEncoder(max_length=max_length, hidden_size=hidden_size)
        self.layer_normalization = nn.LayerNorm([hidden_size, max_length])
        self.convolution_list = nn.ModuleList(
            [DepthwiseSeparableConvolution(hidden_size, hidden_size, config.EMBEDDING_ENCODER_CONVOLUTION_KERNEL_SIZE)
             for _ in range(convolution_number)]
        )
        self.self_attention = MultiHeadAttention(hidden_size)

    def forward(self, x, mask):
        x = self.conv1d(x)
        x = self.position_encoder(x)
        for i, conv in enumerate(self.convolution_list):
            raw = x
            x = self.layer_normalization(x)
            x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
            x = conv(x)
            # TODO add input first or dropout first
            x = F.dropout(x, config.LAYERS_DROPOUT * (i + 1) / self.convolution_number, training=self.training)
            x = raw + x
        x = self.layer_normalization(x)
        x = self.self_attention(x, mask)
        return x


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
        self.embedding_encoder = EncoderBlock(convolution_number=config.ENCODE_CONVOLUTION_NUMBER,
                                              max_length=config.PARA_LIMIT, hidden_size=config.HIDDEN_SIZE)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (torch.zeros_like(Cwid) == Cwid).float()
        qmask = (torch.zeros_like(Qwid) == Qwid).float()
        Cw, Cc = self.word_embedding(Cwid), self.char_embedding(Ccid)
        Qw, Qc = self.word_embedding(Qwid), self.char_embedding(Qcid)
        C, Q = self.embedding(Cc, Cw), self.embedding(Qc, Qw)
        C, Q = self.embedding_encoder(C, cmask), self.embedding_encoder(Q, qmask)
