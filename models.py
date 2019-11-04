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
        self.linear = nn.ModuleList(
            [nn.Linear(output_length, output_length) for _ in range(self.n)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(output_length, output_length) for _ in range(self.n)]
        )
        for linear, gate in zip(self.linear, self.gate):
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(gate.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.transpose(1, 2)
        for linear, gate in zip(self.linear, self.gate):
            gate = torch.sigmoid(gate(x))
            output = linear(x)
            output = F.relu(output)
            output = F.dropout(output, p=config.LAYERS_DROPOUT, training=self.training)
            x = gate * output + (1 - gate) * x
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

    def __init__(self, wemb_dim, cemb_dim, out_features):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, cemb_dim, kernel_size=(1, 5))
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.highway = Highway(2, wemb_dim + cemb_dim)
        self.resizer = nn.Linear(wemb_dim + cemb_dim, out_features)

    def forward(self, cemb: torch.Tensor, wemb: torch.Tensor):
        cemb = cemb.permute((0, 3, 1, 2))
        cemb = F.dropout(cemb, p=config.CHAR_EMBEDDING_DROPOUT, training=self.training)
        cemb = self.conv2d(cemb)
        cemb = F.relu(cemb)
        cemb, _ = torch.max(cemb, dim=3)

        wemb = F.dropout(wemb, p=config.WORD_EMBEDDING_DROPOUT, training=self.training)
        wemb = wemb.transpose(1, 2)

        emb = torch.cat((cemb, wemb), dim=1)
        emb = self.highway(emb)

        emb = emb.transpose(1, 2)
        emb = self.resizer(emb)
        emb = emb.transpose(1, 2)

        return emb


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, activation=F.relu):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise_convolution = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.pointwise_convolution = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=0,
            kernel_size=1,
            bias=bias
        )
        self.activation = activation

    def forward(self, x):
        x = self.depthwise_convolution(x)
        x = self.pointwise_convolution(x)
        if self.activation:
            x = self.activation(x)
        return x


class PositionEncoder(nn.Module):
    """
    copy from https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py
    input:
        x: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    output:
        x: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
    """

    def __init__(self, hidden_size, max_length=512, min_timescale=1.0, max_timescale=1.0e4, ):
        super(PositionEncoder, self).__init__()
        position = torch.arange(max_length).float()
        num_timescales = hidden_size // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (
            float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        self.signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1)
        self.signal = nn.ZeroPad2d((0, (hidden_size % 2), 0, 0))(self.signal)
        self.signal = self.signal.view(1, max_length, hidden_size).to(device)

    def forward(self, x):
        x = x.transpose(1, 2)
        x += self.signal[:, :x.shape[1], :]
        x = x.transpose(1, 2)
        return x


# TODO trace mask
def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target + (-1e30) * (1 - mask)


class EncoderBlock(nn.Module):
    """
    input:
        x: shape [batch_size, hidden_size, max length] => [8, 500, 400]
        mask: shape [batch_size, max length] => [8, 400]
    output:
        x: shape [batch_size, hidden_size, max length] => [8, 128, 400]
    """

    def __init__(self, convolution_number, hidden_size, kernel_size, head_number=8):
        super(EncoderBlock, self).__init__()
        self.convolution_number = convolution_number
        self.position_encoder = PositionEncoder(hidden_size)
        self.convolution_list = nn.ModuleList([
            DepthwiseSeparableConvolution(hidden_size, hidden_size, kernel_size)
            for _ in range(convolution_number)
        ])

        self.convolution_normalization_list = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(convolution_number)]
        )
        self.attention_layer_normalization = nn.LayerNorm(hidden_size)

        self.self_attention = nn.MultiheadAttention(hidden_size, head_number,
                                                    dropout=config.LAYERS_DROPOUT)
        self.feedforward_normalization = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.feedforward.weight, nonlinearity='relu')

    def forward(self, x, mask):
        x = self.position_encoder(x)
        for i in range(self.convolution_number):
            raw = x
            x = self.convolution_normalization_list[i](x.transpose(1, 2)).transpose(1, 2)
            x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
            x = self.convolution_list[i](x.cuda())
            x = F.dropout(x, config.LAYERS_DROPOUT * (i + 1) / self.convolution_number,
                          training=self.training)
            x = raw + x

        raw = x
        x = self.attention_layer_normalization(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
        x = x.permute(2, 0, 1)
        x, _ = self.self_attention(x, x, x, key_padding_mask=mask.bool())
        x = x.permute(1, 2, 0)
        x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
        x = raw + x

        raw = x
        x = self.feedforward_normalization(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
        x = self.feedforward(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = F.dropout(x, config.LAYERS_DROPOUT, training=self.training)
        x = raw + x
        return x


class CQAttention(nn.Module):
    """
    input:
        C: shape [batch_size, hidden_size, context max length] => [8, 128, 400]
        Q: shape [batch_size, hidden_size, question max length] => [8, 128, 50]
        cmask: shape [batch_size, context max length] => [8, 400]
        qmask: shape [batch_size, question max length] => [8, 50]
    output:
        attention: shape [batch_size, hidden_size, context max length] => [8, 512, 400]
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.line_project = torch.empty(hidden_size * 3, 1)
        nn.init.xavier_uniform_(self.line_project)
        self.line_project = nn.Parameter(self.line_project.squeeze(), requires_grad=True)
        self.resizer = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, C, Q, cmask, qmask):
        # calculate CQ similarity
        C = C.transpose(1, 2)  # shape [batch_size, context max length, hidden_size]
        Q = Q.transpose(1, 2)  # shape [batch_size, question max length, hidden_size]
        cmask = cmask.unsqueeze(2)  # shape [batch_size, context max length, 1]
        qmask = qmask.unsqueeze(1)  # shape [batch_size, 1, question max length]

        # calculate S via trilinear function
        # (batch_size, context max length, question max length, hidden_size)
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)  # element-wise multiplication
        S = torch.matmul(torch.cat((Ct, Qt, CQ), dim=3), self.line_project)

        # context-wise softmax (one context word to every question word)
        S_row_sofmax = F.softmax(mask_logits(S, qmask), dim=2)

        # question-wise softmax (one question word to every context word)
        S_column_softmax = F.softmax(mask_logits(S, cmask), dim=1)

        A = torch.bmm(S_row_sofmax, Q)
        B = torch.bmm(torch.bmm(S_row_sofmax, S_column_softmax.transpose(1, 2)), C)
        output = torch.cat((C, A, torch.mul(C, A), torch.mul(C, B)), dim=2)
        output = F.dropout(output, p=config.LAYERS_DROPOUT, training=self.training)
        output = self.resizer(output)
        output = output.transpose(1, 2)
        return output


class OutputLayer(nn.Module):
    def __init__(self, hidden_size):
        super(OutputLayer, self).__init__()
        self.weight1 = torch.empty(hidden_size * 2, 1)
        self.weight2 = torch.empty(hidden_size * 2, 1)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        self.weight1 = nn.Parameter(self.weight1.squeeze(), requires_grad=True)
        self.weight2 = nn.Parameter(self.weight2.squeeze(), requires_grad=True)

    def forward(self, stacked_model_output1, stacked_model_output2, stacked_model_output3, cmask):
        start = torch.cat((stacked_model_output1, stacked_model_output2), dim=1)
        end = torch.cat((stacked_model_output1, stacked_model_output3), dim=1)
        start = torch.matmul(self.weight1, start)
        end = torch.matmul(self.weight2, end)
        start = mask_logits(start, cmask)
        end = mask_logits(end, cmask)
        return start, end


class QANet(nn.Module):
    """
    input:
        Cwid: context word id, shape [batch_size, context max length] => [8, 400]
        Ccid: context word id, shape [batch_size, context max length, word length] => [8, 400, 16]
        Qwid: context word id, shape [batch_size, Question max length] => [8, 50]
        Qcid: context word id, shape [batch_size, Question max length, word length] => [8, 50, 16]
    output:
        start: start position probability, shape [batch_size, context max length] => [8, 400]
        end: end position probability, shape [batch_size, context max length] => [8, 400]
    """

    def __init__(self, word_mat, char_mat):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.char_embedding = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.embedding = Embedding(word_mat.shape[1], char_mat.shape[1], config.HIDDEN_SIZE)
        self.embedding_encoder = EncoderBlock(
            convolution_number=config.EMBEDDING_ENCODE_CONVOLUTION_NUMBER,
            hidden_size=config.HIDDEN_SIZE,
            kernel_size=config.EMBEDDING_ENCODER_CONVOLUTION_KERNEL_SIZE
        )
        self.cq_attention = CQAttention(hidden_size=config.HIDDEN_SIZE)
        output_encoder_block = EncoderBlock(
            convolution_number=config.MODEL_ENCODER_CONVOLUTION_NUMBER,
            hidden_size=config.HIDDEN_SIZE,
            kernel_size=config.MODEL_ENCODER_CONVOLUTION_KERNEL_SIZE)
        self.model_encoder = nn.ModuleList(
            [output_encoder_block for _ in range(config.MODEL_ENCODER_BLOCK_NUMBER)])
        self.output = OutputLayer(hidden_size=config.HIDDEN_SIZE)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (torch.zeros_like(Cwid) != Cwid).float()
        qmask = (torch.zeros_like(Qwid) != Qwid).float()
        Cw, Cc = self.word_embedding(Cwid), self.char_embedding(Ccid)
        Qw, Qc = self.word_embedding(Qwid), self.char_embedding(Qcid)
        C, Q = self.embedding(Cc, Cw), self.embedding(Qc, Qw)
        C, Q = self.embedding_encoder(C, cmask), self.embedding_encoder(Q, qmask)
        CQ_attention = self.cq_attention(C, Q, cmask, qmask)
        stacked_model_input = F.dropout(CQ_attention, p=config.LAYERS_DROPOUT,
                                        training=self.training)
        for enc in self.model_encoder:
            stacked_model_input = enc(stacked_model_input, cmask)
        stacked_model_output1 = stacked_model_input
        for enc in self.model_encoder:
            stacked_model_input = enc(stacked_model_input, cmask)
        stacked_model_output2 = stacked_model_input
        for enc in self.model_encoder:
            stacked_model_input = enc(stacked_model_input, cmask)
        stacked_model_output3 = stacked_model_input
        start, end = self.output(stacked_model_output1, stacked_model_output2,
                                 stacked_model_output3, cmask)
        return start, end
