import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SimpleConv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class PositionalEmbedding(nn.Module):
    """
    :param d_model: pe编码维度，一般与word embedding相同，方便相加
    :param max_len: 序列最长长度
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # 计算pe编码
        pe = torch.zeros(max_len, d_model)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *  # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len, :].repeat(batch_size, 1, 1), requires_grad=False)
        return x


"""
Scaled Dot-Product Attention
:param Q: 输入与W_Q矩阵相乘后的结果,(batch_size , h , seq_len , d_model // h)
:param K: 输入与W_K矩阵相乘后的结果,(batch_size , h , seq_len , d_model // h)
:param V: 输入与W_V矩阵相乘后的结果,(batch_size , h , seq_len , d_model // h)
:param mask: 掩码矩阵 (batch_size , h, seq_len , seq_len)
"""


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        # 计算QK/根号d_k batch_size,h,seq_len,seq_len
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores.masked_fill_(mask=mask, value=torch.tensor(-1e9))

        # 以最后一个维度进行softmax(也就是最内层的行),batch_size,h,seq_len,seq_len
        scores = F.softmax(scores, dim=-1)

        # 与V相乘。第一个输出的 batch_size,h,seq_len,d_model//h,第二个输出的 batch,h,seq_len,seq_len
        return torch.matmul(scores, V), scores


"""
MultiHead Attention
:param input_Q: (batch_size , seq_len , d_model)
:param input_K: (batch_size , seq_len , d_model)
:param input_V: (batch_size , seq_len , d_model)
:param mask: (batch_size , seq_len , seq_len)
"""


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention = ScaledDotProductAttention()
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_v, bias=False)
        self.out = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.ln = nn.LayerNorm(self.d_model)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, input_Q, input_K, input_V, mask=None):
        batch_size = input_Q.size(0)
        residual = input_Q
        # Q: batch_size , seq_len , d_model
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        # K: batch_size , seq_len , d_model
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        # V: batch_size , seq_len , d_model
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        if mask is not None:
            # mask : (batch_size, n_heads, seq_len, seq_len)
            mask = mask.reshape(batch_size, 1, mask.size(1), mask.size(2)).repeat(1, self.n_heads, 1, 1)

        # context: batch_size, n_heads, seq_len, d_v, attn: batch,h,seq_len,seq_len
        context, scores = self.attention(Q, K, V, mask)

        # context: batch_size, len_q, n_heads * d_v = batch_size, seq_len, d_model
        context = context.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.d_v)
        # batch_size, len_q, d_model
        context = self.out(context)
        return self.ln(context + residual), scores


"""
FeedForward
:param x: batch_size , seq_len , d_model
"""


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_in = nn.Linear(self.d_model, self.d_ff)
        self.linear_out = nn.Linear(self.d_ff, self.d_model)
        self.ln = nn.LayerNorm(self.d_model)

        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x):
        res = F.relu(self.linear_in(x))
        res = self.linear_out(res)
        return self.ln(res + x)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.multi_head_attention = MultiHeadedAttention(self.n_heads, self.d_model, self.d_k, self.d_v)
        self.feed_forward = FeedForward(self.d_model, self.d_ff)

    def forward(self, enc_inputs):
        enc_outputs, scores = self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs, None)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs, scores


class Transformer(nn.Module):
    def __init__(self, n_axis, n_classes):
        super(Transformer, self).__init__()
        self.in_conv = SimpleConv(n_axis, 64, kernel_size=1, stride=1, padding=0)
        self.position_embedding = PositionalEmbedding(64)
        self.encoders = nn.ModuleList([EncoderLayer(8, 64, 8, 8, 512) for _ in range(3)])
        self.out_linear = nn.Linear(64, n_classes)

        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.in_conv(x)
        x = x.permute(0, 2, 1)
        x = self.position_embedding(x)
        for encoder in self.encoders:
            x, score = encoder(x)
        return torch.sigmoid(self.out_linear(x))

    def get_model_name(self):
        return self.__class__.__name__
