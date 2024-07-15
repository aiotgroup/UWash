import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UTFormerEVOConfig(object):
    """
    Watch: in_channel = 6, n_head = 8, segment_length = 64, hidden_dim = 16, n_layer = 3
    """

    def __init__(self, model_name: str, n_layer: int):
        self.model_name = model_name
        self.MAX_PATCH_NUM = 512
        self.n_axis = 3
        self.in_channel = self.n_axis * 2
        self.n_head = 8
        self.segment_length = 64
        self.hidden_dim = 16
        self.n_layer = n_layer
        self.n_path = 3
        self.dropout = 0.1


class BaseModel(nn.Module):
    model_name = None

    def __init__(self, model_name: str):
        super(BaseModel, self).__init__()
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name


class SimpleConv(nn.Module):
    """
    forward: 卷积 + Norm + ReLU

    init:
        in_channels(int): 输入数据通道数
        out_channels(int): 输出通道数
        kernel_size(int): 卷积核大小
        stride(int): 卷积核步长
        padding(int): 卷积padding
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(SimpleConv, self).__init__()

        self.conv = nn.Conv1d(in_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.BatchNorm1d(out_channel)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.conv(x))))


class DownSample(nn.Module):
    """
    下采样：kernel = 3, stride = 2, padding = 1 的SimpleConv
    输入数据维度 -> 输出数据维度:
        batch_size  -> batch_size
        channel     -> channel * 2
        seq_len     -> seq_len / 2

    init:
        in_channels(int): 输入数据通道数
        out_channels(int): 输出通道数
    """

    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()

        self.down_sample = SimpleConv(in_channel, out_channel,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    """
    上采样：逆卷积
        输入数据维度 -> 输出数据维度:
        channel -> channel / 2
        seq_len -> seq_len * 2
    """

    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()

        self.up_sample = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_sample(x)


class TemporalConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
        )
        self.conv_out = nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv_out(self.conv2(self.conv1(x)))


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    """
    Scaled Dot-Product Attention
    :param query: 输入与w_query矩阵相乘后的结果, (batch_size , h , seq_len , d_model // h)
    :param key: 输入与w_key矩阵相乘后的结果, (batch_size , h , seq_len , d_model // h)
    :param value: 输入与w_value矩阵相乘后的结果, (batch_size , h , seq_len , d_model // h)
    """

    def forward(self, query, key, value):
        d_k = query.size(-1)
        # 计算attention score, 即QK/根号d_k, batch_size,h,seq_len,seq_len
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 以最后一个维度进行softmax(也就是最内层的行), batch_size,h,seq_len,seq_len
        scores = F.softmax(scores, dim=-1)

        # 与V相乘. 第一个输出的 batch_size,h,seq_len,d_model//h
        return torch.matmul(scores, value)


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention = ScaledDotProductAttention()
        self.w_query = TemporalConv(self.d_model, self.n_head * self.d_k)
        self.w_key = TemporalConv(self.d_model, self.n_head * self.d_k)
        self.w_value = TemporalConv(self.d_model, self.n_head * self.d_k)
        self.out = TemporalConv(self.n_head * self.d_v, self.d_model)
        self.ln = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def _partition_head(self, input):
        # input: batch_size, seq_len, d_model
        # output: batch_size, n_head, seq_len, d_model // n_head
        batch_size, seq_len, d_model = input.size()
        assert d_model % self.n_head == 0
        return input.view(batch_size, seq_len, self.n_head, -1).permute(0, 2, 1, 3)

    """
    Multi-Head Attention
    :param query: (batch_size , seq_len , d_model)
    :param key: (batch_size , seq_len , d_model)
    :param value: (batch_size , seq_len , d_model)
    """

    def forward(self, query, key, value):
        batch_size = query.size(0)
        residual = query
        query = self._partition_head(self.dropout(self.w_query(query.permute(0, 2, 1)).permute(0, 2, 1)))
        key = self._partition_head(self.dropout(self.w_key(key.permute(0, 2, 1)).permute(0, 2, 1)))
        value = self._partition_head(self.dropout(self.w_value(value.permute(0, 2, 1)).permute(0, 2, 1)))

        # context: batch_size, n_head, seq_len, d_v
        context = self.attention(query, key, value)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_head * self.d_v)
        context = self.dropout(self.out(context.permute(0, 2, 1)).permute(0, 2, 1))
        return self.ln(context + residual)


class MultiPathConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, n_path: int = 3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        for index in range(n_path):
            self.convs.append(
                SimpleConv(in_channel, out_channel, kernel_size=1 + 2 * index, stride=1, padding=index, dropout=dropout)
            )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        result = x
        for output in outputs:
            result = result + output
        return result


class UTFormerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_head: int, n_path: int = 3, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_path = n_path

        self.multi_head_attention = MultiHeadedAttention(n_head,
                                                         hidden_dim,
                                                         hidden_dim // n_head,
                                                         hidden_dim // n_head,
                                                         dropout)
        self.multi_path_conv = MultiPathConv(hidden_dim, hidden_dim, n_path, dropout)

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        # 强化每个时间点的特征
        x = self.multi_head_attention(x, x, x)
        # 融合相邻时间点相同特征
        x = self.multi_path_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.ln(x + residual)


class UTFormerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, n_head: int, n_path: int = 3, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_path = n_path

        self.multi_head_attention_in = MultiHeadedAttention(n_head, hidden_dim,
                                                            hidden_dim // n_head, hidden_dim // n_head, dropout)
        self.multi_head_attention_out = MultiHeadedAttention(n_head, hidden_dim,
                                                             hidden_dim // n_head, hidden_dim // n_head, dropout)
        self.multi_path_conv = MultiPathConv(hidden_dim, hidden_dim, n_path, dropout)

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, query):
        residual = x

        x = self.multi_head_attention_in(x, x, x)
        x = self.multi_head_attention_out(query, x, x)
        x = self.multi_path_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.ln(x + residual)


class UTFormerEVO(BaseModel):
    def __init__(self, config: UTFormerEVOConfig):
        super().__init__(config.model_name)
        self.config = config
        self.hidden_dim = self.config.hidden_dim
        self.segment_length = self.config.segment_length
        self.n_head = self.config.n_head
        self.n_layer = self.config.n_layer

        self.embedding = nn.Linear(self.config.in_channel, self.hidden_dim, bias=False)
        self.cls_embedding = nn.Parameter(torch.zeros((1, self.segment_length, self.hidden_dim)),
                                          requires_grad=True)
        self.hidden_dim = self.hidden_dim * 2
        self.position_embedding = nn.Parameter(torch.empty((1, self.segment_length, self.hidden_dim)),
                                               requires_grad=True)

        self.encoders = nn.ModuleList([
            UTFormerEncoder(hidden_dim=self.hidden_dim * int(2 ** _), n_head=self.n_head,
                            n_path=self.config.n_path, dropout=self.config.dropout) for _ in range(self.config.n_layer)
        ])
        self.projector_lefts = nn.ModuleList([
            DownSample(in_channel=self.hidden_dim * int(2 ** _),
                       out_channel=self.hidden_dim * int(2 ** (_ + 1))) for _ in range(self.config.n_layer)
        ])
        self.encoder_out = UTFormerEncoder(hidden_dim=self.hidden_dim * int(2 ** self.n_layer), n_head=self.n_head,
                                           n_path=self.config.n_path, dropout=self.config.dropout)

        self.decoders = nn.ModuleList([
            UTFormerDecoder(hidden_dim=self.hidden_dim * int(2 ** _), n_head=self.n_head,
                            n_path=self.config.n_path, dropout=self.config.dropout) for _ in range(self.config.n_layer)
        ])
        self.projector_rights = nn.ModuleList([
            UpSample(in_channel=self.hidden_dim * int(2 ** (_ + 1)),
                     out_channel=self.hidden_dim * int(2 ** _)) for _ in range(self.config.n_layer)
        ])
        self.decoder_in = UTFormerDecoder(hidden_dim=self.hidden_dim * int(2 ** self.n_layer), n_head=self.n_head,
                                          n_path=self.config.n_path, dropout=self.config.dropout)

        nn.init.kaiming_normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.position_embedding)

    def forward(self, x):
        # acc, gyr = (batch_size, n_axis, segment_length), (batch_size, n_axis, segment_length)
        # -> x = (batch_size, segment_length, n_axis * 2)
        x = torch.cat(x, dim=1).permute(0, 2, 1)
        # embedding
        x = self.embedding(x)
        batch_size = x.size(0)
        # class token embedding
        x = torch.cat([self.cls_embedding.repeat(batch_size, 1, 1), x], dim=-1)
        # position embedding
        x = x + self.position_embedding.repeat(batch_size, 1, 1)

        features = []
        # encode and down sample
        for encoder, lefter in zip(self.encoders, self.projector_lefts):
            x = encoder(x)
            features.append(x)
            x = lefter(x.permute(0, 2, 1)).permute(0, 2, 1)
        # bottom
        x = self.encoder_out(x)
        x = self.decoder_in(x, x)

        # decoder and up sample
        for feature, decoder, righter in zip(features[::-1], self.decoders[::-1], self.projector_rights[::-1]):
            x = righter(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = decoder(x, feature)
        # batch_size, hidden_dim, segment_length
        return x.permute(0, 2, 1)

    def get_hidden_dim(self):
        return self.hidden_dim


if __name__ == '__main__':
    acc, gyr = torch.randn((8, 3, 64)), torch.randn((8, 3, 64))
    utformer_evo = UTFormerEVO(UTFormerEVOConfig(("utformer_evo"), 3))
    abc = utformer_evo((acc, gyr))
