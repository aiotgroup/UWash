import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from UWasher.model.ModelConfig import ModelConfig


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

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SimpleConv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


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

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.down_sample = SimpleConv(in_channels, out_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)

    def forward(self, x):
        return self.down_sample(x)


class DownConv(nn.Module):
    """
    下采样 + Conv
        输入数据维度 -> 输出数据维度:
        batch_size  -> batch_size
        channel     -> channel * 2
        seq_len     -> seq_len / 2
    init:
        in_channels(int): 输入数据通道数
    """

    def __init__(self, in_channels):
        super(DownConv, self).__init__()
        self.down = DownSample(in_channels, in_channels * 2)
        self.out_conv = SimpleConv(in_channels * 2, in_channels * 2)

    def forward(self, x):
        return self.out_conv(self.down(x))


class UpSample(nn.Module):
    """
    上采样：逆卷积
        输入数据维度 -> 输出数据维度:
        channel -> channel / 2
        seq_len -> seq_len * 2
    """

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.up_sample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_sample(x)


class UpConv(nn.Module):
    """
    上采样 + Conv
        输入数据维度 -> 输出数据维度:
        batch_size  -> batch_size
        in_channel  -> out_channel
        seq_len     -> seq_len * 2
    """

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = UpSample(in_channels, out_channels)
        self.out_conv = SimpleConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, before_x):
        feature = torch.cat([before_x, x], dim=1)
        return self.out_conv(self.up(feature))


class SpatialPyramidPooling(nn.Module):
    """
    空间金字塔池化

    init:
        pool_sizes(list(int)): 多层pooling的尺寸
    """

    def __init__(self, pool_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.avg_pools = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=size, stride=size) for size in pool_sizes]
        )

    def forward(self, x):
        features = [avg(x) for avg in self.avg_pools]
        return features


class SPPConvUp(nn.Module):
    """
    空间金字塔池化 + SimpleConv + Upsample

    init:
        pool_sizes(list(int)): 多层pooling的尺寸
        in_channels(int): 输入数据通道数
        out_channels(int): 输出数据通道数
    """

    def __init__(self, pool_sizes, in_channels, out_channels):
        super(SPPConvUp, self).__init__()
        self.spp = SpatialPyramidPooling(pool_sizes)
        self.convs = nn.ModuleList(
            [SimpleConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0) for _ in range(len(pool_sizes))]
        )
        self.ups = nn.ModuleList(
            [nn.Upsample(scale_factor=size, mode="nearest") for size in pool_sizes]
        )

    def forward(self, x):
        features = self.spp(x)
        features = [conv(feature) for feature, conv in zip(features, self.convs)]
        features = [up(feature) for feature, up in zip(features, self.ups)]
        return torch.cat(features, dim=1)


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        fc_in = nn.Linear(channels, channels // reduction, bias=False)
        fc_out = nn.Linear(channels // reduction, channels, bias=False)

        nn.init.xavier_uniform_(fc_in.weight)
        nn.init.xavier_uniform_(fc_out.weight)

        self.fc = nn.Sequential(
            fc_in,
            nn.ReLU(inplace=True),
            fc_out,
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        y = self.avg_pooling(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y.expand_as(x)


class CoreUNet(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes, bottom_channels, num_layers=3):
        super(CoreUNet, self).__init__()

        factors = [int(2 ** i) for i in range(num_layers)]

        self.in_conv = SimpleConv(in_channels, out_channels)

        self.downs = nn.ModuleList(
            [DownConv(out_channels * factor) for factor in factors]
        )

        self.spp = SPPConvUp(pool_sizes, out_channels * int(2 ** num_layers), bottom_channels)

        self.ups = nn.ModuleList(
            [UpConv(bottom_channels * num_layers + out_channels * factor * 2, out_channels * factor) if i == 0
             else UpConv(out_channels * factor * 4, out_channels * factor)
             for i, factor in enumerate(factors[::-1])]
        )

        factors.append(int(2 ** num_layers))
        self.skips = nn.ModuleList(
            [SimpleConv(out_channels * factor, out_channels * factor, kernel_size=1, stride=1, padding=0)
             for factor in factors[::-1]]
        )

    def forward(self, x):
        x = self.in_conv(x)
        before_x = x
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)

        x = self.spp(x)

        for feature, up, skip in zip(features[::-1], self.ups, self.skips):
            x = up(x, skip(feature))

        x = torch.cat([before_x, x], dim=1)
        return x


class UWasher(nn.Module):
    def __init__(self, config=ModelConfig()):
        super(UWasher, self).__init__()
        self.acc_spp_unet = CoreUNet(config.n_axis, config.init_channels, config.pool_sizes, config.bottom_channels)
        self.acc_attention = SELayer(config.init_channels * 2)

        self.gyr_spp_unet = CoreUNet(config.n_axis, config.init_channels, config.pool_sizes, config.bottom_channels)
        self.gyr_attention = SELayer(config.init_channels * 2)

        self.out = nn.Sequential(
            SimpleConv(config.init_channels * 2 * 2, config.init_channels * 2 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(config.init_channels * 2 * 2, config.n_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        acc, gyr = x
        acc = self.acc_spp_unet(acc)
        acc = self.acc_attention(acc)

        gyr = self.gyr_spp_unet(gyr)
        gyr = self.gyr_attention(gyr)

        x = torch.cat([acc, gyr], dim=1)
        return self.out(x)

####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
