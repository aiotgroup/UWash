import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SimpleConv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class BottleNeck(nn.Module):
    def __init__(self, in_out_channels):
        super(BottleNeck, self).__init__()
        self.in_conv = SimpleConv(in_out_channels, in_out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.calc_conv = SimpleConv(in_out_channels // 4, in_out_channels // 4, kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Conv1d(in_out_channels // 4, in_out_channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm1d(in_out_channels)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.out_conv.weight)

    def forward(self, residual):
        x = self.in_conv(residual)
        x = self.calc_conv(x)

        return self.activation(residual + self.norm(self.out_conv(x)))


class UNet(nn.Module):

    def __init__(self, in_features, seq_len, n_classes):
        super(UNet, self).__init__()
        self.in_conv = SimpleConv(in_features, 64, kernel_size=1, stride=1, padding=0)

        self.in_bottleneck1 = BottleNeck(64)
        self.down1 = SimpleConv(64, 128, kernel_size=1, stride=2, padding=0)

        self.in_bottleneck2 = BottleNeck(128)
        self.down2 = SimpleConv(128, 256, kernel_size=1, stride=2, padding=0)

        self.in_bottleneck3 = BottleNeck(256)
        self.down3 = SimpleConv(256, 512, kernel_size=1, stride=2, padding=0)

        self.bottom_bottleneck = BottleNeck(512)

        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.out_conv3 = SimpleConv(512, 256, kernel_size=1, stride=1, padding=0)
        self.out_bottleneck3 = BottleNeck(256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.out_conv2 = SimpleConv(256, 128, kernel_size=1, stride=1, padding=0)
        self.out_bottleneck2 = BottleNeck(128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.out_conv1 = SimpleConv(128, 64, kernel_size=1, stride=1, padding=0)
        self.out_bottleneck1 = BottleNeck(64)

        self.out_linear = nn.Linear(64, n_classes)

        nn.init.xavier_uniform_(self.out_linear.weight)

    # batch_size, n_axis, seq_len
    def forward(self, x):
        # batch_size, 64, seq_len
        left_1 = self.in_bottleneck1(self.in_conv(x))
        # batch_size, 128, seq_len // 2
        left_2 = self.in_bottleneck2(self.down1(left_1))
        # batch_size, 256, seq_len // 4
        left_3 = self.in_bottleneck3(self.down2(left_2))
        # batch_size, 256, seq_len // 4
        right_3 = self.up3(self.bottom_bottleneck(self.down3(left_3)))
        # batch_size, 256, seq_len // 4
        x3 = self.out_conv3(torch.cat([left_3, right_3], dim=1))
        # batch_size, 128, seq_len // 2
        right_2 = self.up2(self.out_bottleneck3(x3))
        # batch_size, 128, seq_len // 2
        x2 = self.out_conv2(torch.cat([left_2, right_2], dim=1))
        # batch_size, 64, seq_len
        right_1 = self.up1(self.out_bottleneck2(x2))
        # batch_size, 64, seq_len
        x1 = self.out_conv1(torch.cat([left_1, right_1], dim=1))
        # batch_size, 64, seq_len
        x = self.out_bottleneck1(x1)

        # channel 和 sequence 互换
        # batch_size, seq_len, 64
        x = x.permute(0, 2, 1)
        # batch_size, seq_len, n_classes
        return torch.sigmoid(self.out_linear(x))

    def get_model_name(self):
        return self.__class__.__name__
