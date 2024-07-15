import torch
import torch.nn as nn


class ResNet1DConfig:
    """
        model_name format: resnet1d_(layers)
    """
    model_name = 'resnet1d_18'
    in_channel = 6
    inplane = 64

    def __init__(self):
        super(ResNet1DConfig, self).__init__()


class BasicConv(nn.Module):
    def __init__(self, in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias=False,
                 activation=False):
        super(BasicConv, self).__init__()
        block = [
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channel),
        ]
        if activation:
            block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicConv(inplanes, planes,
                               kernel_size=3, stride=stride, padding=1, activation=True)
        self.conv2 = BasicConv(planes, planes,
                               kernel_size=3, stride=1, padding=1, activation=False)
        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BasicConv(inplanes, planes,
                               kernel_size=1, stride=1, padding=0, activation=True)
        self.conv2 = BasicConv(planes, planes,
                               kernel_size=3, stride=stride, padding=1, activation=True)
        self.conv3 = BasicConv(planes, planes * self.expansion,
                               kernel_size=1, stride=1, padding=0, activation=False)
        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, config: ResNet1DConfig):
        super(ResNet1D, self).__init__()
        self.model_name = config.model_name

        self.inplane = config.inplane

        self.in_conv = BasicConv(config.in_channel, config.inplane,
                                 kernel_size=7, stride=2, padding=3, activation=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.output_size = 512 * block.expansion
        self.head = nn.Linear(self.output_size, 10)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplane != planes * block.expansion:
            downsample = BasicConv(self.inplane, planes * block.expansion,
                                   kernel_size=1, stride=stride, padding=0, activation=False)
        layers = []
        layers.append(block(self.inplane, planes, stride, downsample))
        self.inplane = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=-1)
        return self.head(x)

    def get_output_size(self):
        return self.output_size

    def get_model_name(self):
        return self.model_name


def resnet1d(config: ResNet1DConfig):
    if config.model_name == 'resnet1d_18':
        return ResNet1D(BasicBlock, [2, 2, 2, 2], config)
    elif config.model_name == 'resnet1d_34':
        return ResNet1D(BasicBlock, [3, 4, 6, 3], config)
    elif config.model_name == 'resnet1d_50':
        return ResNet1D(Bottleneck, [3, 4, 6, 3], config)
    elif config.model_name == 'resnet1d_101':
        return ResNet1D(Bottleneck, [3, 4, 23, 3], config)
