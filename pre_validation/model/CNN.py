import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class CNN(nn.Module):

    def __init__(self, in_channels, seq_len, n_classes):
        super(CNN, self).__init__()
        self.conv1 = SimpleConv(in_channels, 64)
        self.conv2 = SimpleConv(64, 8)
        self.conv3 = SimpleConv(8, 1)
        self.out = nn.Linear(seq_len, n_classes)

        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(dim=1)
        return self.out(x)

    def get_model_name(self):
        return self.__class__.__name__


if __name__ == "__main__":
    cnn = CNN(6, 64, 10)
    print(cnn.get_model_name())
