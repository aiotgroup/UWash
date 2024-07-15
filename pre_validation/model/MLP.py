import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_features, n_classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features, 1024)
        self.linear2 = nn.Linear(1024, in_features)
        self.linear3 = nn.Linear(in_features, n_classes)

        self.activation = torch.sigmoid

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_model_name(self):
        return self.__class__.__name__
