import torch
import numpy as np


def trans_from_onehot(y):
    batch_size = y.size(0)
    n_classes = y.size(1)
    result = [0 for i in range(batch_size)]
    for i in range(batch_size):
        mark = 0
        for j in range(n_classes):
            if int(y[i][j]) == 1:
                mark = j
                break
        result[i] = mark
    return torch.from_numpy(np.array(result)).type(torch.LongTensor)


def trans_to_onehot(y, n_classes):
    batch_size = y.size(0)
    result = torch.zeros(batch_size, n_classes)
    result = result.scatter(dim=1, index=result, src=1)
    return result
