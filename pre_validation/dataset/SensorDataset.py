import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from pre_validation.train_eval.config import Config
from pre_validation.train_eval.util import *


class SensorDatasource():
    config = None
    k = 0

    SEQ_LEN = 0
    OFFSET = 0

    n_axis = 0
    n_features = 0
    n_classes = 0

    x_data = None
    y_data = None
    indexes = None
    sample_nums = 0

    label_method = "segment"

    label_names = {'action1_1': 1,
                   'action1_2': 1,
                   'action1_3': 1,
                   'action1_4': 1,
                   'action2_1_1': 2,
                   'action2_1_2': 2,
                   'action2_1_3': 2,
                   'action2_1_4': 2,
                   'action2_2_1': 3,
                   'action2_2_2': 3,
                   'action2_2_3': 3,
                   'action2_2_4': 3,
                   'action3_1': 4,
                   'action3_2': 4,
                   'action3_3': 4,
                   'action3_4': 4,
                   'action4_1': 5,
                   'action4_2': 5,
                   'action4_3': 5,
                   'action4_4': 5,
                   'action5_1_1': 6,
                   'action5_1_2': 6,
                   'action5_1_3': 6,
                   'action5_1_4': 6,
                   'action5_2_1': 7,
                   'action5_2_2': 7,
                   'action5_2_3': 7,
                   'action5_2_4': 7,
                   'action6_1_1': 8,
                   'action6_1_2': 8,
                   'action6_1_3': 8,
                   'action6_1_4': 8,
                   'action6_2_1': 9,
                   'action6_2_2': 9,
                   'action6_2_3': 9,
                   'action6_2_4': 9,
                   }

    # 将数据集分为K块，K-1块作为训练集，1块作为测试集
    def __init__(self, data_path, config):
        data = pd.read_csv(data_path)

        self.config = config
        self.k = config.k_fold
        self.SEQ_LEN = config.seq_len
        self.OFFSET = config.overlap_offset
        self.n_axis = config.n_axis
        self.n_features = self.SEQ_LEN * self.n_axis
        self.n_classes = config.n_classes
        self.label_method = config.label_method

        x_data, y_data = self.generateOverlapDatasource(data)

        self.x_data = torch.from_numpy(x_data).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(y_data).type(torch.FloatTensor)
        self.sample_nums = x_data.shape[0]
        self.indexes = [i for i in range(self.sample_nums)]
        self.shuffle_data()

    def analyzeLabel(self):
        labels = trans_from_onehot(self.y_data)
        classes = [0 for i in range(config.n_classes)]
        for x in labels:
            classes[x] += 1
        plt.pie(classes, labels=['noaction',
                                 'action1',
                                 'action2-1',
                                 'action2-2',
                                 'action3',
                                 'action4',
                                 'action5-1',
                                 'action5-2',
                                 'action6-1',
                                 'action6-2', ], autopct="%1.2f%%")
        plt.show()

    def analyzeFrameLabel(self):
        classes = [0 for i in range(config.n_classes)]
        for i in range(self.sample_nums):
            for j in range(self.SEQ_LEN):
                for k in range(self.n_classes):
                    if int(self.y_data[i][j][k]) == 1:
                        classes[k] += 1
        print(classes)
        plt.pie(classes, labels=['noaction',
                                 'action1',
                                 'action2-1',
                                 'action2-2',
                                 'action3',
                                 'action4',
                                 'action5-1',
                                 'action5-2',
                                 'action6-1',
                                 'action6-2', ], autopct="%1.2f%%")
        plt.show()

    def getData(self, index):
        return self.x_data[index], self.y_data[index]

    def shuffle_data(self):
        random.shuffle(self.indexes)

    def generateOverlapDatasource(self, data):
        x_data = []
        y_data = []
        total_len = len(data.index)
        index = 0
        while index * self.OFFSET + self.SEQ_LEN <= total_len:
            x_vector, y_vector = self.generate_vector(data, index * self.OFFSET, self.label_method)
            if x_vector is not None and y_vector is not None:
                x_data.append(x_vector)
                y_data.append(y_vector)
            index += 1
        return np.array(x_data), np.array(y_data)

    def generate_vector(self, data, startIndex, method):
        if method == "segment":
            label = int(data.loc[startIndex]["label"])
            if self.config.get_rid_of_zero and label == 0:
                return None, None
            x_vector = [0.0 for _ in range(self.n_features)]
            y_vector = [0.0 for _ in range(self.n_classes)]
            y_vector[int(label)] = 1.0
            for i in range(0, self.SEQ_LEN):
                if label != int(data.loc[startIndex + i]["label"]):
                    return None, None
                x_vector[i * self.n_axis] = data.loc[startIndex + i]["acc_x"]
                x_vector[i * self.n_axis + 1] = data.loc[startIndex + i]["acc_y"]
                x_vector[i * self.n_axis + 2] = data.loc[startIndex + i]["acc_z"]
                x_vector[i * self.n_axis + 3] = data.loc[startIndex + i]["gyr_x"]
                x_vector[i * self.n_axis + 4] = data.loc[startIndex + i]["gyr_y"]
                x_vector[i * self.n_axis + 5] = data.loc[startIndex + i]["gyr_z"]
            return x_vector, y_vector
        elif method == "frame":
            x_vector = np.zeros((self.SEQ_LEN, self.n_axis))
            y_vector = np.zeros((self.SEQ_LEN, self.n_classes))
            for i in range(0, self.SEQ_LEN):
                x_vector[i][0] = data.loc[startIndex + i]["acc_x"]
                x_vector[i][1] = data.loc[startIndex + i]["acc_y"]
                x_vector[i][2] = data.loc[startIndex + i]["acc_z"]
                x_vector[i][3] = data.loc[startIndex + i]["gyr_x"]
                x_vector[i][4] = data.loc[startIndex + i]["gyr_y"]
                x_vector[i][5] = data.loc[startIndex + i]["gyr_z"]
                y_vector[i][int(data.loc[startIndex + i]["label"])] = 1.0
            return x_vector, y_vector

    def getOneBlockIndexes(self, index):
        offset = int((1 / self.k) * self.sample_nums)
        if index == self.k - 1:
            return self.indexes[(self.k - 1) * offset:]
        else:
            return self.indexes[(index * offset):((index + 1) * offset)]

    def getOtherBlocksIndexes(self, index):
        offset = int((1 / self.k) * self.sample_nums)
        if index == 0:
            return self.indexes[offset:]
        elif index == self.k - 1:
            return self.indexes[:(self.k - 1) * offset]
        else:
            return np.concatenate([self.indexes[:(index * offset)], self.indexes[(index + 1) * offset:]])

    def getWholeIndexes(self):
        return self.indexes


class SensorTrainDataset(Dataset):
    dataSource = None
    indexes = None
    sample_nums = None

    def __init__(self, dataSource, index):
        self.dataSource = dataSource
        self.indexes = dataSource.getOtherBlocksIndexes(index)
        self.sample_nums = len(self.indexes)

    def __getitem__(self, index):
        return self.dataSource.getData(self.indexes[index])

    def __len__(self):
        return self.sample_nums


class SensorEvalDataset(Dataset):
    dataSource = None
    indexes = None
    sample_nums = None

    def __init__(self, dataSource, index):
        self.dataSource = dataSource
        self.indexes = dataSource.getOneBlockIndexes(index)
        self.sample_nums = len(self.indexes)

    def __getitem__(self, index):
        return self.dataSource.getData(self.indexes[index])

    def __len__(self):
        return self.sample_nums


class SensorTestDataset(Dataset):
    dataSource = None
    indexes = None
    sample_nums = None

    def __init__(self, dataSource, index):
        self.dataSource = dataSource
        self.indexes = dataSource.getWholeIndexes(index)
        self.sample_nums = len(self.indexes)

    def __getitem__(self, index):
        return self.dataSource.getData(self.indexes[index])

    def __len__(self):
        return self.sample_nums


if __name__ == '__main__':
    config = Config()
    config.label_method = "frame"
    dataSource = SensorDatasource("./labelData.csv", config)
    dataSource.analyzeFrameLabel()
