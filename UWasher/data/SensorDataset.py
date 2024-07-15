import os
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.utils import readJSONFromFile


class SensorDatasource():
    config = None

    source = {}
    label = {}
    locations = []
    persons = []
    eval_len = None

    def __init__(self, config=DatasetConfig()):
        self.config = config
        source_list = os.listdir(config.base_path + config.filename_datasource)

        for filename_source in source_list:
            person_info = filename_source[:filename_source.index(".mat")]
            mat_source = scio.loadmat(config.base_path + config.filename_datasource + filename_source)
            self.source[person_info] = mat_source

        self.label = readJSONFromFile(config.base_path + config.filename_labels)
        self.locations = list(readJSONFromFile(config.base_path + config.filename_locations).values())
        self.persons = list(readJSONFromFile(config.base_path + config.filename_persons).keys())

    def getLocationLength(self):
        return len(self.locations)

    def getPersonLength(self):
        return len(self.persons)

    def analyzeLabel(self):
        classes = [0 for i in range(self.config.n_classes)]
        for key, value in self.source.items():
            label = value["label"]
            for i in range(len(label)):
                for j in range(len(label[i])):
                    classes[label[i][j]] += 1
        plt.pie(classes, labels=['no action',
                                 'action1',
                                 'action2',
                                 'action3',
                                 'action4',
                                 'action5',
                                 'action6',
                                 'action7',
                                 'action8',
                                 'action9'], autopct="%1.2f%%")
        plt.savefig("./LabelDistribute.png", format='png')
        plt.show()

    def loadDataset(self, mode="normal", index=0):
        train_acc = []
        train_gyr = []
        train_label = []

        eval_acc = []
        eval_gyr = []
        eval_label = []

        test_ranges = {}
        test_data = {
            'acc': {},
            'gyr': {}
        }

        if mode == "normal":
            self.eval_len = [0]
            train_ratio = 0.8
            last_begin = 0
            for i, key in enumerate(self.persons):
                value = self.source[key]
                length = len(value["label"])
                part_index = int(length * train_ratio)
                train_acc.extend(value["accData"][:part_index])
                train_gyr.extend(value["gyrData"][:part_index])
                train_label.extend(value["label"][:part_index])

                eval_acc.extend(value["accData"][part_index:])
                eval_gyr.extend(value["gyrData"][part_index:])
                eval_label.extend(value["label"][part_index:])
                self.eval_len.append(len(eval_label))

                the_length = self.eval_len[len(self.eval_len) - 1] - self.eval_len[len(self.eval_len) - 2]

                test_ranges[key] = [last_begin, last_begin + the_length + 63]

                last_begin = last_begin + the_length + 63

                def generate_origin(data):
                    data = data.transpose(0, 2, 1)
                    ret = []
                    for piece in data:
                        ret.append(piece[0])
                    for piece in data[len(data) - 1][1:]:
                        ret.append(piece)
                    return np.array(ret)
                test_data['acc'][key] = generate_origin(value['accData'][part_index:])
                test_data['gyr'][key] = generate_origin(value['gyrData'][part_index:])

            scio.savemat("result_ranges.mat", test_ranges)
            scio.savemat("result_data.mat", test_data)
        elif mode == "user-independent":
            self.eval_len = [0]
            for i, key in enumerate(self.persons):
                value = self.source[key]
                if i == index:
                    eval_acc.extend(value["accData"])
                    eval_gyr.extend(value["gyrData"])
                    eval_label.extend(value["label"])
                    self.eval_len.append(len(eval_label))
                else:
                    train_acc.extend(value["accData"])
                    train_gyr.extend(value["gyrData"])
                    train_label.extend(value["label"])
        elif mode == "location-independent":
            self.eval_len = [0]
            for i, key in enumerate(self.persons):
                value = self.source[key]
                if key[:key.index("_")] == self.locations[index]:
                    eval_acc.extend(value["accData"])
                    eval_gyr.extend(value["gyrData"])
                    eval_label.extend(value["label"])
                    self.eval_len.append(len(eval_label))
                else:
                    train_acc.extend(value["accData"])
                    train_gyr.extend(value["gyrData"])
                    train_label.extend(value["label"])
        elif mode == "whole":
            self.eval_len = [0]
            for i, key in enumerate(self.persons):
                value = self.source[key]
                eval_acc.extend(value["accData"])
                eval_gyr.extend(value["gyrData"])
                eval_label.extend(value["label"])
                self.eval_len.append(len(eval_label))
            return None, NormalDataset(eval_acc, eval_gyr, eval_label)
        else:
            return None, None

        return NormalDataset(np.array(train_acc), np.array(train_gyr), np.array(train_label)), \
               NormalDataset(np.array(eval_acc), np.array(eval_gyr), np.array(eval_label))


class NormalDataset(Dataset):
    acc_source = None
    gyr_source = None
    label = None
    sample_nums = None

    def __init__(self, acc_source, gyr_source, label):
        self.acc_source = torch.from_numpy(acc_source).float()
        self.gyr_source = torch.from_numpy(gyr_source).float()
        self.label = torch.from_numpy(label).long()
        self.sample_nums = len(self.label)

    def __getitem__(self, index):
        return [self.acc_source[index], self.gyr_source[index]], self.label[index]

    def __len__(self):
        return self.sample_nums


if __name__ == '__main__':
    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_64/"
    dataset_config.seq_len = 64
    data_source = SensorDatasource(dataset_config)
    eval_config = EvalConfig()
    eval_config.mode = "normal"
    eval_config.index = 0
    eval_config.filename_model = "UWasher-500-epochs.pth"
    # for i in range(51):
    #     eval_config.index = i
    #     train_dataset, eval_dataset = data_source.loadDataset(mode=eval_config.mode, index=eval_config.index)
    print("==================================================================================================")
    eval_config.mode = "normal"
    eval_config.index = 0
    train_dataset, eval_dataset = data_source.loadDataset(mode=eval_config.mode, index=eval_config.index)
