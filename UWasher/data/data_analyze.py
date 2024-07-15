import os
import random

import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm

def normalize(data):
    def get_mean_std(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

    def normalize(data, mean, std):
        return (data - mean) / std
    for key in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
        mean, std = get_mean_std(data[key])
        data[key] = normalize(data[key], mean, std)

def get_data(path: os.path):
    data = {
        'acc_x': [[] for _ in range(9)], 'acc_y': [[] for _ in range(9)], 'acc_z': [[] for _ in range(9)],
        'gyr_x': [[] for _ in range(9)], 'gyr_y': [[] for _ in range(9)], 'gyr_z': [[] for _ in range(9)],
    }
    file_list = os.listdir(path)
    random.shuffle(file_list)
    for file_name in file_list:
        print("Now Processing %s ..." % file_name)
        csv_data = pd.read_csv(os.path.join(path, file_name))
        normalize(csv_data)
        for i in tqdm(range(len(csv_data.index))):
            if csv_data.loc[i]['label'] >= 1:
                for key in data.keys():
                    data[key][int(csv_data.loc[i]['label']) - 1].append(csv_data.loc[i][key])
    return data


if __name__ == '__main__':
    samsung_data_path = os.path.join("D:/WatchCollectData/shift_source")
    apple_data_path = os.path.join("C:/Users/SylarWu/Desktop/UWasher/rebuttal/shift_source")

    samsung_data = get_data(samsung_data_path)

    apple_data = get_data(apple_data_path)

    for key in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
        for action in range(9):
            plt.title("%s-action%d" % (key, action + 1))
            seaborn.distplot(samsung_data[key][action], color="blue")
            seaborn.distplot(apple_data[key][action], color="red")
            plt.savefig(os.path.join("./analyze_result", "%s-action%d" % (key, action + 1)))
            plt.show()