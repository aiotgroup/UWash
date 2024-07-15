from datetime import datetime
import time
import pandas as pd
import numpy as np
import json
import os.path
import logging
import scipy.io as scio


def cropData(data, gap=150):
    length = len(data.index)
    begin = 0
    end = length - 1
    while int(data["label"][begin]) == 0:
        begin += 1
    while int(data["label"][end]) == 0:
        end -= 1
    begin = 0 if begin - gap < 0 else begin - gap
    end = length - 1 if end + gap >= length else end + gap

    return data.loc[begin: end + 1].reset_index()


def generateOneSample(data, startIndex, length, seq_len):
    if startIndex + seq_len >= length:
        return None
    acc_data = data[["acc_x", "acc_y", "acc_z"]][startIndex:startIndex + seq_len]
    acc_data = acc_data.to_numpy(dtype=np.float32)
    acc_data = acc_data.T
    gyr_data = data[["gyr_x", "gyr_y", "gyr_z"]][startIndex:startIndex + seq_len]
    gyr_data = gyr_data.to_numpy(dtype=np.float32)
    gyr_data = gyr_data.T
    label_data = data["label"][startIndex:startIndex + seq_len]
    label_data = label_data.to_numpy(dtype=np.int32)
    return acc_data, gyr_data, label_data


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

    def swap(data, key_1, key_2):
        temp = data[key_1]
        data[key_1] = data[key_2]
        data[key_2] = temp

    # data['acc_x'] = -data['acc_x']
    # data['acc_y'] = -data['acc_y']
    # data['acc_z'] = -data['acc_z']


if __name__ == "__main__":
    is_normalize = True
    for seq_length, stride in zip([64], [1]):
        base_path = "C:/Users/SylarWu/Desktop/UWasher/rebuttal/"
        shift_path = base_path + "shift_source/"
        target_path = base_path + "datasource_" + str(seq_length) + "_normalize/"

        shift_source_list = os.listdir(shift_path)

        for shift_source in shift_source_list:
            print("Now processing %s." % shift_source)
            person_info = shift_source[:shift_source.index(".csv")]
            data = pd.read_csv(shift_path + shift_source)
            if is_normalize:
                normalize(data)
            data = cropData(data)
            length = len(data.index)
            acc_target = []
            gyr_target = []
            label_target = []
            for i in range(0, length, stride):
                temp = {}
                result = generateOneSample(data, i, length, seq_length)
                if result is None:
                    break
                acc_target.append(result[0])
                gyr_target.append(result[1])
                label_target.append(result[2])
            acc_target = np.array(acc_target)
            gyr_target = np.array(gyr_target)
            label_target = np.array(label_target)
            scio.savemat(target_path + person_info + ".mat", {"accData": acc_target,
                                                              "gyrData": gyr_target,
                                                              "label": label_target})
