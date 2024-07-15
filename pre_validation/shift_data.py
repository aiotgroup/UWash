from datetime import datetime
import time
import pandas as pd
import numpy as np
import json
import os.path
import logging


def readJSONFromFile(path):
    with open(path, "r") as load_f:
        load_dict = json.load(load_f)
    return load_dict


def shiftDataLabels(data, offset):
    length = len(data.index)
    if offset > 0:
        for i in range(length):
            if i + offset >= length:
                data["label"][i] = 0.0
                continue
            data["label"][i] = data["label"][i + offset]
    elif offset < 0:
        for i in range(length - 1, -1, -1):
            if i + offset < 0:
                data["label"][i] = 0.0
                continue
            data["label"][i] = data["label"][i + offset]
    return data


if __name__ == '__main__':

    base_path = "C:/Users/SylarWu/Desktop/UWasher/rebuttal/"
    source_path = base_path + "source/"
    shift_path = base_path + "shift_source/"
    offsets_path = base_path + "offsets.json"
    offsets = readJSONFromFile(offsets_path)
    for filename_source, offset in offsets.items():
        print("Now processing %s. Offset = %d." % (filename_source, offset))
        data = pd.read_csv(source_path + filename_source + ".csv")

        data = shiftDataLabels(data, offset)

        data.to_csv(shift_path + filename_source + ".csv", sep=',', index=False)
