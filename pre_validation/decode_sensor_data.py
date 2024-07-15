from datetime import datetime
import time
import pandas as pd
import numpy as np
import json
import os.path
import logging

"""

数据处理流程:
    decode_sensor_data.py -> shift_data.py -> augment_data.py
    decode_sensor_data.py: 通过每个地点每个人的data.csv文件读取到
        实验对象开始对应洗手动作的开始帧和结束帧定位，已经将相应的原始
        加速度和陀螺仪数据打上相应动作的标签。最终生成个人的csv文件。
        在根目录的source/目录下。
        
    shift_data.py: 由于设备与手表有时间上的误差，所以通过观察得出各个
        人的offset，然后将source/目录下的数据文件加上offset。
        
    augment_data.py: 此时数据已经无误，数据增强64倍，以序列长度为64
        或128。

"""


def readJSONFromFile(path):
    with open(path, "r") as load_f:
        load_dict = json.load(load_f)
    return load_dict


def readVideoTimestampFromFile(path):
    def date_to_timestamp(x):
        timestamp = datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        timestamp = int(time.mktime(timestamp.timetuple()) * 1000.0 + timestamp.microsecond / 1000.0)
        return timestamp

    data = pd.read_table(path, sep="\n", names=['timestamp'])
    data['timestamp'] = data['timestamp'].apply(date_to_timestamp)
    return data["timestamp"].tolist()


def readSensorDataFromFile(path):
    def date_to_timestamp(x):
        timestamp = datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f')
        timestamp = int(time.mktime(timestamp.timetuple()) * 1000.0 + timestamp.microsecond / 1000.0)
        return timestamp

    data = pd.read_table(path, sep=" ", names=['x', 'y', 'z', 'timestamp'])
    data['timestamp'] = data['timestamp'].apply(date_to_timestamp)
    return data


def approximateSync(acc_data, gyr_data):
    """
    将加速度和陀螺仪数据进行大致同步
    :param acc_data: 加速度数据
    :param gyr_data: 陀螺仪数据
    :return: 两者同步后的数据
    """
    data = pd.DataFrame(columns=('acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'timestamp', 'label'))
    acc_len = len(acc_data.index)
    gyr_len = len(gyr_data.index)
    i, j, k = 0, 0, 0
    while i < acc_len and j < gyr_len:
        if np.abs(acc_data["timestamp"][i] - gyr_data["timestamp"][j]) <= 20:

            data.loc[k] = [acc_data["x"][i], acc_data["y"][i], acc_data["z"][i],
                           gyr_data["x"][j], gyr_data["y"][j], gyr_data["z"][j],
                           acc_data["timestamp"][i], 0]

            i += 1
            j += 1
            k += 1
        elif acc_data["timestamp"][i] > gyr_data["timestamp"][j]:
            j += 1
        else:
            i += 1
    return data


def sliceDataByTimePeriod(data, start_time, end_time):
    i = approximateBiSearch(data['timestamp'], start_time)
    j = approximateBiSearch(data['timestamp'], end_time)
    return i, j


def approximateBiSearch(data, target):
    n = len(data)
    begin = 0
    end = n - 1
    while begin <= end:
        middle = (begin + end) // 2
        if np.abs(data[middle] - target) <= 20:
            return middle
        if data[middle] > target:
            end = middle - 1
        else:
            begin = middle + 1
    if begin < 0 or end < 0:
        return 0
    elif begin >= n or end >= n:
        return n - 1
    else:
        return begin if np.abs(data[begin] - target) < np.abs(data[end] - target) else end


if __name__ == '__main__':
    """
        通过视频标注生成加速度和陀螺仪数据的标签
        生成的标签文件有误差，需要加上offset
    """
    base_path = "D:/WatchCollectData/"
    raw_path = base_path + "origin_data_raw/"
    generate_path = base_path + "source/"
    filename_label = "label_action.json"
    filename_location = "location_info.json"
    filename_person = "person_info.json"
    filename_wait_to_assemble = "data.csv"

    filename_video = "video.avi"
    filename_video_timestamp = "time.txt"
    filename_acc_timestamp = "Others/accData.txt"
    filename_gyr_timestamp = "Others/gyrData.txt"

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=generate_path + 'time_errors.log', level=logging.INFO, format=LOG_FORMAT)

    labels = readJSONFromFile(base_path + filename_label)
    locations = readJSONFromFile(base_path + filename_location)

    time_errors = []

    for loc_key, loc_value in locations.items():
        print("Now processing %s->%s." % (loc_key, loc_value))
        loc_path = raw_path + loc_key + "/"
        if not os.path.exists(loc_path):
            continue
        persons = readJSONFromFile(loc_path + filename_person)
        for per_key, per_value in persons.items():
            print("Now processing %s->%s." % (per_key, per_value))
            per_path = loc_path + per_key + "/"
            if not os.path.exists(per_path):
                continue

            # 读取已经对不同时间阶段打标签的数据
            timeSlot = pd.read_csv(per_path + filename_wait_to_assemble)
            len_actions = len(timeSlot.index)

            # 读取加速度和陀螺仪数据并将二者整合，此时所有的标签为0
            accData = readSensorDataFromFile(per_path + filename_acc_timestamp)
            gyrData = readSensorDataFromFile(per_path + filename_gyr_timestamp)
            assembleData = approximateSync(accData, gyrData)

            # 读取视频帧对应时间的文件
            video_timestamp = readVideoTimestampFromFile(per_path + filename_video_timestamp)

            # 根据已标注的时间段对数据打标签
            for i in range(len_actions):
                startTime = video_timestamp[timeSlot.loc[i]["video_start_frame"] - 1]
                endTime = video_timestamp[timeSlot.loc[i]["video_end_frame"] - 1]

                begin, finish = sliceDataByTimePeriod(assembleData, startTime, endTime)

                print("Start time error: %f. End time error: %f." % (
                    startTime - assembleData.loc[begin]["timestamp"], endTime - assembleData.loc[finish]["timestamp"]))
                logging.info("Start time error: %f. End time error: %f." % (
                    startTime - assembleData.loc[begin]["timestamp"], endTime - assembleData.loc[finish]["timestamp"]))

                assembleData.loc[begin:finish, 'label'] = labels[timeSlot.loc[i]["action_label"]]

            assembleData.to_csv(generate_path + per_key + ".csv", sep=',', index=False)
