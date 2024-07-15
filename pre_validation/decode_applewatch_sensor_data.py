from datetime import datetime
import time
import pandas as pd
import numpy as np
import json
import os.path
import logging

"""
数据处理流程:
    decode_apple_watch_sensor_data.py -> shift_data.py -> augment_data.py
    decode_sensor_data.py: 通过每个地点每个人的data.csv文件读取到
        实验对象开始对应洗手动作的开始帧和结束帧定位，已经将相应的原始
        加速度和陀螺仪数据打上相应动作的标签。最终生成个人的csv文件。
        在根目录的source/目录下。
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

    data = pd.read_table(path, sep="/n", names=['timestamp'])
    data['timestamp'] = data['timestamp'].apply(date_to_timestamp)
    return data["timestamp"].tolist()


def readSensorDataFromFile(path):
    def date_to_timestamp(x):
        timestamp = datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f+08:00')
        timestamp = int(time.mktime(timestamp.timetuple()) * 1000.0 + timestamp.microsecond / 1000.0)
        return timestamp

    source = pd.read_csv(path)
    source['loggingTime(txt)'] = source['loggingTime(txt)'].apply(date_to_timestamp)
    return source


def sliceDataByTimePeriod(data, start_time, end_time):
    i = approximateBiSearch(data['loggingTime(txt)'], start_time)
    j = approximateBiSearch(data['loggingTime(txt)'], end_time)
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
    base_path = os.path.join("C:/Users/SylarWu/Desktop/UWasher/rebuttal/")
    raw_path = os.path.join(base_path, "raw")
    generate_path = os.path.join(base_path, "source")

    filename_wait_to_assemble = "data.csv"
    filename_video = "video.avi"
    filename_video_timestamp = "time.txt"
    filename_data_timestamp = ""

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=generate_path + 'time_errors.log', level=logging.INFO, format=LOG_FORMAT)

    time_errors = []
    labels = {
        'action1': 1,
        'action2': 2,
        'action3': 3,
        'action4': 4,
        'action5': 5,
        'action6': 6,
        'action7': 7,
        'action8': 8,
        'action9': 9,
    }

    filename_persons = os.listdir(raw_path)
    for per_key in filename_persons:
        print("Now processing %s." % per_key)
        per_path = os.path.join(raw_path, per_key)
        if not os.path.exists(os.path.join(per_path, filename_wait_to_assemble)):
            continue
        # 读取已经对不同时间阶段打标签的数据
        timeSlot = pd.read_csv(os.path.join(per_path, filename_wait_to_assemble))
        len_actions = len(timeSlot.index)
        filename_data_timestamp = timeSlot['data_timestamp'][0]

        raw_data = readSensorDataFromFile(os.path.join(per_path, filename_data_timestamp))
        result = pd.DataFrame(columns=('acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'timestamp', 'label'))
        for i in range(len(raw_data.index)):
            result.loc[i] = [raw_data["accelerometerAccelerationX(G)"][i],
                             raw_data["accelerometerAccelerationY(G)"][i],
                             raw_data["accelerometerAccelerationZ(G)"][i],
                             raw_data["motionRotationRateX(rad/s)"][i],
                             raw_data["motionRotationRateY(rad/s)"][i],
                             raw_data["motionRotationRateZ(rad/s)"][i],
                             raw_data["loggingTime(txt)"][i], 0]

        # 读取视频帧对应时间的文件
        video_timestamp = readVideoTimestampFromFile(os.path.join(per_path, filename_video_timestamp))

        # 根据已标注的时间段对数据打标签
        for i in range(len_actions):
            startTime = video_timestamp[timeSlot.loc[i]["video_start_frame"] - 1]
            endTime = video_timestamp[timeSlot.loc[i]["video_end_frame"] - 1]

            begin, finish = sliceDataByTimePeriod(raw_data, startTime, endTime)

            print("Start time error: %f. End time error: %f." % (
                startTime - raw_data.loc[begin]["loggingTime(txt)"],
                endTime - raw_data.loc[finish]["loggingTime(txt)"]))
            logging.info("Start time error: %f. End time error: %f." % (
                startTime - raw_data.loc[begin]["loggingTime(txt)"],
                endTime - raw_data.loc[finish]["loggingTime(txt)"]))

            result.loc[begin:finish, 'label'] = labels[timeSlot.loc[i]["action_label"]]
        result.to_csv(os.path.join(generate_path, per_key + ".csv"), sep=',', index=False)
