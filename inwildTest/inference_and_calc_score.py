import os
import numpy as np
import torch
from collections import defaultdict
from SPPUnet import UWasher


def votePrediction(prediction, n_classes, n, seq_len=64, stride=1):
    voteResult = prediction.copy()
    for i in range(n):
        count_classes = [0 for _ in range(n_classes)]
        for k in range(stride):
            for j in range(seq_len):
                if i - j < 0 or k + j * stride >= seq_len:
                    break
                count_classes[prediction[i - j][k + j * stride]] += 1

        result = np.argmax(count_classes)

        for k in range(stride):
            for j in range(seq_len):
                if i - j < 0 or k + j * stride >= seq_len:
                    break
                voteResult[i - j][k + j * stride] = result
    for i in range(stride, seq_len):
        count_classes = [0 for _ in range(n_classes)]
        for j in range(seq_len):
            if i + j * stride >= seq_len or n - 1 - j < 0:
                break
            count_classes[prediction[n - 1 - j][i + j * stride]] += 1

        result = np.argmax(count_classes)

        for j in range(seq_len):
            if i + j * stride >= seq_len or n - 1 - j < 0:
                break
            voteResult[n - 1 - j][i + j * stride] = result
    return voteResult


def smoothPrediction(prediction, n_classes, gap=64):
    smoothResult = prediction.copy()
    length = len(smoothResult)
    for i in range(length):
        count_classes = [0 for _ in range(n_classes)]
        if i - gap < 0:
            for j in range(0, i + gap):
                count_classes[prediction[j]] += 1
        elif i + gap > length:
            for j in range(i - gap, length):
                count_classes[prediction[j]] += 1
        else:
            for j in range(i - gap, i + gap):
                count_classes[prediction[j]] += 1
        smoothResult[i] = np.argmax(count_classes)
    return smoothResult


def generateOrigin(prediction, n, seq_len, stride):
    origin = []
    for i in range(n):
        for k in range(stride):
            origin.append(prediction[i][k])
    for i in range(stride, seq_len):
        origin.append(prediction[n - 1][i])
    return np.array(origin, dtype=np.int32)


def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    acc_data = data[:, :3]
    gyr_data = data[:, 3:6]

    return acc_data, gyr_data


def slice_data(data, window_length, step):
    num_samples = data.shape[0]
    windows = []
    for start in range(0, num_samples - window_length + 1, step):
        window = data[start:start + window_length]
        windows.append(window)
    return np.array(windows)


def infer(model, acc_data, gyr_data):
    acc_data = torch.tensor(acc_data, dtype=torch.float32).permute(0, 2, 1).cuda()
    gyr_data = torch.tensor(gyr_data, dtype=torch.float32).permute(0, 2, 1).cuda()

    model.eval()
    with torch.no_grad():
        output = model((acc_data, gyr_data))
    return output


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def calcWashScore(whole_action, n_classes):
    gesture_standard = [0, 4.9, 3.65, 3.65, 5.4, 4, 3.45, 3.45, 4.1, 4.1]
    one_seg_score = 100 / (n_classes - 1)
    gesture_count = np.zeros(n_classes)
    for i in range(len(whole_action)):
        gesture_count[whole_action[i]] += 1
    score = []
    for i in range(1, n_classes):
        score.append(
            round(np.min((0.02 * gesture_count[i] * one_seg_score * (1 / gesture_standard[i]), one_seg_score)) * 9, 2))

    # print("Scores: ", score)
    return np.sum(score) / 9


def run_inference_and_calc_score(file_path, model_file_path, window_length=64, step=1):
    acc_data, gyr_data = load_data(file_path)

    acc_data = slice_data(acc_data, window_length, step)
    gyr_data = slice_data(gyr_data, window_length, step)

    model = UWasher()
    model.load_state_dict(torch.load(model_file_path))
    model.cuda()
    model.eval()

    result_prediction = []
    for i in range(acc_data.shape[0]):
        batch_x = acc_data[i, :, :]
        batch_y = gyr_data[i, :, :]

        batch_x = torch.tensor(batch_x, dtype=torch.float32).permute(1, 0).unsqueeze(0).cuda()
        batch_y = torch.tensor(batch_y, dtype=torch.float32).permute(1, 0).unsqueeze(0).cuda()

        prediction = model([batch_x, batch_y])

        prediction = prediction.squeeze(0).permute(1, 0)
        prediction = prediction.max(dim=1)[1]

        prediction = prediction.cpu().numpy()
        result_prediction.append(prediction)

    result_prediction = np.array(result_prediction)

    vote_prediction = votePrediction(result_prediction, 10, result_prediction.shape[0], seq_len=64, stride=1)
    vote_prediction = generateOrigin(vote_prediction, result_prediction.shape[0], seq_len=64, stride=1)
    prediction_numpy = smoothPrediction(vote_prediction, 10, 64)

    return (calcWashScore(prediction_numpy, 10))



if __name__ == '__main__':

    model_file_path = "./UWasher-TrainAcc-99.47-EvalAcc-87.18.pth"

    data_dir = 'inwildTest_dataset'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    grouped_files = defaultdict(list)

    for f in files:
        # 提取文件名中的组别部分 (例如 '1.1', '1.2', '1.3')
        group_key = f.split('_')[-1].split('.')[0] + '.' + f.split('_')[-1].split('.')[1]

        if group_key in ['1.1', '1.2', '1.3']:
            grouped_files[group_key].append(f)

    with open('scores_raw.txt', 'w') as score_file:
        for group, files_in_group in grouped_files.items():
            print(f"Group {group} len: {len(files_in_group)}")

            group_values = []

            for file in files_in_group:
                file_path = os.path.join(data_dir, file)
                score = run_inference_and_calc_score(file_path, model_file_path)
                print(file, "score:", score)
                score_file.write(f"{file} score: {score}\n")
                score_file.flush()
                group_values.append(score)

            overall_average = sum(group_values) / len(group_values) if group_values else 0
            print(f"Group {group} has {len(files_in_group)} files, average value: {overall_average:.2f}")





