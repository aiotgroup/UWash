import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from UWasher.model.SPPUnet import UWasher
from UWasher.train_eval.utils import *
from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.data.SensorDataset import NormalDataset
from UWasher.train_eval.train_eval import normal_eval_procedure, generateOrigin, votePrediction, smoothPrediction

def print_result(result_list, title):
    info = "%s: " % title
    for result in result_list:
        info += "& %.2f " % result
    info += "& %.2f " % np.mean(result_list)
    print(info)


result_accuracy = [0.8901683459540647, 0.871570796460177, 0.7819853193386842, 0.7246602951693648, 0.9337992953744744, 0.9512115685252736, 0.5723001446631863, 0.9203156056134434, 0.6986127864897467, 0.8630631858769]
result_start_mean = [0.096000, 0.328000, 0.536000, 0.580000, 0.212000, 0.080000, 1.264000, 0.132000, 0.392000, 0.240000]
result_start_std = [0.067409, 0.219308, 0.143889, 0.326435, 0.111427, 0.082946, 0.649233, 0.113561, 0.164730, 0.132665]
result_end_mean = [0.156000, 0.140000, 0.064000, 0.224000, 0.256000, 0.076000, 0.200000, 0.124000, 0.096000, 0.036000]
result_end_std = [0.097488, 0.040000, 0.055714, 0.231655, 0.097488, 0.038781, 0.137405, 0.114123, 0.094149, 0.029394]
result_score_mean = [10.945111, 11.479333, 16.776000, 23.849333, 5.189778, 3.834444, 36.832222, 5.705556, 21.564444, 6.856000]
result_score_std = [1.774530, 8.833742, 3.695091, 10.662517, 1.638226, 3.206277, 4.450522, 5.272460, 8.585347, 4.144249]

print_result(result_accuracy, "accuracy")
print_result(result_start_mean, "start_mean")
print_result(result_start_std, "start_std")
print_result(result_end_mean, "end_mean")
print_result(result_end_std, "end_std")
print_result(result_score_mean, "score_mean")
print_result(result_score_std, "score_std")

chose_device = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = chose_device

user_info = []
for i in range(1, 11):
    user_info.append("hospital_%d" % i)
n_classes = 10


def pickup_model(filename: str, mode: str, index: int):
    config = EvalConfig()
    config.filename_model = filename
    config.mode = mode
    config.index = index
    return config


eval_model_list = [
    pickup_model("UWasher-490-epochs-TrainAcc-99.59-EvalAcc-86.86.pth", "normal", 0),
]


def eval_preprocess(batch_x, batch_y, cuda):
    batch_acc, batch_gyr = batch_x
    if cuda:
        batch_acc = batch_acc.type(torch.FloatTensor).cuda()
        batch_gyr = batch_gyr.type(torch.FloatTensor).cuda()
        batch_y = batch_y.type(torch.LongTensor).cuda()
    else:
        batch_acc = batch_acc.type(torch.FloatTensor)
        batch_gyr = batch_gyr.type(torch.FloatTensor)
        batch_y = batch_y.type(torch.LongTensor)
    return (batch_acc, batch_gyr), batch_y


def eval_postprocess(prediction, cuda):
    return prediction.max(dim=1)[1]


def load_addition_data(datasource_path: os.path):
    addition_data_list = ["%s.mat" % info for info in user_info]

    result = {}

    for i in range(10):
        mat = scio.loadmat(os.path.join(datasource_path, addition_data_list[i]))
        if i == 0:
            mat['accData'] = mat['accData'][:15916, :, :]
            mat['gyrData'] = mat['gyrData'][:15916, :, :]
            mat['label'] = mat['label'][:15916, :]
        dataset = NormalDataset(mat['accData'], mat['gyrData'], mat['label'])
        result[user_info[i]] = dataset
    return result


if __name__ == "__main__":
    # datasource_path = "/data/wuxilei/WatchDataProcess/datasource_64/"
    # datasource_path = "D:/WatchCollectData/datasource_64/"
    datasets = load_addition_data(datasource_path)

    label = {
        info: [] for info in user_info
    }
    prediction = {
        info: [] for info in user_info
    }

    for eval_config in eval_model_list:
        for user_id in user_info:
            dataset = datasets[user_id]
            loader = DataLoader(dataset=dataset, batch_size=eval_config.batch_size, shuffle=False)
            model_path = eval_config.check_point_path + \
                         eval_config.mode + "-" + \
                         str(eval_config.index) + "-" + \
                         str(64) + "/"

            model = UWasher()
            model.load_state_dict(torch.load(model_path + eval_config.filename_model))
            model.eval()
            model_name = "UWasher"
            if eval_config.cuda:
                model = model.cuda()

            result_prediction = []
            result_label = []
            for batch_x, batch_y in tqdm(loader, desc='Evaluating'):
                with torch.no_grad():
                    batch_x, batch_y = eval_preprocess(batch_x, batch_y, eval_config.cuda)

                    prediction = model(batch_x)

                    prediction = eval_postprocess(prediction, eval_config.cuda)

                    result_prediction.extend(prediction.cpu().numpy())
                    result_label.extend(batch_y.cpu().numpy())

            result_label = np.array(result_label)
            result_prediction = np.array(result_prediction)

            result_gt = []
            result_none = []
            result_vote = []
            result_smooth = []
            result_both = []

            temp_label = result_label
            temp_prediction = result_prediction
            n = temp_label.shape[0]
            seq_len = temp_label.shape[1]
            stride = 1 if seq_len == 64 else 2

            temp_label = generateOrigin(temp_label, n, seq_len, stride)

            none_prediction = generateOrigin(temp_prediction, n, seq_len, stride)
            vote_prediction = votePrediction(temp_prediction, n_classes, n, seq_len, stride)
            vote_prediction = generateOrigin(vote_prediction, n, seq_len, stride)
            both_prediction = smoothPrediction(vote_prediction, gap=seq_len, n_classes=n_classes)
            smooth_prediction = smoothPrediction(none_prediction, gap=seq_len, n_classes=n_classes)

            result_gt.append(np.array(temp_label, dtype=np.int32))
            result_none.append(np.array(none_prediction, dtype=np.int32))
            result_vote.append(np.array(vote_prediction, dtype=np.int32))
            result_smooth.append(np.array(smooth_prediction, dtype=np.int32))
            result_both.append(np.array(both_prediction, dtype=np.int32))

            # showResult(result_gt, result_both, eval_config.n_classes)

            # showStartEndErrors(result_gt, result_both, eval_config.n_classes)

            showPersonWashScores(result_gt, result_both, eval_config.n_classes)
