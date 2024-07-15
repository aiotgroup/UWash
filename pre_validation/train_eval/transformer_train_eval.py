from pre_validation.model.Transformer import Transformer
from pre_validation.dataset.SensorDataset import SensorDatasource
from pre_validation.train_eval.util import *
from pre_validation.train_eval.train_eval import *


def train_preprocess(batch_x, batch_y, cuda):
    if cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    return batch_x, batch_y


def eval_preprocess(batch_x, batch_y, cuda):
    batch_y = batch_y.max(dim=2)[1]
    if cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    return batch_x, batch_y


def eval_postprocess(prediction, cuda):
    if cuda:
        prediction = prediction.cuda()
    return prediction.max(dim=2)[1]


if __name__ == '__main__':

    config = Config()
    config.label_method = "frame"
    sensor_data_source = SensorDatasource("../dataset/labelData.csv", config)
    print("数据加载完毕")

    # K次交叉验证，选择测试准确率最高的
    for k in range(config.k_fold):
        kth_trainning_eval(k_index=k,
                           sensor_data_source=sensor_data_source,
                           model=Transformer(config.n_axis, config.n_classes),
                           criterion=torch.nn.BCELoss(reduction="sum"),
                           train_preprocess=train_preprocess,
                           train_postprocess=None,
                           eval_preprocess=eval_preprocess,
                           eval_postprocess=eval_postprocess,
                           config=config)
