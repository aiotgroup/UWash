import torch

from UWasher.train_eval.TrainConfig import TrainConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.model.SPPUnet import UWasher
from UWasher.train_eval.train_eval import pickup_training

if __name__ == "__main__":
    train_config = TrainConfig()
    train_config.mode = "user-independent"
    train_config.index = 50
    train_config.loss = "CEL"

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_64/"
    dataset_config.seq_len = 64

    chose_device = "4"
    model = UWasher()
    model.load_state_dict(torch.load("./checkpoint/user-independent-50-64/UWasher-150-epochs-EvalAcc-78.70.pth"))

    pickup_training(model, 150, train_config, dataset_config, chose_device)
