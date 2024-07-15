from UWasher.train_eval.TrainConfig import TrainConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import train_eval_procedure

if __name__ == "__main__":
    train_config = TrainConfig()
    train_config.mode = "normal"
    train_config.index = 0
    train_config.loss = "CEL"

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_64_normalize/"
    dataset_config.seq_len = 64

    chose_device = "1"
    train_eval_procedure(train_config, dataset_config, chose_device)
