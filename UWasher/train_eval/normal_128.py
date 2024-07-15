from UWasher.train_eval.TrainConfig import TrainConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import train_eval_procedure

if __name__ == "__main__":
    train_config = TrainConfig()
    train_config.mode = "normal"
    train_config.index = 0

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_128/"
    dataset_config.seq_len = 128

    chose_device = "4"
    train_eval_procedure(train_config, dataset_config, chose_device)
