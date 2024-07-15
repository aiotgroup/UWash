from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import normal_eval_procedure

if __name__ == "__main__":
    eval_config = EvalConfig()
    eval_config.mode = "normal"
    eval_config.index = 0

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_128/"
    dataset_config.seq_len = 128

    chose_device = "4"

    normal_eval_procedure(eval_config, dataset_config, chose_device)
