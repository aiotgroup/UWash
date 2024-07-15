from UWasher.train_eval.TrainConfig import TrainConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import train_eval_procedure

import argparse

parser = argparse.ArgumentParser(description='Recording videos and timestamps.')
parser.add_argument('-s', '--start', default=0, )
parser.add_argument('-e', '--end', default=0, )
parser.add_argument('-d', '--device', default=4, )

args = parser.parse_args()

if __name__ == "__main__":

    start = int(args.start)
    end = int(args.end)
    print(start, end)
    for i in range(start, end):
        train_config = TrainConfig()
        train_config.mode = "user-independent"
        train_config.index = i
        train_config.loss = "CEL"

        dataset_config = DatasetConfig()
        dataset_config.filename_datasource = "datasource_64/"
        dataset_config.seq_len = 64

        chose_device = str(args.device)
        train_eval_procedure(train_config, dataset_config, chose_device)
