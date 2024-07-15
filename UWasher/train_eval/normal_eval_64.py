from UWasher.train_eval.utils import *
from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import normal_eval_procedure

if __name__ == "__main__":
    eval_config = EvalConfig()
    eval_config.mode = "normal"
    eval_config.index = 0
    eval_config.filename_model = "UWasher-490-epochs-TrainAcc-99.59-EvalAcc-86.86.pth"

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_64/"
    dataset_config.seq_len = 64

    chose_device = "0"

    gt, none, vote, smooth, both = normal_eval_procedure(eval_config, dataset_config, chose_device)

    print("None: ")
    showResult(gt, none, eval_config.n_classes)
    print("=======================================================================")
    print("Vote: ")
    showResult(gt, vote, eval_config.n_classes)
    print("=======================================================================")
    print("Smooth: ")
    showResult(gt, smooth, eval_config.n_classes)
    print("=======================================================================")
    print("Both: ")
    showResult(gt, both, eval_config.n_classes)
    print("=======================================================================")

    showStartEndErrors(gt, both, eval_config.n_classes)
    print("=======================================================================")

    showPersonWashScores(gt, both, eval_config.n_classes)
    print("=======================================================================")
    mat_gt = []
    mat_prediction = []
    for i in range(len(gt)):
        mat_gt.extend(gt[i])
        mat_prediction.extend(both[i])

    scio.savemat("./normal_for_confusion_matrix.mat", {"gt": np.array(mat_gt), "prediction": np.array(mat_prediction)})
