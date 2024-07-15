from UWasher.train_eval.utils import *
from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.train_eval.train_eval import normal_eval_procedure

if __name__ == "__main__":
    finest_model_names = readJSONFromFile("./ui_finest_model_names.json")["finest_model_names"]

    eval_config = EvalConfig()
    eval_config.mode = "user-independent"

    dataset_config = DatasetConfig()
    dataset_config.filename_datasource = "datasource_64/"
    dataset_config.seq_len = 64

    chose_device = "4"

    confusion = np.zeros((eval_config.n_classes, eval_config.n_classes), dtype=np.int32)

    label = []
    prediction = []

    result_path = "./ui_results/"

    for i in range(51):
        eval_config.index = i
        eval_config.filename_model = finest_model_names[i]

        print("Mode: %s-%d." % (eval_config.mode, eval_config.index))
        gt, none, vote, smooth, both = normal_eval_procedure(eval_config, dataset_config, chose_device)

        for j in range(len(gt)):
            label.append(gt[j])
            prediction.append(both[j])

            name = "%d-%d.mat" % (i, j)
            scio.savemat(result_path + name, {"label": gt[j],
                                              "none": none[j],
                                              "vote": vote[j],
                                              "smooth": smooth[j],
                                              "both": both[j]})

    showResult(label, prediction, eval_config.n_classes)

    showStartEndErrors(label, prediction, eval_config.n_classes)

    showPersonWashScores(label, prediction, eval_config.n_classes)
