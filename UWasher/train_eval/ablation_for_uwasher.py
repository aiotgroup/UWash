import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from UWasher.model.ablation_unet import AblationUNetConfig, UWasher
from UWasher.train_eval.TrainConfig import TrainConfig
from UWasher.train_eval.EvalConfig import EvalConfig
from UWasher.data.DatasetConfig import DatasetConfig
from UWasher.data.SensorDataset import SensorDatasource, NormalDataset
from UWasher.train_eval.utils import showResult, showStartEndErrors, showPersonWashScores


def train_one_epoch(loader, model, optimizer, criterion, cuda=True, preprocess=None, postprocess=None):
    # 计算这个epoch的总loss
    cost = 0.0
    for batch_x, batch_y in loader:
        # 如果需要对数据进行预处理
        if preprocess is not None:
            batch_x, batch_y = preprocess(batch_x, batch_y, cuda)
        # 初试化梯度
        optimizer.zero_grad()
        # 对结果进行预测
        prediction = model(batch_x)
        if postprocess is not None:
            prediction = postprocess(prediction, cuda)
        # 结果输入loss反向传播
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()

        cost += loss.item()
    return cost


# 对结果进行准确率计算
def eval(loader, model, cuda=True, preprocess=None, postprocess=None):
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        with torch.no_grad():
            # 如果需要对数据进行预处理
            if preprocess is not None:
                batch_x, batch_y = preprocess(batch_x, batch_y, cuda)

            prediction = model(batch_x)
            if postprocess is not None:
                prediction = postprocess(prediction, cuda)
            correct += prediction.eq(batch_y.data.long()).sum()
            temp_total = 1
            for temp_size in batch_y.size():
                temp_total *= temp_size
            total += temp_total
    accuracy = 100 * float(correct) / total
    return accuracy


def train_preprocess(batch_x, batch_y, cuda):
    batch_acc, batch_gyr = batch_x
    if cuda:
        batch_acc = batch_acc.type(torch.FloatTensor).cuda()
        batch_gyr = batch_gyr.type(torch.FloatTensor).cuda()
        batch_y = batch_y.type(torch.LongTensor).cuda()
    else:
        batch_acc = batch_acc.type(torch.FloatTensor)
        batch_gyr = batch_gyr.type(torch.FloatTensor)
        batch_y = batch_y.type(torch.LongTensor)
    return [batch_acc, batch_gyr], batch_y


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
    return [batch_acc, batch_gyr], batch_y


def eval_postprocess(prediction, cuda):
    return prediction.max(dim=1)[1]


def train_eval_procedure(model_name: str, train_config, dataset_config, chose_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = chose_device

    print("选择设备：cuda-%s" % chose_device)

    data_source = SensorDatasource(dataset_config)

    train_dataset, eval_dataset = data_source.loadDataset(train_config.mode, index=train_config.index)

    train_mode = train_config.mode + "-" + str(train_config.index) + "-" + str(dataset_config.seq_len)

    if train_dataset is None and eval_dataset is None:
        print("数据加载错误")
        exit(0)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_config.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=train_config.batch_size, shuffle=False)

    if not os.path.exists(train_config.check_point_path + train_mode):
        os.mkdir(train_config.check_point_path + train_mode)

    train_mode_path = train_config.check_point_path + train_mode + "/"

    print("数据加载完毕")

    model = UWasher(AblationUNetConfig(model_name.find('ds') >= 0,
                                       model_name.find('ppm') >= 0,
                                       model_name.find('se') >= 0))
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    if train_config.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("模型加载完毕")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, train_config.lr_down_ratio)

    # 保存每轮训练loss
    costs = []
    # 保存每轮训练后在测试集上准确率
    train_accs = []
    eval_accs = []
    max_eval_acc = 0
    print("开始训练")
    for epoch in range(train_config.num_epochs):
        # 首先训练
        model.train()
        cost = train_one_epoch(train_loader, model, optimizer, criterion, train_config.cuda,
                               preprocess=train_preprocess,
                               postprocess=None)
        costs.append(cost)
        scheduler.step()
        # 然后评估准确率
        model.eval()
        train_acc = None
        eval_acc = None
        if epoch % 10 == 0:
            eval_acc = eval(eval_loader, model, train_config.cuda,
                            preprocess=eval_preprocess,
                            postprocess=eval_postprocess)
            eval_accs.append(eval_acc)
            if epoch % 50 == 0:
                train_acc = eval(train_loader, model, train_config.cuda,
                                 preprocess=eval_preprocess,
                                 postprocess=eval_postprocess)
                train_accs.append(train_acc)
            # 保存100次epoch训练后准确率最高模型
            if epoch >= 100 and eval_acc > max_eval_acc:
                if eval_acc > max_eval_acc:
                    max_eval_acc = eval_acc
                torch.save(model.state_dict(),
                           train_mode_path + ("%s-%d-epochs-EvalAcc-%.2f.pth" % (
                               model_name, epoch, eval_acc)))
        info = "Model:%s. Train Mode:%s. Epoch:%d. Loss:%f." \
               % (model_name, train_mode, epoch, cost)

        if train_acc is not None:
            info += "Train Accuracy = %f%%." % train_acc
        if eval_acc is not None:
            info += "Eval Accuracy = %f%%." % eval_acc

        print(info)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.title("Model:%s Relation between Loss & Epochs" % model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot([_ for _ in range(1, len(costs) + 1)], costs)
    plt.savefig(train_mode_path + "Model-%s-loss-epoch.png" % model_name, format='png')
    plt.show()

    plt.title("Model:%s Relation between Accuracy on Train Set & Epochs" % model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot([_ for _ in range(1, len(train_accs) + 1)], train_accs)
    plt.savefig(train_mode_path + "Model-%s-train-acc-epoch.png" % model_name, format='png')
    plt.show()

    plt.title("Model:%s Relation between Accuracy on Eval Set & Epochs" % model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot([_ for _ in range(1, len(eval_accs) + 1)], eval_accs)
    plt.savefig(train_mode_path + "Model-%s-eval-acc-epoch.png" % model_name, format='png')
    plt.show()

    # 保存最终模型
    torch.save(model.state_dict(), train_mode_path + ("%s-%d-epochs.pth" % (model_name, train_config.num_epochs)))


def generateOrigin(prediction, n, seq_len, stride):
    origin = []
    for i in range(n):
        for k in range(stride):
            origin.append(prediction[i][k])
    for i in range(stride, seq_len):
        origin.append(prediction[n - 1][i])
    return np.array(origin, dtype=np.int32)


def votePrediction(prediction, n_classes, n, seq_len=64, stride=1):
    voteResult = prediction.copy()
    # 每一行前stride个做vote
    for i in range(n):
        count_classes = [0 for _ in range(n_classes)]
        for k in range(stride):
            # 斜线统计
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
    # 最后一行每一个做vote
    for i in range(stride, seq_len):
        count_classes = [0 for _ in range(n_classes)]
        # 斜线统计
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


def normal_eval_procedure(model_name, eval_config, dataset_config, chose_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = chose_device

    print("选择设备：cuda-%s" % chose_device)

    data_source = SensorDatasource(dataset_config)

    train_dataset, eval_dataset = data_source.loadDataset(mode=eval_config.mode, index=eval_config.index)

    if eval_dataset is None:
        print("数据加载错误")
        exit(0)

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=eval_config.batch_size, shuffle=False)

    eval_mode_path = eval_config.check_point_path + \
                     eval_config.mode + "-" + \
                     str(eval_config.index) + "-" + \
                     str(dataset_config.seq_len) + "/"

    print("数据加载完毕")

    model = UWasher(AblationUNetConfig(model_name.find('ds') >= 0,
                                       model_name.find('ppm') >= 0,
                                       model_name.find('se') >= 0))

    model.load_state_dict(torch.load(eval_mode_path + eval_config.filename_model))
    model.eval()

    if eval_config.cuda:
        model = model.cuda()

    result_prediction = []
    result_label = []

    for batch_x, batch_y in tqdm(eval_loader, desc='Evaluating'):
        with torch.no_grad():
            batch_x, batch_y = eval_preprocess(batch_x, batch_y, eval_config.cuda)

            prediction = model(batch_x)

            prediction = eval_postprocess(prediction, eval_config.cuda)

            result_prediction.extend(prediction.cpu().numpy())
            result_label.extend(batch_y.cpu().numpy())

    result_label = np.array(result_label)
    result_prediction = np.array(result_prediction)
    indexes = data_source.eval_len

    result_gt = []
    result_none = []
    result_vote = []
    result_smooth = []
    result_both = []

    for i in range(len(indexes) - 1):
        temp_label = result_label[indexes[i]:indexes[i + 1], :]
        temp_prediction = result_prediction[indexes[i]:indexes[i + 1], :]
        n = temp_label.shape[0]
        seq_len = temp_label.shape[1]
        stride = 1 if seq_len == 64 else 2

        temp_label = generateOrigin(temp_label, n, seq_len, stride)

        none_prediction = generateOrigin(temp_prediction, n, seq_len, stride)
        vote_prediction = votePrediction(temp_prediction, dataset_config.n_classes, n, seq_len, stride)
        vote_prediction = generateOrigin(vote_prediction, n, seq_len, stride)
        both_prediction = smoothPrediction(vote_prediction, gap=seq_len, n_classes=dataset_config.n_classes)
        smooth_prediction = smoothPrediction(none_prediction, gap=seq_len, n_classes=dataset_config.n_classes)

        result_gt.append(np.array(temp_label, dtype=np.int32))
        result_none.append(np.array(none_prediction, dtype=np.int32))
        result_vote.append(np.array(vote_prediction, dtype=np.int32))
        result_smooth.append(np.array(smooth_prediction, dtype=np.int32))
        result_both.append(np.array(both_prediction, dtype=np.int32))
    return result_gt, result_none, result_vote, result_smooth, result_both


if __name__ == '__main__':
    is_train = True
    model_name = 'ablation_unet'
    if is_train:
        train_config = TrainConfig()
        train_config.mode = "normal"
        train_config.index = 0
        train_config.loss = "CEL"

        dataset_config = DatasetConfig()
        dataset_config.filename_datasource = "datasource_64/"
        dataset_config.seq_len = 64

        chose_device = "3"
        train_eval_procedure(model_name, train_config, dataset_config, chose_device)
    else:
        eval_config = EvalConfig()
        eval_config.mode = "normal"
        eval_config.index = 0
        eval_config.filename_model = "%s-300-epochs.pth" % model_name

        dataset_config = DatasetConfig()
        dataset_config.filename_datasource = "datasource_64/"
        dataset_config.seq_len = 64

        chose_device = "3"

        gt, none, vote, smooth, both = normal_eval_procedure(model_name, eval_config, dataset_config, chose_device)

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

        showStartEndErrors(gt, none, eval_config.n_classes)
        print("=======================================================================")

        showPersonWashScores(gt, none, eval_config.n_classes)
        print("=======================================================================")
