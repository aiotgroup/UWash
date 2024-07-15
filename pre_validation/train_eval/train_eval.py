import torch
import matplotlib.pyplot as plt

from pre_validation.dataset.SensorDataset import SensorTrainDataset
from pre_validation.dataset.SensorDataset import SensorEvalDataset
from pre_validation.train_eval.config import Config


def train(loader, model, optimizer, criterion, cuda=True, preprocess=None, postprocess=None):
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


def kth_trainning_eval(k_index, sensor_data_source, model, criterion,
                       train_preprocess=None, train_postprocess=None,
                       eval_preprocess=None, eval_postprocess=None,
                       config=Config()):
    model_name = model.get_model_name()
    save_prefix = "./checkpoint/" + model_name + "/"

    sensor_train_dataset = SensorTrainDataset(sensor_data_source, k_index)
    train_loader = DataLoader(dataset=sensor_train_dataset, batch_size=config.batch_size, shuffle=True)

    sensor_eval_dataset = SensorEvalDataset(sensor_data_source, k_index)
    eval_loader = DataLoader(dataset=sensor_eval_dataset, batch_size=config.batch_size, shuffle=False)

    if config.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    # 保存每轮训练loss
    costs = []
    # 保存每轮训练后在测试集上准确率
    accuracies = []
    max_acc = 0
    for epoch in range(config.epochs):
        # 首先训练
        model.train()
        cost = train(train_loader, model, optimizer, criterion, config.cuda,
                     preprocess=train_preprocess,
                     postprocess=train_postprocess)
        costs.append(cost)
        # 然后评估准确率
        model.eval()
        accuracy = eval(eval_loader, model, config.cuda,
                        preprocess=eval_preprocess,
                        postprocess=eval_postprocess)
        accuracies.append(accuracy)
        print("Model:%s. %dth training epoch is done! Now cost is %f, and accuracy on test set is %f." %
              (model_name, epoch, cost, accuracy))
        # 保存100次epoch训练后准确率最高模型
        if epoch >= 100 and accuracy > max_acc:
            max_acc = accuracy
            torch.save(model.state_dict(),
                       save_prefix + ("%s-%dth-%d-epochs-Acc-%.2f.pth" % (model_name, k_index, epoch, accuracy)))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.title("Model:%s 第%d折训练loss与epoch间关系" % (model_name, k_index))
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot([_ for _ in range(1, len(costs) + 1)], costs)
    plt.savefig(save_prefix + "Model-%s-%dth-loss-epoch.png" % (model_name, k_index), format='png')
    plt.show()

    plt.title("Model:%s 第%d折训练后在测试验证集上准确率与epoch间关系" % (model_name, k_index))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot([_ for _ in range(1, len(accuracies) + 1)], accuracies)
    plt.savefig(save_prefix + "Model-%s-%dth-acc-epoch.png" % (model_name, k_index), format='png')
    plt.show()

    # 保存第K次最终模型
    torch.save(model.state_dict(), save_prefix + ("%s-%dth-%d-epochs.pth" % (model_name, k_index, config.epochs)))
