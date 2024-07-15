import json
import numpy as np
import scipy.io as scio


def readJSONFromFile(path):
    with open(path, "r") as load_f:
        load_dict = json.load(load_f)
    return load_dict


def isOneSegment(result, start, end, n_classes):
    if end - start < 1000 or end - start > 3000:
        return False
    color_count = [0 for _ in range(n_classes)]
    for i in range(start, end + 1):
        color_count[result[i]] += 1
    return (np.sum(color_count[1:n_classes]) / np.sum(color_count)) >= 0.8


def isStartStart(result, n, start, gap=50):
    if start < gap or start + gap >= n:
        return False
    zero_count = 0
    color_count = 0
    for i in range(start - gap, start):
        if result[i] == 0:
            zero_count += 1
    for i in range(start, start + gap):
        if result[i] != 0:
            color_count += 1
    return zero_count >= int(gap * 0.8) and color_count >= int(gap * 0.8)


def isEndEnd(result, n, end, gap=50):
    if end - gap < 0 or end + gap >= n:
        return False
    zero_count = 0
    color_count = 0
    for i in range(end - gap, end):
        if result[i] != 0:
            color_count += 1
    for i in range(end, end + gap):
        if result[i] == 0:
            zero_count += 1
    return zero_count >= int(gap * 0.8) and color_count >= int(gap * 0.8)


def getStartEndIndex(result, n_classes):
    n = len(result)
    start = -1
    end = -1
    temp_start = []
    temp_end = []
    for i in range(1, n):
        if result[i - 1] == 0 and result[i] == 1:
            start = i
        if result[i - 1] == 9 and result[i] == 0:
            end = i - 1

        if start != -1 and isStartStart(result, n, start):
            temp_start.append(start)
            start = -1
        if end != -1 and isEndEnd(result, n, end):
            temp_end.append(end)
            end = -1
    start_end = []
    for start in temp_start:
        for end in temp_end:
            if isOneSegment(result, start, end, n_classes):
                start_end.append((start, end))
    return start_end


def calcStartEndErrors(start_end_1, start_end_2):
    return np.abs(start_end_1[0] - start_end_2[0]) + np.abs(start_end_1[1] - start_end_2[1])


def getConfusion(label, prediction, n_classes):
    n = len(label)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(n):
        confusion[prediction[i]][label[i]] += 1
    return confusion


def calcActionAccuracy(confusion):
    n_classes = confusion.shape[0]
    accuracy = [0 for i in range(n_classes)]
    for i in range(n_classes):
        accuracy[i] = confusion[i][i] / np.sum(confusion[i, :]) if confusion[i][i] != 0 else 0
    return accuracy


def getAccuracy(confusion):
    correct = 0
    n_classes = confusion.shape[0]
    for i in range(n_classes):
        correct += confusion[i][i]
    return correct / np.sum(confusion)


def getPrecisionRecallF1(confusion):
    n_classes = len(confusion)
    precision = [0 for _ in range(n_classes)]
    recall = [0 for _ in range(n_classes)]
    f1 = [0 for _ in range(n_classes)]

    for i in range(n_classes):
        precision[i] = confusion[i][i] / np.sum(confusion[i, :])
        recall[i] = confusion[i][i] / np.sum(confusion[:, i])

    for i in range(n_classes):
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return np.mean(precision), np.mean(recall), np.mean(f1)


def showResult(label, prediction, n_classes):
    n = len(label)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for i in range(n):
        temp_conf = getConfusion(label[i], prediction[i], n_classes)
        temp_acc = getAccuracy(temp_conf)
        temp_pre, temp_rec, temp_f1 = getPrecisionRecallF1(temp_conf)

        accuracy.append(temp_acc)
        precision.append(temp_pre)
        recall.append(temp_rec)
        f1.append(temp_f1)

        confusion += temp_conf

    mAccuracy = getAccuracy(confusion)
    mPrecision, mRecall, mF1 = getPrecisionRecallF1(confusion)
    print("Confusion: \n", confusion)

    print("Action Accuracy: ", calcActionAccuracy(confusion))
    print("mAccuracy: ", mAccuracy)
    print("Accuracy: ", accuracy)

    print("mPrecision: ", mPrecision)
    print("Precision: ", precision)

    print("mRecall: ", mRecall)
    print("Recall: ", recall)

    print("mF1: ", mF1)
    print("F1: ", f1)


def findPreStartEndByGT(gt_start_end, prediction, gap=100):
    n = len(prediction)
    result = []
    for i in range(len(gt_start_end)):
        gt_start, gt_end = gt_start_end[i]
        start = gt_start - gap
        end = gt_end - gap
        for j in range(gt_start - gap, gt_start + gap + 1):
            if j <= 0:
                continue
            if j >= n:
                break
            if prediction[j] != 0 and prediction[j - 1] == 0 and gt_start - j <= gt_start - start:
                start = j
        for j in range(gt_end - gap, gt_end + gap + 1):
            if j < 0:
                continue
            if j >= n - 1:
                break
            if prediction[j] != 0 and prediction[j + 1] == 0 and gt_end - j <= gt_end - end:
                end = j
        result.append((start, end))
    return result


def showStartEndErrors(label, prediction, n_classes):
    n = len(label)
    starts = []
    ends = []
    for i in range(n):
        gt_start_end = getStartEndIndex(label[i], n_classes)
        if len(gt_start_end) == 0:
            continue
        pre_start_end = findPreStartEndByGT(gt_start_end, prediction[i])
        print(gt_start_end, pre_start_end)
        for j in range(len(gt_start_end)):
            starts.append(np.abs(gt_start_end[j][0] - pre_start_end[j][0]))
            ends.append(np.abs(gt_start_end[j][1] - pre_start_end[j][1]))
    print("Start: Mean = %fs, Std = %fs" % (0.02 * np.mean(starts), 0.02 * np.std(starts)))
    print("End: Mean = %fs, Std = %fs" % (0.02 * np.mean(ends), 0.02 * np.std(ends)))


def calcWashScore(whole_action, n_classes):
    gesture_standard = [0, 4.9, 3.65, 3.65, 5.4, 4, 3.45, 3.45, 4.1, 4.1]
    one_seg_score = 100 / (n_classes - 1)
    gesture_count = np.zeros(n_classes)
    for i in range(len(whole_action)):
        gesture_count[whole_action[i]] += 1
    print("Lasting: ", np.array(gesture_count) * 0.02)
    score = []
    for i in range(1, n_classes):
        score.append(
            round(np.min((0.02 * gesture_count[i] * one_seg_score * (1 / gesture_standard[i]), one_seg_score)) * 9, 2))

    print("Scores: ", score)
    return np.sum(score) / 9


def showPersonWashScores(label, prediction, n_classes):
    n = len(label)
    gt_start_ends = []
    pre_start_ends = []
    for i in range(n):
        temp_gt = getStartEndIndex(label[i], n_classes)
        if len(temp_gt) == 0:
            continue
        temp_pre = findPreStartEndByGT(temp_gt, prediction[i])
        gt_start_ends.append(temp_gt)
        pre_start_ends.append(temp_pre)

    prediction_scores = np.zeros(n)
    label_scores = np.zeros(n)

    temp_pred_score = np.zeros(0)
    temp_gt_score = np.zeros(0)

    for i in range(len(gt_start_ends)):
        print("Index ", i)
        print("Prediction: ")
        score = 0
        for j in range(len(pre_start_ends[i])):
            temp_score = calcWashScore(prediction[i][pre_start_ends[i][j][0]:pre_start_ends[i][j][1] + 1], n_classes)
            temp_pred_score = np.append(temp_pred_score, temp_score)
            # temp_pred_score[j] = temp_score
            score += temp_score
        prediction_scores[i] = score / len(pre_start_ends[i])

        print("Label: ")
        score = 0
        for j in range(len(gt_start_ends[i])):
            temp_score = calcWashScore(label[i][gt_start_ends[i][j][0]:gt_start_ends[i][j][1] + 1], n_classes)
            temp_gt_score = np.append(temp_gt_score, temp_score)
            # temp_gt_score[j] = temp_score
            score += temp_score
        label_scores[i] = score / len(gt_start_ends[i])

    result_scores = np.abs(prediction_scores - label_scores)
    print(prediction_scores)
    print(label_scores)
    print("Scores Errors: ", result_scores)
    print("Scores Errors: Mean = %f. Std = %f." % (np.mean(result_scores), np.std(result_scores)))
    print("Temp Scores Errors: Mean = %f. Std = %f." % (np.mean(np.abs(temp_pred_score - temp_gt_score)),
                                                        np.std(np.abs(temp_pred_score - temp_gt_score))))


if __name__ == "__main__":
    data = [1, 2, 5, 6, 10, 7, 9, 8]
    print(np.argmax(data))
