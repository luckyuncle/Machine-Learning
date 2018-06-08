def calcMixMatrix(predictLabels, realLabels):
    length = min(len(predictLabels), len(realLabels))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(length):
        if(realLabels[i] == 1):
            if(predictLabels[i] == 1):
                tp += 1
            else:
                fn += 1
        else:
            if(predictLabels[i] == 1):
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def calcPrecision(tp, fp):
    if tp + fp == 0:
        return 0
    precision = tp / (tp + fp)
    return precision


def calcRecall(tp, fn):
    if tp + fn == 0:
        return 0
    recall = tp / (tp + fn)
    return recall


def calcAccuracy(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def calcTNR(tn, fp):
    if tn + fp == 0:
        return 0
    tNR = tn / (tn + fp)
    return tNR


def calcFPR(fp, tn):
    if tn + fp == 0:
        return 0
    fPR = fp / (tn + fp)
    return fPR


def calcG_mean(tp, tn, fp, fn):
    g_mean = (calcTNR(tn, fp) * calcPrecision(tp, fp))**0.5
    return g_mean


def calcF_measure(tp, fp, fn):
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure


def clacAUC(tprs, fprs):
    from sklearn.metrics import auc
    return auc(tprs, fprs)


def evaluateClassifier(predictLabels, realLabels):
    tp, tn, fp, fn = calcMixMatrix(predictLabels, realLabels)
    '''print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " +
          str(fn))'''
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    accuracy = calcAccuracy(tp, tn, fp, fn)
    '''plt.figure(figsize=(6, 6))
    plt.axis([0, 1, 0, 1])
    plt.plot(fpr, tpr)
    plt.show()'''
    '''print("正确率：" + str(precision))
    print("召回率：" + str(recall))
    print("准确率：" + str(accuracy))'''
    return precision, recall, accuracy


def output(precisions, recalls, accuracys):
    precisionSum = 0
    recallSum = 0
    accuracySum = 0
    for preNum in precisions:
        precisionSum += preNum
    for recNum in recalls:
        recallSum += recNum
    for accNum in accuracys:
        accuracySum += accNum
    precision = precisionSum / len(precisions)
    recall = recallSum / len(recalls)
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    print("正确率：" + str(precision))
    print("召回率：" + str(recall))
    print("准确率：" + str(accuracySum / len(accuracys)))
    print("f-measure：" + str(f_measure))
    return precision, recall, f_measure
