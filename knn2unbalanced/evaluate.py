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
                fp += 1
        else:
            if(predictLabels[i] == 1):
                fn += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def calcPrecision(tp, fp):
    precision = tp / (tp + fp)
    return precision


def calcRecall(tp, fn):
    recall = tp / (tp + fn)
    return recall


def calcAccuracy(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def calcTNR(tn, fp):
    tNR = tn / (tn + fp)
    return tNR


def calcFPR(fp, tn):
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


def plotROC():
    pass
