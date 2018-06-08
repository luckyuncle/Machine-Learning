from numpy import tile, zeros, argsort
import evaluate
import operator


# 构造kNN类别加权分类器
def classify0(inX, dataSet, labels, k):
    # 输入向量inX, 输入训练样本集dataSet, 标签向量labels, 选择最近邻居的数目k
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # print(1 / (distances + 0.5))
    sims = 1 / (0.001 + distances)
    nSumSims = 0.0
    nSum = 0.0
    pSumSims = 0.0
    pSum = 0.0
    for i in range(len(labels)):
        if labels[i] == 1:
            pSumSims += sims[i]
            pSum += 1
        else:
            nSumSims += sims[i]
            nSum += 1
    pWeight = pSumSims / pSum
    nWeight = nSumSims / nSum
    for i in range(len(sims)):
        if labels[i] == 1:
            sims[i] *= pWeight
        else:
            sims[i] *= nWeight
    # sims = (1 / (distances + 0.5))**2
    # print(sims)
    sortedDistIndicies = argsort(-sims)
    # classCount = {}
    nSumWeights = 0.0
    pSumWeights = 0.0
    for i in range(k):
        if labels[sortedDistIndicies[i]] == 1:
            pSumWeights += sims[sortedDistIndicies[i]]
        else:
            nSumWeights += sims[sortedDistIndicies[i]]
    if nSumWeights > pSumWeights:
        return -1
    else:
        return 1


# 构造kNN类别2加权分类器
def classify4(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sims = 1 / (0.001 + distances)
    nNumber = 0
    pNumber = 0
    for label in labels:
        if label == 1:
            pNumber += 1
        else:
            nNumber += 1
    nWeight = nNumber / len(labels)
    pWeight = pNumber / len(labels)
    for i in range(len(labels)):
        if labels[i] == 1:
            sims[i] *= pWeight
        else:
            sims[i] *= nWeight
    sortedDistIndicies = argsort(-sims)
    nSumWeights = 0.0
    pSumWeights = 0.0
    for i in range(k):
        if labels[sortedDistIndicies[i]] == 1:
            pSumWeights += sims[sortedDistIndicies[i]]
        else:
            nSumWeights += sims[sortedDistIndicies[i]]
    if nSumWeights > pSumWeights:
        return -1
    else:
        return 1


# 构造kNN样本加权分类器
def classify1(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sims = 1 / (0.001 + distances)
    sims = sims * sims
    sortedDistIndicies = argsort(-sims)
    # classCount = {}
    nSumWeights = 0.0
    pSumWeights = 0.0
    for i in range(k):
        if labels[sortedDistIndicies[i]] == 1:
            pSumWeights += sims[sortedDistIndicies[i]]
        else:
            nSumWeights += sims[sortedDistIndicies[i]]
    if nSumWeights > pSumWeights:
        return -1
    else:
        return 1


# 构造kNN类别和样本加权分类器
def classify2(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sims = 1 / (0.001 + distances)
    tmpSims = 1 / (0.001 + distances)
    nSumSims = 0.0
    nSum = 0.0
    pSumSims = 0.0
    pSum = 0.0
    for i in range(len(labels)):
        if labels[i] == 1:
            pSumSims += sims[i]
            pSum += 1
        else:
            nSumSims += sims[i]
            nSum += 1
    pWeight = pSumSims / pSum
    nWeight = nSumSims / nSum
    for i in range(len(sims)):
        if labels[i] == 1:
            sims[i] *= pWeight
        else:
            sims[i] *= nWeight
    sims *= tmpSims
    sortedDistIndicies = argsort(-sims)
    nSumWeights = 0.0
    pSumWeights = 0.0
    for i in range(k):
        if labels[sortedDistIndicies[i]] == 1:
            pSumWeights += sims[sortedDistIndicies[i]]
        else:
            nSumWeights += sims[sortedDistIndicies[i]]
    if nSumWeights > pSumWeights:
        return -1
    else:
        return 1


# 原kNN分类器
def classify3(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 处理数据，读取文件转化为NumPy
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    tmpNum = 0
    features = []
    for line in arrayOLines:
        if line[0] == '@':
            tmpNum += 1
            if line.split(' ')[0] == '@inputs':
                line = line.replace(',', '')
                features = line.strip('\n').split(' ')[1:]
        else:
            break
    numberOfLines = len(arrayOLines) - tmpNum
    featureNum = len(arrayOLines[-2].split(',')) - 1
    returnMat = zeros((numberOfLines, featureNum))
    classLabelVector = []
    index = 0
    for line in arrayOLines[tmpNum:]:
        line = ''.join(line.split())
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[0:featureNum]
        if (listFromLine[-1] == "negative"):
            classLabelVector.append(-1)
        else:
            classLabelVector.append(1)
        index += 1
    return returnMat, classLabelVector, features


# 归一化数值
def autoNorm(minVals, maxVals, dataSet):
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet


# 交叉验证方法
def crossAuth(dataMat, labels, i):
    length = int(len(labels) / 10)
    '''
    trainingMat = dataMat[0:length * 8, :]
    hmLabels = labels[0:length * 8]
    testMat = dataMat[length * 8:, :]
    chmLabels = labels[length * 8:]
    '''
    featureNum = dataMat.shape[1]  # 行数
    trainingMat = zeros((dataMat.shape[0] - length, featureNum))
    testMat = zeros((length, featureNum))
    hmLabels = []
    chmLabels = []
    trainingMat[0:i * length, :] = dataMat[0:i * length, :]
    trainingMat[i * length:, :] = dataMat[(i + 1) * length:, :]
    testMat[0:length, :] = dataMat[i * length:(i + 1) * length, :]
    hmLabels[0:i * length] = labels[0:i * length]
    hmLabels[i * length:] = labels[(i + 1) * length:]
    chmLabels[0:length] = labels[i * length:(i + 1) * length]
    return trainingMat, testMat, hmLabels, chmLabels


# 测试IPC_KNN
def testIPKNN(fileName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for j in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, j)
        # 根据情况决定是否归一化
        minVals = normTrainingMat.min(0)
        maxVals = normTrainingMat.max(0)
        normTrainingMat = autoNorm(minVals, maxVals, normTrainingMat)
        normTestMat = autoNorm(minVals, maxVals, normTestMat)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            classifierResult = classify0(normTestMat[i, :], normTrainingMat,
                                         hmLabels, kValue)
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluateClassifier(
                predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试IPS_KNN
def testIP2KNN(fileName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for j in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, j)
        minVals = normTrainingMat.min(0)
        maxVals = normTrainingMat.max(0)
        normTrainingMat = autoNorm(minVals, maxVals, normTrainingMat)
        normTestMat = autoNorm(minVals, maxVals, normTestMat)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            classifierResult = classify1(normTestMat[i, :], normTrainingMat,
                                         hmLabels, kValue)
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluateClassifier(
                predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试IPCS_KNN
def testIP3KNN(fileName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for j in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, j)
        minVals = normTrainingMat.min(0)
        maxVals = normTrainingMat.max(0)
        normTrainingMat = autoNorm(minVals, maxVals, normTrainingMat)
        normTestMat = autoNorm(minVals, maxVals, normTestMat)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            classifierResult = classify2(normTestMat[i, :], normTrainingMat,
                                         hmLabels, kValue)
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluateClassifier(
                predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试IPCS_KNN
def testIP4KNN(fileName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for j in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, j)
        minVals = normTrainingMat.min(0)
        maxVals = normTrainingMat.max(0)
        normTrainingMat = autoNorm(minVals, maxVals, normTrainingMat)
        normTestMat = autoNorm(minVals, maxVals, normTestMat)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            classifierResult = classify4(normTestMat[i, :], normTrainingMat,
                                         hmLabels, kValue)
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluateClassifier(
                predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试KNN
def testKNN(fileName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for j in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, j)
        minVals = normTrainingMat.min(0)
        maxVals = normTrainingMat.max(0)
        normTrainingMat = autoNorm(minVals, maxVals, normTrainingMat)
        normTestMat = autoNorm(minVals, maxVals, normTestMat)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            classifierResult = classify3(
                normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluateClassifier(
                predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试SVM
def testSVM(fileName):
    from sklearn.svm import SVC
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        model = SVC(kernel='rbf')  # kernel='linear'
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluateClassifier(
            predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试随机森林
def testRandomForest(fileName):
    from sklearn.ensemble import RandomForestClassifier
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        model = RandomForestClassifier(n_estimators=8)
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluateClassifier(
            predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试logistics回归
def testLogistic(fileName):
    from sklearn.linear_model import LogisticRegression
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        model = LogisticRegression(penalty='l2')
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluateClassifier(
            predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试朴素贝叶斯
def testBayes(fileName):
    # MultinomialNB
    from sklearn.naive_bayes import GaussianNB
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        # model = MultinomialNB(alpha=0.01)
        model = GaussianNB()
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluateClassifier(
            predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


# 测试决策树
def testDecisionTree(fileName):
    from sklearn import tree
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        model = tree.DecisionTreeClassifier()
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluateClassifier(
            predictLabels, chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    return output(precisions, recalls, accuracys)


def evaluateClassifier(predictLabels, realLabels):
    tp, tn, fp, fn = evaluate.calcMixMatrix(predictLabels, realLabels)
    '''print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " +
          str(fn))'''
    precision = evaluate.calcPrecision(tp, fp)
    recall = evaluate.calcRecall(tp, fn)
    accuracy = evaluate.calcAccuracy(tp, tn, fp, fn)
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
