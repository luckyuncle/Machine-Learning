from numpy import tile, zeros, argsort
import evaluate
import kNN1

# import operator


# 构造分类器
def classify0(inX, dataSet, labels, k):
    # 输入向量inX, 输入训练样本集dataSet, 标签向量labels, 选择最近邻居的数目k
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # print(1 / (distances + 0.5))
    sims = 1 / (1 + distances)
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
    '''
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    print(nSumWeights)
    print(pSumWeights)
    '''
    if nSumWeights > pSumWeights:
        return -1
    else:
        return 1


def loadDataSet(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    featureNum = len(arrayOLines[0].strip().split('\t')) - 1
    returnMat = zeros((numberOfLines, featureNum))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0:featureNum]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


# 读取文件转化为NumPy
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    featureNum = len(arrayOLines[0].split(',')) - 1
    returnMat = zeros((numberOfLines, featureNum))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = ''.join(line.split())
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[0:featureNum]
        if (listFromLine[-1] == "negative"):
            classLabelVector.append(-1)
        else:
            classLabelVector.append(1)
        index += 1
    return returnMat, classLabelVector


# 归一化数值
def autoNorm(minVals, maxVals, dataSet):
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet


# 交叉验证方法
def crossAuth(dataMat, labels, i):
    featureNum = dataMat.shape[1]  # 行数
    length = int(len(labels) / 10)
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


# 测试数据
def datingClassTest():
    # trainingMat, hmLabels = file2matrix('haberman/haberman1tra.txt')
    # testMat, chmLabels = file2matrix('haberman/haberman1tst.txt')
    # normTrainingMat, hmLabels = file2matrix('ecoli/ecoli5tra.txt')
    # normTestMat, chmLabels = file2matrix('ecoli/ecoli5tst.txt')
    dataMat, labels = file2matrix('ecoli1.dat')
    normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
        dataMat, labels, 3)
    # 根据情况决定是否归一化
    # minVals = trainingMat.min(0)
    # maxVals = trainingMat.max(0)
    # normTrainingMat = autoNorm(minVals, maxVals, trainingMat)
    # normTestMat = autoNorm(minVals, maxVals, testMat)
    errorCount = 0.0
    m = len(normTestMat)
    predictLabels = []
    for i in range(m):
        classifierResult = classify0(normTestMat[i, :], normTrainingMat,
                                     hmLabels, 5)
        '''
        print("the classifier came back with: %s, the real answer is: %s" %
              ("negative" if classifierResult == -1 else "positive", "negative"
               if chmLabels[i] == -1 else "positive"))
        '''
        if (classifierResult != chmLabels[i]):
            errorCount += 1.0
        predictLabels.append(classifierResult)
    tp, tn, fp, fn = evaluate.calcMixMatrix(predictLabels, chmLabels)
    print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " +
          str(fn))
    print("正确率：" + str(evaluate.calcPrecision(tp, fp)))
    print("召回率：" + str(evaluate.calcRecall(tp, fn)))
    print("准确率：" + str(evaluate.calcAccuracy(tp, tn, fp, fn)))
    print("\nthe total numbers of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m)))


def datingClassTest1():
    dataMat, labels = file2matrix('ecoli1.dat')
    normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
        dataMat, labels, 3)
    errorCount = 0.0
    m = len(normTestMat)
    predictLabels = []
    for i in range(m):
        classifierResult = kNN1.classify0(normTestMat[i, :], normTrainingMat,
                                          hmLabels, 5)
        if (classifierResult != chmLabels[i]):
            errorCount += 1.0
        predictLabels.append(classifierResult)
    tp, tn, fp, fn = evaluate.calcMixMatrix(predictLabels, chmLabels)
    print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " +
          str(fn))
    print("正确率：" + str(evaluate.calcPrecision(tp, fp)))
    print("召回率：" + str(evaluate.calcRecall(tp, fn)))
    print("准确率：" + str(evaluate.calcAccuracy(tp, tn, fp, fn)))
    print("\nthe total numbers of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m)))
