from numpy import tile, zeros
import operator


# 构造分类器
def classify0(inX, dataSet, labels, k):
    # 输入向量inX, 输入训练样本集dataSet, 标签向量labels, 选择最近邻居的数目k
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


# 测试数据
def datingClassTest():
    # trainingMat, hmLabels = file2matrix('haberman/haberman1tra.txt')
    # testMat, chmLabels = file2matrix('haberman/haberman1tst.txt')
    normTrainingMat, hmLabels = file2matrix('ecoli/ecoli5tra.txt')
    normTestMat, chmLabels = file2matrix('ecoli/ecoli5tst.txt')
    # 根据情况决定是否归一化
    # minVals = trainingMat.min(0)
    # maxVals = trainingMat.max(0)
    # print(minVals, maxVals)
    # normTrainingMat = autoNorm(minVals, maxVals, trainingMat)
    # normTestMat = autoNorm(minVals, maxVals, testMat)
    errorCount = 0.0
    m = len(normTestMat)
    for i in range(m):
        classifierResult = classify0(normTestMat[i, :], normTrainingMat,
                                     hmLabels, 3)
        print("the classifier came back with: %s, the real answer is: %s" %
              ("negative" if classifierResult == -1 else "positive", "negative"
               if chmLabels[i] == -1 else "positive"))
        if (classifierResult != chmLabels[i]):
            errorCount += 1.0
    print("\nthe total numbers of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m)))
