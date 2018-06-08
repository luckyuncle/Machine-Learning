from numpy import argsort, tile
import operator


# 计算距离
def calcDistance(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    return distances


# 构造kNN基于距离的类别加权分类器
def IPDCKNNClassify(inX, dataSet, labels, k):
    distances = calcDistance(inX, dataSet)
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


# 构造kNN基于数量的类别加权分类器
def IPNCKNNClassify(inX, dataSet, labels, k):
    distances = calcDistance(inX, dataSet)
    sims = 1 / (0.001 + distances)
    nNumber = 0
    pNumber = 0
    for label in labels:
        if label == 1:
            pNumber += 1
        else:
            nNumber += 1
    nWeight = pNumber / len(labels) * 6
    pWeight = nNumber / len(labels)
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


# 构造kNN基于距离的样本加权分类器
def IPDSKNNClassify(inX, dataSet, labels, k):
    distances = calcDistance(inX, dataSet)
    sims = 1 / (0.001 + distances)
    sims = sims * sims
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


# 构造kNN基于距离的类别和样本加权分类器
def IPDCSKNNClassify(inX, dataSet, labels, k):
    distances = calcDistance(inX, dataSet)
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


# kNN分类器
def kNNClassify(inX, dataSet, labels, k):
    distances = calcDistance(inX, dataSet)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 实现SVM
def testSVM(flag=True):
    from sklearn.svm import SVC
    if flag:
        return SVC(kernel='rbf')
    else:
        return SVC(kernel='linear')


# 实现随机森林
def testRandomForest(estimators=8):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=estimators)


# 实现logistics回归
def testLogistic(penaltys='l2'):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(penalty=penaltys)


# 实现朴素贝叶斯
def testBayes(flag=True):
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    if flag:
        return GaussianNB()
    else:
        return MultinomialNB(alpha=0.01)


# 实现决策树
def testDecisionTree():
    from sklearn import tree
    return tree.DecisionTreeClassifier()
