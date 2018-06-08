from dataTreat import file2matrix, crossAuth, loadFileData, autoNorm
import algorithmRealize as ar
import evaluate


# 测试其它机器学习分类算法
def testOthersByFile(fileName, algorithmName, flag=1):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    allPredictLabels = []
    allRealLabels = []
    model = None
    for i in range(10):
        normTrainingMat, normTestMat, hmLabels, chmLabels = crossAuth(
            dataMat, labels, i)
        if algorithmName == 'SVM':
            model = ar.testSVM(flag)
        elif algorithmName == 'RF':
            model = ar.testRandomForest()
        elif algorithmName == 'Bayes':
            model = ar.testBayes(flag)
        elif algorithmName == 'Tree':
            model = ar.testDecisionTree()
        else:
            pass
        if model is None:
            raise NameError('algorithm input error')
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluate.evaluateClassifier(
            predictLabels, chmLabels)
        allPredictLabels.extend(predictLabels)
        allRealLabels.extend(chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    avgPrecision, avgRecall, avgAccuracy = evaluate.output(
        precisions, recalls, accuracys)
    return \
        avgPrecision, avgRecall, avgAccuracy, allPredictLabels, allRealLabels


# 测试其它机器学习分类算法
def testOthersByDir(fileName, algorithmName, flag=1):
    precisions = []
    recalls = []
    accuracys = []
    allPredictLabels = []
    allRealLabels = []
    model = None
    for i in range(5):
        normTrainingMat, normTestMat, hmLabels, \
            chmLabels, minVals, maxVals = loadFileData(fileName, i)
        if algorithmName == 'SVM':
            model = ar.testSVM(flag)
        elif algorithmName == 'RF':
            model = ar.testRandomForest()
        elif algorithmName == 'Bayes':
            model = ar.testBayes(flag)
        elif algorithmName == 'Tree':
            model = ar.testDecisionTree()
        else:
            pass
        if model is None:
            raise NameError('algorithm input error')
        model.fit(normTrainingMat, hmLabels)
        predictLabels = model.predict(normTestMat)
        precision, recall, accuracy = evaluate.evaluateClassifier(
            predictLabels, chmLabels)
        allPredictLabels.extend(predictLabels)
        allRealLabels.extend(chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    avgPrecision, avgRecall, avgAccuracy = evaluate.output(
        precisions, recalls, accuracys)
    return \
        avgPrecision, avgRecall, avgAccuracy, allPredictLabels, allRealLabels


# 测试KNN分类算法
def testKNNByFile(fileName, algorithmName, kValue=5):
    dataMat, labels, features = file2matrix(fileName)
    precisions = []
    recalls = []
    accuracys = []
    allPredictLabels = []
    allRealLabels = []
    classifierResult = None
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
            if algorithmName == 'KNN':
                classifierResult = ar.kNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDC_KNN':
                classifierResult = ar.IPDCKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDS_KNN':
                classifierResult = ar.IPDSKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDCS_KNN':
                classifierResult = ar.IPDCSKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPNC_KNN':
                classifierResult = ar.IPNCKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            else:
                pass
            if classifierResult is None:
                raise NameError('algorithm input error')
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluate.evaluateClassifier(
            predictLabels, chmLabels)
        allPredictLabels.extend(predictLabels)
        allRealLabels.extend(chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    avgPrecision, avgRecall, avgAccuracy = evaluate.output(
        precisions, recalls, accuracys)
    return \
        avgPrecision, avgRecall, avgAccuracy, allPredictLabels, allRealLabels


# 测试KNN分类算法
def testKNNByDir(fileName, algorithmName, kValue=5):
    precisions = []
    recalls = []
    accuracys = []
    allPredictLabels = []
    allRealLabels = []
    classifierResult = None
    for j in range(5):
        normTrainingMat, normTestMat, hmLabels, \
            chmLabels, minVals, maxVals = loadFileData(fileName, j)
        errorCount = 0.0
        m = len(normTestMat)
        predictLabels = []
        for i in range(m):
            if algorithmName == 'KNN':
                classifierResult = ar.kNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDC_KNN':
                classifierResult = ar.IPDCKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDS_KNN':
                classifierResult = ar.IPDSKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPDCS_KNN':
                classifierResult = ar.IPDCSKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            elif algorithmName == 'IPNC_KNN':
                classifierResult = ar.IPNCKNNClassify(
                    normTestMat[i, :], normTrainingMat, hmLabels, kValue)
            else:
                pass
            if classifierResult is None:
                raise NameError('algorithm input error')
            if (classifierResult != chmLabels[i]):
                errorCount += 1.0
            predictLabels.append(classifierResult)
        precision, recall, accuracy = evaluate.evaluateClassifier(
            predictLabels, chmLabels)
        allPredictLabels.extend(predictLabels)
        allRealLabels.extend(chmLabels)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
    avgPrecision, avgRecall, avgAccuracy = evaluate.output(
        precisions, recalls, accuracys)
    return \
        avgPrecision, avgRecall, avgAccuracy, allPredictLabels, allRealLabels
