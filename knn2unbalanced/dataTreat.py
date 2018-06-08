from numpy import tile, zeros, array
from os import listdir


# 处理文件中的数据
def loadFileData(file, i):
    fileList = listdir(file)
    fileName = fileList[i]
    minVals, maxVals = getMaxAndMin(file + '/' + fileName)
    trainingMat, hmLabels, features = file2matrix(file + '/' + fileName)
    fileName = fileList[i + 1]
    testMat, chmLabels, features = file2matrix(file + '/' + fileName)
    return trainingMat, testMat, hmLabels, chmLabels, minVals, maxVals


# 处理文件数据，读取文件转化为NumPy
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


def createIR(file):
    fileList = listdir(file)
    labels = []
    for i in range(len(fileList)):
        dataMat, label, features = file2matrix(file + '/' + fileList[i])
        labels.extend(label)
    return dataMat, label, features, len(labels)


# 根据文件得到最大最小值
def getMaxAndMin(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    minVals = []
    maxVals = []
    for line in arrayOLines:
        if line[0] == '@':
            lineList = line.strip('\n').replace('[', '').replace(']',
                                                                 '').split(' ')
            if lineList[0] == '@attribute' and lineList[1].lower() != 'class':
                minVals.append(float(lineList[3].replace(',', '')))
                maxVals.append(float(lineList[4]))
        else:
            break
    return array(minVals), array(maxVals)


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
