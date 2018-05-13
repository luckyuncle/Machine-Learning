import numpy as np
import matplotlib.pyplot as plt
import random


def createIRPlotter(dataMat, labels, features, name):
    positiveNum = 0
    negativeNum = 0
    for i in labels:
        if int(i) == 1:
            positiveNum += 1
        else:
            negativeNum += 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    feature = random.sample([i for i in range(len(features))], 2)
    ax.scatter(dataMat[:, feature[0]], dataMat[:, feature[1]], c=labels)
    plt.xlabel(features[feature[0]])
    plt.ylabel(features[feature[1]])
    plt.title(name + '(IR=' + str(int(negativeNum / positiveNum)) + ')')
    plt.show()


def createKNNPlotter(kValues, kNNEva, kNNIPEva, kNNIP2Eva, kNNIP3Eva, evaName):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kValues, kNNEva, c='r', label='kNN')
    ax.plot(kValues, kNNIPEva, c='g', label='IPC_kNN')
    ax.plot(kValues, kNNIP2Eva, c='k', label='IPS_kNN')
    ax.plot(kValues, kNNIP3Eva, c='b', label='IPCS_kNN')
    plt.xlabel('k-value')
    plt.ylabel(evaName)
    plt.legend()
    plt.show()


def createClassPlotter(precisions, recalls, f_measures):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    index = np.arange(8)
    bar_width = 0.5
    ax.bar(index, precisions, bar_width / 2, color='r', label='Precision')
    ax.bar(
        index + bar_width / 2,
        recalls,
        bar_width / 2,
        color='g',
        label='Recall')
    ax.bar(
        index + bar_width,
        f_measures,
        bar_width / 2,
        color='b',
        label='F-measure')
    plt.xlabel('Algorithm')
    plt.ylabel('Ratio')
    plt.xticks(index + bar_width / 2, ('IPC_K', 'IPS_K', 'IPCS_K', 'KNN',
                                       'SVM', 'RF', 'Bayes', 'Tree'))
    plt.legend()
    plt.show()
