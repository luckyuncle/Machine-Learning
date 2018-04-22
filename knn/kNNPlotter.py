import matplotlib.pyplot as plt
from numpy import array


def createPlot(dataMat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1], dataMat[:, 2], 15.0 * array(labels),
               15.0 * array(labels))
    plt.show()
