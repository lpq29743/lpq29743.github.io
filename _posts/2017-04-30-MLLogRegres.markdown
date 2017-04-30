---
layout:     post
title:      "机器学习游玩记第四站"
subtitle:   "Logistic回归分类器"
date:       2017-04-30 08:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 期待已久的最优化算法！


## 前言

最优化问题在生活中十分常见，同样在分类中也经常使用到，这一篇文章我们将利用Logistic回归来进行分类。

---

## 正文

正如这个系列的第一篇文章所讲，文章将不注重理论，更偏向于代码的实现，如果想了解Logistic回归，可以查看原书或者查阅相关资料。所谓Logistic回归，就是利用现有数据对分类边界线建立回归公式，从而实现分类的目的，初步实现的代码如下：

```python
from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights.getA())
```

在这个程序中，loadDataSet函数负责打开testSet.txt文件，并将其转换为相应的数据集和标签集；gradAscent函数则是完成梯度上升算法，函数传入两个参数，分别是数据集和标签集，然后根据Logistic回归梯度上升优化算法进行计算，最终返回一组回归系数；得到回归系数后，plotBestFit函数就根据这些回归系数画出不同类别数据之间的分割线。

梯度上升算法在每次更新回归系数时都需要遍历整个数据集，这种方法在处理小型数据集尚可，但处理大型数据集的话，计算复杂度就太高了。一种改进方法是一次仅用一个样本点来更新回归系数，即随机梯度上升算法，具体实现如下：

```python
from numpy import *
import logRegres1


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = logRegres1.sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

if __name__ == '__main__':
    dataArr, labelMat = logRegres1.loadDataSet()
    weights = stocGradAscent1(array(dataArr), labelMat)
    logRegres1.plotBestFit(weights)
```

## 后记

与Logistic回归类似的有一种分类方法，被认为目前最好的现成算法之一，它就是支持向量机，下一篇文章我们将一起学习它。
