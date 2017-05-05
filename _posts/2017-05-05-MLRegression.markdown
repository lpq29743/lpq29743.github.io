---
layout:     post
title:      "机器学习游玩记第七站"
subtitle:   "迈开回归第一步"
date:       2017-05-05 16:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 预测数值型数据：回归


## 前言

回归这个概念我们在高中就接触过了，那时候计算回归方法还是用一支笔和一张纸，但随着数据量的增大，这变得越来越不现实，我们要开始尝试在计算机下求解回归方程。

---

## 正文

利用普通最小二乘法，即OLS就可以求出相应的线性回归方程，代码相对简单，具体如下：

```python
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
```

在这个程序中，loadDataSet函数用于加载数据，而standRegres函数则用来计算最佳拟合直线，最后再通过调用以上函数画出相应的拟合直线。

线性回归的一个问题是有可能出现欠拟合现象，如果出现这种现象，就不能取得最好的预测解决，所以我们可以用局部加权线性回归，即LWLR来解决这个问题。在此算法中，我们给待预测点附近的每个点赋予一定的权重，在进行普通回归，最终得到的代码如下：

```python
from numpy import *
import matplotlib.pyplot as plt
import regression1


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


if __name__ == '__main__':
    xArr, yArr = regression1.loadDataSet('ex0.txt')
    lwlr(xArr[0], xArr, yArr, 1.0)
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0],
               mat(yArr).T.flatten().A[0],
               s=2,
               c='red')
    xCopy = xMat.copy()
    xCopy.sort(0)
    plt.show()
```

运行上面这个程序，我们可以看到不错的拟合结果，这对我们的预测带来了很大的帮助。

## 后记

本文相比原书缩减了很多内容，比如岭回归和逐步线性回归两个知识点，也包括预测鲍鱼年龄和玩具售价两个例子。之所以这么做，也是为了减轻一次学习的负担，让学习效果更好，后面如果学习到的话，也会对本文进行补充。
