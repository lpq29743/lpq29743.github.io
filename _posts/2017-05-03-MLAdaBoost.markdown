---
layout:     post
title:      "机器学习游玩记第六站"
subtitle:   "利用AdaBoost元算法提高分类性能"
date:       2017-05-03 18:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 最流行的元算法——AdaBoost


## 前言

当我们把不同的分类器组合起来，这种组合结果就被称为集成方法或元算法，而元算法常见有两类，bagging和boosting，前者是用S个分类器，选择分类器投票结果中最多的类别作为最后的分类结果，而后者则是每个新分类器都根据已训练出的分类器的性能来进行训练。在boosting方法中，最流行的版本无疑是AdaBoost。

---

## 正文

使用弱分类器和多个实例能否构建一个强分类器呢？实际上是可以的，AdaBoost算法就可以解决这个问题。首先我们可以基于单层决策树构建弱分类器，然后再实现AdaBoost算法，具体如下：

```python
from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


if __name__ == "__main__":
    datMat, classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    print(classifierArray)
```

函数loadSimpData用于加载简单的数据，而stumpClassify函数和buildStump函数则是用于单层决策树的生成，它是一个基于加权输入值进行决策的弱分类器，最后我们利用adaBoostTrainDS函数实现基于单层决策树的AdaBoost训练过程。

## 后记

关于分类问题我们就暂时写到这里了，下一部分的内容是利用回归预测数值型数据。
