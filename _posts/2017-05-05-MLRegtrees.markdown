---
layout:     post
title:      "机器学习游玩记第八站"
subtitle:   "从全局到局部，从线性回归到树回归"
date:       2017-05-05 21:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 回归变得越来越有趣了！


## 前言

上篇文章主要是围绕着线性回归展开，但当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型的想法就显得太难了。实际上，现实生活中很多问题都是非线性的，不可能使用全局线性模型来拟合任何数据。面对这个问题有一种可行的方法，就是将数据集切分成很多份易建模的数据，然后再利用线性回归技术建模。

---

## 正文

前面曾提过决策树，而该文使用的树构建算法是ID3。ID3的做法是每次选取当前最佳的特征来分割数据，并按照该特征的所有可能取值来切分，但ID3存在切分迅速、不能直接处理连续性特征等缺点，所以本文我们提到另一种树构建算法，就是分类回归树，即CART。CART使用二分切分来处理连续型变量，对其稍作修改就可以解决回归问题。要使用CART算法，第一个步骤便是构建树，具体如下：

```python
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

if __name__ == '__main__':
    myDat = loadDataSet('ex00.txt')
    myMat = mat(myDat)
    retTree = createTree(myMat)
    print(retTree)
```

binSplitDataSet函数传入的3个参数分别是数据集合、待切分的特征和该特征的某个值，在给定特征和特征值的情况下，该函数通过数组过滤方式将数据集切分得到两个子集并返回；函数createTree传入数据集和3个可选参数，这些可选参数分别是给出建立叶节点函数的leafType、代表误差计算函数的errType以及包含树构建参数的元组ops。

## 后记

这篇文章与上篇文章一样，省略了一些原书的内容，包括树剪枝以及Tkinter库的使用，这些内容也会随着我的深入学习而逐渐补充。
