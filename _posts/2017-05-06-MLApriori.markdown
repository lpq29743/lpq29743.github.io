---
layout:     post
title:      "机器学习游玩记第十站"
subtitle:   "使用Apriori算法进行关联分析"
date:       2017-05-06 22:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 当机器学习遇到尿片啤酒问题


## 前言

尿片啤酒典故在销售界流传已久，它属于机器学习里面很经典的关联分析问题，今天我们就一起用Apriori原理来解决一下这个问题。

---

## 正文

对于一个数据集，我们可以求出某一项集的支持度和可信度，这两者是用来量化关联分析是否成功的方法。然而，对于一个数据集，项集的数目呈指数增长。为了解决这个问题，我们引入了Apriori原理，这个原理是说如果某个项集是频繁的，那么它的所有子集也是频繁的。利用这一点，我们只要知道某一个项集是非频繁的，就可以不去计算包含它的项集，也就可以避免项集数目的指数增长，从而在合理时间内计算出频繁项集。接下来我们看一下具体的代码：

```python
from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

if __name__ == '__main__':
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    print(L)
```

loadDataSet函数负责加载数据；而createC1函数和scanD函数则是Apriori算法的辅助函数，前者负责构建集合C1，C1是大小为1的所有候选项集的集合，后者传入三个参数，分别是数据集、候选项集列表以及最小支持度；aprioriGen函数和apriori函数则用来构建完整的Apriori算法，默认的最小支持度为0.5。

在上面的这个程序中，我们已经成功地找出了频繁项集，接下来的问题是如何找出关联规则。解决这个问题，我们就需要用到开头提到的可信度，具体实现如下：

```python
from numpy import *
import apriori1


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = apriori1.aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == '__main__':
    dataSet = apriori1.loadDataSet()
    L, suppData = apriori1.apriori(dataSet)
    rules = generateRules(L, suppData)
    print(rules)
```

generateRules函数传入3个参数，分别是频繁项集列表、包含频繁项集支持数据的字典以及最小可信度阈值，最后返回一个包含可信度的规则列表；calcConf函数用于计算可信度值；rulesFromConseq函数则用于合并频繁项集。

## 后记

机器学习是不是变得越来越有趣了呢？原书中还有国会投票和毒蘑菇两个示例，赶快去试一试吧！
