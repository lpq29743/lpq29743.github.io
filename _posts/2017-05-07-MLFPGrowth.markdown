---
layout:     post
title:      "机器学习游玩记第十一站"
subtitle:   "使用FP-growth算法高效发现频繁项集"
date:       2017-05-07 21:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 机器学习
    - Python
---

> 在频繁项集的寻找路上更进一步


## 前言

上文我们了解了Apriori算法，它可以用于发现频繁项集和关联规则，但是它对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，这对于大型数据集的处理会产生较大的问题。实际上，FP-growth也是一个常用的频繁项集发现算法，它比Apriori的性能要好上两个数量级以上，但它只能用于发现频繁项集，不能发现关联规则，这篇文章就让我们一起来学习一下它。

---

## 正文

FP-growth算法只扫描数据集两次，它发现频繁项集的基本过程如下：

1. 构建FP树
2. 从FP树中挖掘频繁项集

首先我们先用代码构建一棵FP树，具体如下：

```python
from numpy import *


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),
                                                 key=lambda p: p[1],
                                                 reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
```

类treeNode定义了FP树，它包括了两个方法，其中inc方法对count变量增加给定值，disp方法用于将树以文本形式显示；createTree函数使用数据集以及最小支持度作为参数来构建FP树；updateTree函数则用于FP树的生长；updateHeader函数确保了节点链接指向树中该元素项的每一个实例；而loadSimpDat和createInitSet两个函数则用于得到一个真正的数据集。

得到一棵FP树后，我们就可以从树中挖掘频繁项集了，其基本步骤如下：

1. 从FP树中获得条件模式基
2. 利用条件模式基，构建一个条件FP树
3. 迭代重复步骤1和步骤2，直到树包含一个元素项为止

```python
from numpy import *
import fpgrowth1


def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = fpgrowth1.createTree(condPattBases, minSup)
        if myHead is not None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    simpDat = fpgrowth1.loadSimpDat()
    initSet = fpgrowth1.createInitSet(simpDat)
    myFPtree, myHeaderTab = fpgrowth1.createTree(initSet, 3)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)
```

函数findPrefixPath遍历链表直到到达结尾，每遇到一个元素项都会调用ascendTree上溯FP树，并收集所有遇到的元素项的名称；mineTree函数则用于递归查找频繁项集。

## 后记

原书有一个关于Twitter的有趣例子，感兴趣的朋友赶快去试一下吧！
