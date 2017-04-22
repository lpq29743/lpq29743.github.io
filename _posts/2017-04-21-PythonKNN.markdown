---
layout:     post
title:      "Python历险记第七站"
subtitle:   "利用Python实现KNN算法"
date:       2017-04-21 21:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 迈开机器学习的第一步


## 前言

从这篇文章开始，我将结合Python和机器学习写一个系列的文章，参考书是《机器学习实战》，话不多说，就此开始！

---

## 正文

这个系列的文章偏向实践，而非理论，如果涉及到理论的东西，还请读者自行查阅资料。另外，原书使用的是Python2，源码可以在[这里](http://www.ituring.com.cn/book/1021)下载，但这系列文章使用的是Python3。

kNN算法即k-近邻算法，它的工作原理是：存在一个样本数据集合（即训练样本集），且样本集中每个数据都存在标签。输入没有标签的新数据，将新数据的每个特征与样本集中数据对应的特征进行比较，算法将提取出样本集中特征最相似数据（最近邻）的分类标签。一般选择样本数据集中前k个最相似的数据，k一般是不大于20的整数。

基于这个原理，我们很快能参考原书写出下面的代码：

```python
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
```

这个程序中的createDataSet函数用于创建数据集和标签，group矩阵每行包含一个不同的数据，每个数据点有两个特征，向量labels包含每个数据点的标签信息，labels包含的元素个数等于group矩阵行数；classify0函数四个参数分别为：用于分类的输入向量，训练样本集，标签向量以及选择最近邻居的数目，这个函数首先计算输入向量与样本集各个数据点的距离，然后选择距离最小的k个点，最后返回发生频率最高的元素标签；这里输入的测试向量为[0, 0]，测试结果为B，读者也可以根据修改[0, 0]为其他值。

上面这个例子比较简单，但没有跟实际情况联系起来，接下来我们做几个实例：

**实例一：改进约会网站的配对效果**

这个实例来自原书，具体如下：海伦收集了1000个约会数据样本，希望通过这些数据测试一个人对她的吸引程度，根据原题和提供的文件，我们可以得到下面这个程序：

```python
from numpy import *
import operator


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m],
                                     3)
        print("the classifier came back with: {classifier}, the real answer is: {answer}".format(classifier=classifierResult, answer=datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: {errorRate}".format(errorRate=errorCount/float(numTestVecs)))
    print(errorCount)

if __name__ == '__main__':
    datingClassTest()
```

这个程序中，file2matrix函数负责将待处理数据的格式改变为分类器可以接受的格式，输入为文件名字符串，输出为训练样本矩阵和类标签向量，函数先得到文件行数，然后创建返回的NumPy矩阵，最后解析文件数据到列表；函数autoNorm用来归一化特征值，由于几个属性值的数值大致范围不一样，所以不能与上面例子一样直接用欧式距离公式求距离，这样会导致特征值权重不一，所以要将数值归一化，这里是把特征值转化为0-1这个区间的值，实际上也可以转换为-1-1这个区间；datingClassTest函数用于测试，它使用的测试数据是文本中10%（由hoRatio参数决定，可以根据需要修改）的内容，这样子可以得到错误率，来检测分类器的性能；我的最终实验的结果是错误率为6.4%，还算在可接受的范围之内。

**实例二：手写识别系统**

这同样是来自原书的例子，这个例子只针对0-9，并且用文本形式表示图像，具体实现如下：

```python
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {classifier}, the real answer is: {answer}".format(classifier=classifierResult, answer=classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: {errorCount}".format(errorCount=errorCount))
    print("\nthe total error rate is: {errorRate}".format(errorRate=errorCount/float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()
```

程序中img2vector函数负责将图像转换为向量：该函数创建1\*1024的Numpy数组，然后打开给定文件，循环读出前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组；handwritingClassTest函数进行具体的测试，获取目录内容并从文件名解析分类数字；最后得到的错误率为1.2%，相当低，效果很不错。

## 后记

k近邻算法是分类数据最简单最有效的算法，但无法给出任何数据的基础结构信息，所以下文我们将使用概率测量方法来处理分类问题，该算法可以解决这个问题。
