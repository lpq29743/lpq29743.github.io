---
layout: wiki
title: Machine Learning in Action
categories: Machine Learning
description: 机器学习实战
keywords: 机器学习, 机器学习实战
---

这个 wiki 的内容为我阅读《机器学习实战》一书所做的笔记。

### 1. KNN 算法

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

k近邻算法是分类数据最简单最有效的算法，但无法给出任何数据的基础结构信息，所以我们将使用概率测量方法来处理分类问题，该算法可以解决这个问题。

### 2. 决策树

k-近邻算法最大的缺点就是无法给出数据的内在含义，决策树的主要优势就在于数据形式非常容易理解，这篇文章，就让我们一起来学习一下决策树。

怎样构造决策树是一个很关键的问题，通过原书，我们可以用以下代码实现：

```python
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat,
                                                               value),
                                                  subLabels)
    return myTree


if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print(myTree)
```

在这个程序中，createDataSet负责创建数据集；calcShannonEnt则是用来计算给定数据集的香农熵，它先是计算数据集中实例的总数，然后为所有可能分类创建数据字典，最后再求出香农熵；splitDataSet函数负责按照给定特征划分数据集，其传入的三个参数分别是待划分的数据集、划分数据集的特征以及需要返回的特征的值，Python语言中，函数传递的是列表的引用，所以对它修改为造成影响，为此，我们创建一个新的列表对象，然后遍历数据集的每个元素，一旦发现符合要求的值，就将其添加到新创建的列表中；划分好数据集后，我们就可以选择最好的数据集划分方式了，负责这个任务的是chooseBestFeatureToSplit函数，试着运行这个函数，可以得到结果0，表示的是第0个特征是最好的用于划分数据集的特征；majorityCnt函数挑选出现次数最多的类别作为返回值；createTree则是利用递归来创建最终的决策树。

上面的程序返回的结果是一个字典，不易于理解，我们尝试用Matplotlib注解绘制树形图，具体代码如下：

```python
import matplotlib.pyplot as plt
import trees1

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


if __name__ == '__main__':
    myTree = retrieveTree(1)
    createPlot(myTree)
```

getNumLeafs函数和getTreeDepth函数分别用来获取叶节点的数目和树的层数，而plotNode则是用文本注解来绘制树节点；程序的主函数是createPlot函数，该函数调用了plotTree，而plotTree又调用了前面的函数，最终绘制成了树形图。

构造好决策树后，我们就可以使用它来执行分类了，具体如下：

```python
import trees1
import trees2


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == '__main__':
    myDat, labels = trees1.createDataSet()
    myTree = trees2.retrieveTree(0)
    print(classify(myTree, labels, [1, 0]))
    print(classify(myTree, labels, [1, 1]))
```

每次使用分类器时，都要构造一次决策树，而构造决策树实际上是很耗时的工作，所以我们可以考虑用pickle模块把决策树存储起来，具体如下：

```python
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()
    
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
```

实际上，这也是决策树的优点之一，像k-近邻算法就无法持久化分类器。另外，在原书中，有一个使用决策树预测隐形眼镜类型的示例，感兴趣的朋友可以去尝试一下，本文由于篇幅原因并不展开。

关于决策树的构造算法有很多，最流行的是C4.5和CART，后面我们有机会也会提到它们。

### 3. 贝叶斯算法

之前两篇文章都是就“数据实例属于哪一类”给出一个明确的答案，但分类器有时会产生错误答案，这时可以要求分类器给出一个最优的类别猜测结果，同时给出相应的估计值。要实现这个要求，就必须用到贝叶斯分类器。

书本中给出的例子是文本分类，首先第一步就是从文本中构建词向量，具体如下：

```python
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {word} is not in my Vocabulary!".format(word=word))
    return returnVec

if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
```

loadDataSet函数创建了一些实验样本，该函数返回的第一个变量是进行词条分割后的文档集合，第二个变量是一个类别标签的集合；之后的createVocabList函数则用来创建一个包含在所有文档中出现的不重复词的列表，这里用到的是无序的set数据类型；获取到词汇表后，就可以使用setOfWords2Vec函数了，该函数的输入参数为词汇表及某个文档，输出的是文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现。

准备好数据后，我们就可以从词向量计算概率，进行朴素贝叶斯分类器的训练，具体如下：

```python
from numpy import *
import bayes1


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

if __name__ == '__main__':
    listOPosts, listClasses = bayes1.loadDataSet()
    myVocabList = bayes1.createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes1.setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)
```

trainNB0就是对应的朴素贝叶斯分类器训练函数，它的输入参数为文档矩阵trainMatrix，以及由每篇文档类别标签所构成的向量trainCategory。函数首先计算侮辱性文档（class=1）的概率，即P(1)。因为这是个二类分类问题，所以可通过P(1)得到P(0)。对于多于两类的分类问题，则需对代码稍作修改。值得一提的是，这里使用ones函数是为了避免其中一个概率值为0所带来的的不良影响。接着函数遍历训练文档集，一旦某一词语（侮辱性或正常词语）在某一文档中出现，则该词对应的个数（p1Num或p0Num）就加1，且在所有文档中，该文档总词数也加1。最后，对每个元素除以该类别中的总词数并取对数（取对数是为了解决下溢出的问题，尽管取对数导致取值不同，但由于以及在同区域增减，故不影响结果）。函数总共返回两个向量和一个概率，概率pAb即侮辱性文档的比例，向量p0V和p1V的第i个元素则分别表示词汇表中第i个元素在类别0和类别1中出现的概率。

最后便可以对算法进行简单的测试了，具体如下：

```python
from numpy import *
import bayes1
import bayes2


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = bayes1.loadDataSet()
    myVocabList = bayes1.createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes1.setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = bayes2.trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bayes1.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bayes1.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    testingNB()
```

classifyNB函数有四个输入，分别是要分类的向量以及使用函数trainNB0计算得到的三个概率，函数使用NumPy的数组来计算两个向量相乘的结果，然后将词汇表中所有词的对应值相加，然后将该值加到类别的对数概率上，最后比较类别的概率返回大概率对应的类别标签。

实际上，在原书还有从词集模型到词袋模型的扩展以及过滤垃圾邮件和从个人广告中获取区域倾向两个例子，但为了避免一次性学太多东西导致学习效果不佳，故等日后再做扩展。

### 4. 逻辑回归

最优化问题在生活中十分常见，同样在分类中也经常使用到，这一篇文章我们将利用Logistic回归来进行分类。

所谓Logistic回归，就是利用现有数据对分类边界线建立回归公式，从而实现分类的目的，初步实现的代码如下：

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

与Logistic回归类似的有一种分类方法，被认为目前最好的现成算法之一，它就是支持向量机，下一篇文章我们将一起学习它。

### 5. 支持向量机

SVM就是支持向量机的意思，它被大多数人认为是最好的现成的分类器，它有很多种实现方式，而本文使用的是序列最小优化算法，即SMO算法。

支持向量就是离分隔超平面最近的点，本文的目的就是试图最大化支持向量到分割面的距离，寻找此问题的优化求解方法，而SMO算法便是一个训练SVM的强大算法，它的目标是求出一系列的alpha和b，一旦求出了alpha，就很容易计算出权重向量w并得到分隔超平面。SMO算法的实现需要大量的代码，为了更好地了解算法的基本工作思路，我们先给出简化的版本，具体如下：

```python
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T
                - dataMatrix[i, :]*dataMatrix[i, :].T
                - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T
                - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T
                - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: {iter} i:{i}, pairs changed {alphaPairsChanged}".format(iter=iter, i=i, alphaPairsChanged=alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: {iter}".format(iter=iter))
    return b, alphas

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
```

与之前的程序一样，loadDataSet函数用于处理文本文件的数据；而selectJrand函数和clipAlpha函数都是简化版SMO算法的辅助函数，前者用于在某个区间范围内随机选择一个整数，而后者用于在数值过大时对其进行调整；smoSimple函数则是用来实现简化版的SMO算法，它的五个输入参数分别是数据集、类别标签、常数C、容错率和退出前的最大循环次数；这个程序实际上运行起来要十几秒，对于这么小的数据，已经算是过慢了，所以接下来我们看看完整的SMO算法：

```python
from numpy import *
import svm1


def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = svm1.selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = svm1.clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i]
        - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j]
        - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: {iter} i:{i}, pairs changed {alphaPairsChanged}"
                      .format(iter=iter, i=i, alphaPairsChanged=alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: {iter} i:{i}, pairs changed {alphaPairsChanged}"
                      .format(iter=iter, i=i, alphaPairsChanged=alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: {iter}".format(iter=iter))
    return oS.b, oS.alphas

if __name__ == '__main__':
    dataArr, labelArr = svm1.loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
```

SMO完整的算法主要是在alpha的选择方式上做了改进，optStruct主要是为了建立一个数据结构保存所有重要的值；辅助函数calcEk、selectJ和updateEk都是用来优化alpha的选择；而innerL和smoP则是实现了完整的SMO算法；另外，这里出现的kernelTrans函数则是用于核转换，它能把数据从某个很难处理的形式转换为一个较容易处理的形式。

还记得第一篇文章中提到的手写识别问题吗，实际上它也可以用SVM解决，具体的解决方法可以查看原书或搜索相关资料。

### 6. AdaBoost 算法

当我们把不同的分类器组合起来，这种组合结果就被称为集成方法或元算法，而元算法常见有两类，bagging和boosting，前者是用S个分类器，选择分类器投票结果中最多的类别作为最后的分类结果，而后者则是每个新分类器都根据已训练出的分类器的性能来进行训练。在boosting方法中，最流行的版本无疑是AdaBoost。

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

关于分类问题我们就暂时写到这里了，下一部分的内容是利用回归预测数值型数据。

### 7. 线性回归

回归这个概念我们在高中就接触过了，那时候计算回归方法还是用一支笔和一张纸，但随着数据量的增大，这变得越来越不现实，我们要开始尝试在计算机下求解回归方程。

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

本文相比原书缩减了很多内容，比如岭回归和逐步线性回归两个知识点，也包括预测鲍鱼年龄和玩具售价两个例子。之所以这么做，也是为了减轻一次学习的负担，让学习效果更好，后面如果学习到的话，也会对本文进行补充。

### 8. 树回归

上篇文章主要是围绕着线性回归展开，但当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型的想法就显得太难了。实际上，现实生活中很多问题都是非线性的，不可能使用全局线性模型来拟合任何数据。面对这个问题有一种可行的方法，就是将数据集切分成很多份易建模的数据，然后再利用线性回归技术建模。

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

这篇文章与上篇文章一样，省略了一些原书的内容，包括树剪枝以及Tkinter库的使用，这些内容也会随着我的深入学习而逐渐补充。

### 9. Apriori 算法

尿片啤酒典故在销售界流传已久，它属于机器学习里面很经典的关联分析问题，今天我们就一起用Apriori原理来解决一下这个问题。

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

机器学习是不是变得越来越有趣了呢？原书中还有国会投票和毒蘑菇两个示例，赶快去试一试吧！

### 10. KMeans 算法

在这篇文章之前，我们所接触的机器学习算法都是监督学习，从这篇文章开始，我们将开始接触无监督学习，而迎面而来的第一个算法，就是K-均值聚类算法。

之前我们学过分类，而这篇文章的中心则是聚类，两者的主要区别是分类的目标事先已知，而聚类则未知。K-均值是发现给定数据集的k个簇的算法，k由用户给出，每个簇都由其质心，即簇中所有点的中心来描述。K-均值算法的流程如下：

1. 随机确定k个初始点作为质心
2. 将数据集中的每个点分配到一个簇中

接下来我们用Python来实现一下K-均值算法，具体如下：

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


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 4)
```

loadDataSet函数负责加载数据；distEclud函数则是计算两个向量的欧氏距离，这是比较常见的距离函数之一；randCent函数负责构建一个包含k个随机质心的集合；而kMeans函数则实现具体的K-均值聚类算法。

对于K-均值聚类算法，k的选择起到了关键性的作用，在实际使用中，我们也会发现它的聚类效果不佳，主要是因为K-均值算法收敛到了局部最小值，而非全局最小值。为了解决这个问题，我们将其改进为二分K-均值算法。该算法首先将所有点作为一个簇，然后将该簇一分为二，之后选择其中一个簇继续进行划分，选择哪一个簇划分取决于对其划分是否可以最大程度降低误差平方和，即SSE的值，不断重复这个操作，直到得到指定簇数目为止。另一种做法是可以选择SSE最大的簇进行划分，直到簇数目达到要求为止，它的具体实现代码如下：

```python
from numpy import *
import kmeans1


def biKmeans(dataSet, k, distMeas=kmeans1.distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kmeans1.kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

if __name__ == '__main__':
    datMat = mat(kmeans1.loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat, 3)
```

有没有想过在出去旅游的时候也把你想去的地点来一个聚类呢？实际上，在书中就有这个例子，感兴趣的话就去实现一下吧！

### 11. FP-growth 算法

上文我们了解了Apriori算法，它可以用于发现频繁项集和关联规则，但是它对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，这对于大型数据集的处理会产生较大的问题。实际上，FP-growth也是一个常用的频繁项集发现算法，它比Apriori的性能要好上两个数量级以上，但它只能用于发现频繁项集，不能发现关联规则，这篇文章就让我们一起来学习一下它。

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

原书有一个关于Twitter的有趣例子，感兴趣的朋友赶快去试一下吧！

### 12. 机器学习工具

上文我们了解了Apriori算法，它可以用于发现频繁项集和关联规则，但是它对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，这对于大型数据集的处理会产生较大的问题。实际上，FP-growth也是一个常用的频繁项集发现算法，它比Apriori的性能要好上两个数量级以上，但它只能用于发现频繁项集，不能发现关联规则，这篇文章就让我们一起来学习一下它。

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

原书有一个关于Twitter的有趣例子，感兴趣的朋友赶快去试一下吧！