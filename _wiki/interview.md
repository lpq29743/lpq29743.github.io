---
layout: wiki
title: Interview
categories: Algorithm
description: 面试题
keywords: 面试题
---

### Question

#### Math

##### Fun Math

Q1.有 1000 个一模一样的瓶子，其中有 999 瓶是普通的水，有一瓶是毒药。任何喝下毒药的生物都会在一星期之后死亡。现在，你只有 10 只小白鼠和一星期的时间，如何检验出哪个瓶子里有毒药？

A1.可以用二进制编码的思维来解决。先对1000个瓶子以二进制数的形式进行编号，则至少需要十位的二进制数进行表示。再用这10只小白鼠分别对应二进制数的10个数位，让每只小白鼠喝下编号对应数位值为1的瓶子。最后根据小白鼠的死亡情况得到一个十位二进制数，编号与其相等的瓶子里面有毒药。

Q2.1000桶水，其中一桶有毒，猪喝毒水后会在15分钟内死去，想用一个小时找到这桶毒水，至少需要几头猪？

A2.由于一只猪总共有五种状态，所以可以用五进制的方法解决，将1000桶水表示成五进制至少需要5位，所以至少需要五头猪。

Q3.1000桶水中两桶有毒，猪喝毒水后会在15分钟内死去，想用一个小时找到毒水，至少需要几只猪？如何实现？

A3.从理论上来讲，1000桶水两桶有毒，有499500种状态，而一头猪有5种状态，求解 5^l > 499500，求得 l > 8.15，所以9只猪就可以找到毒水：

1. 我们先把1000桶水分为10组，每只猪试验一组，一组空白
2. 如果只有一只猪死，那么说明两桶毒水都在那100桶水中，再把这100桶水分为10组，剩下的9只猪试验9组，另外10桶不实验，这样问题可一步一步缩小
3. 如果有两只猪死，则说明有两组100桶水各有一桶毒水，每组用3头猪即可实验

Q4.一个村子里有很多户人家，每家都养了一条狗。现在发现村子里面出现了n只疯狗，村里规定，谁要是发现了自己的狗是疯狗，就要将自己的狗枪毙。但问题是，村子里面的人只能看出别人家的狗是不是疯狗，而不能看出自己的狗是不是疯的，如果看出别人家的狗是疯狗，也不能告诉别人。于是大家开始观察，第一天晚上，没有枪声，第二天晚上，没有枪声，直到第n天晚上，这n只狗才被同时枪毙，请问这是为什么？

A4.具体分析流程如下：

1. 首先从只有一条疯狗分析起，如果只有一条疯狗，那么疯狗的主人在第1天就会发现其他家庭一只疯狗都没有，从而枪毙自己家的狗
2. 如果有两条疯狗，那么拥有疯狗的家庭在第一天由于看到别人家有疯狗，就不会枪毙自己家的狗，但发现第一晚没人枪毙自己的狗后，他会知道村子里有两条疯狗，其中一条就是自己的。实际上，整个村子的人都会在第一晚没人枪毙自己的狗后，知道整个村子至少有两条疯狗
3. 继续分析，如果第二晚还没有枪声，那说明拥有疯狗的人都看到了至少两条疯狗，所以他不会认为自己拥有疯狗，但经过没有枪声的第二晚后，全村人便达成了村子至少有三条疯狗的事实
4. 同理可得，拥有疯狗的家庭由于能看到n-1条狗，他需要n天才能判断村子里至少有n条疯狗，其中一条就是自己家的狗，从而在第n天晚上枪毙自己家的狗

Q5.五个同事决定计算他们的平均工资，在大家互相不告诉薪水的情况下，如何才能做到这一点？

A5.这道题的方法有很多，比如有：

1. 每个人把自己的工资随意拆成四个数的和，分别把四个数字告诉自己以外的四个人；每个人手里收到四个数字，加起来，报出；五个人的数字相加，即可得到五个人的总收入，除以5得到平均工资
2. 找个计算器，叫第一个人输入一个巨大的不规则的数，然后把自己的收入加上去，之后依次传给其他人，每人都把自己的收入加上之前的数。最后传回第一个人。第一个人再把最后的总和减去Ta选中的那个不规则数，然后除以人数，即可得到大家的平均。

Q6.圆桌上有1到1000号，1号右手边是2号，左手边是1000号。1号开枪打死2号，把枪交给3号，3号打死4号交给5号。。999号打死1000号后把枪交给1号，继续循环。最后留下来的是几号？

A6.约瑟夫环问题，套公式 f(n) = 2(n - 2^log2(n)) + 1 直接得到结果为977。

Q7.一幢 200 层的大楼，给你两个鸡蛋。如果在第 n 层扔下鸡蛋，鸡蛋不碎，那么从前 n-1 层扔鸡蛋都不碎。这两只鸡蛋一模一样，不碎的话可以扔无数次。已知鸡蛋在0层扔不会碎。提出一个策略，要保证能测出鸡蛋恰好会碎的楼层，并使此策略在最坏情况下所扔次数最少

A7.这是一道非常经典的面试题，我们用两种方法来解决：

1. 分析法：对于每一次扔鸡蛋，都可以看作是一次决策，所以最终扔的方案应该是构成一棵决策树，问题就可以转换成求最矮决策树的高度。假设第一次扔的楼层是第k层楼，则碎子树的高度为k-1，如果第一次扔鸡蛋没碎，则设第二次扔的高度为m，则对于m来讲，其碎子树高度为m-k-1，相对根节点高度则为m-k。由于要尽可能保证子树的高度一致，所以得m-k=k-1，故可得第二次扔的高度要比前一次高k-1层。从而得到方程k(k+1)/2 = 200，从而解得高度为14
2. 动态规划法：这道题是很经典的动态规划问题，设楼层次数为n，我们可以得到状态转移方程`f(n) = min(max(k-1, f(n - k))) + 1 (0 < k <= n)`，如果我们再加入鸡蛋数目变量m，则状态转移方程为`f(n, m) = min(max(f(k - 1, m - 1), f(n - k, m))) + 1 (0 < k <= n)`

Q8.海盗博弈

A8.[链接1](https://www.zhihu.com/question/20014343)、[链接2](https://zhuanlan.zhihu.com/p/27388049)、[链接3](https://www.zhihu.com/question/47973941)

Q9.五个囚犯先后从100颗绿豆中抓绿豆。抓得最多和最少的人将被处死，不能交流，可以摸出剩下绿豆的数量，谁的存活几率最大？

A9.[链接](https://www.zhihu.com/question/19912025)

Q10.三个极度嫉妒的人分一个蛋糕，采用什么策略，能让三人都觉得公平？

A10.[链接](https://www.zhihu.com/question/20615717)

Q11.有 8 个台球，其中一个比其他的 7 个都要重一些。如果仅仅是使用天平而不称出具体的重量，请问最少几次能找出那个最重的台球？

A11.2 次。把所有的球分成 3 组，其中 2 组是 3 个球，最后一组是两个球。首先，把 3 个球的两组放在天平上。如果其中一方比较重，把偏重的那一组球任意拿出来两个放在天平上。如果两组球一样重，那就把剩下的 2 个球放在天平上称重。

##### Probability Theory and Mathematical Statistics

Q1.为什么样本方差（sample variance）的分母是 n-1？

A1.分母是n-1是为了保证方差的估计是无偏的。如果直接使用n为分母作为估计，那么会倾向于低估方差（可用数学方法证明），所以为了正确的估计方差，所以可以把原先的估计值稍微放大一点，即把分母n改为n-1。

这里也可以用自由度的角度进行分析。对于n个样本，样本均值是先定的，因此只剩下n-1个样本的值是可以变化的。换句话说，样本中原有的n个自由度，有一个被分配给计算样本均值，剩下自由度即为n-1，所以用n-1作为分母来计算样本方差。

Q2.给一枚硬币，但扔出正反的概率未知，如何得到等概率的二元随机数

A2.扔两次，00、11时无输出重扔，01输出0，10输出1。

Q3.如何用一个骰子等概率地生成1到7的随机数

A3.将一个筛子扔两次可以得到36种组合，每五种组合代表一个数字，剩下的一种表示重扔。

Q4.抛的硬币直到连续出现两次正面为止，平均要扔多少次

A4.用马尔可夫链，可做图求解递归方程。

假定扔出正面(H)的概率为 p，扔出反面(T)的概率为 1 - p。我们需要扔出连续 2 个 H。在整个过程有这么几种状态：

1. 当前连续 0 个正面（0H）
2. 当前连续 1 个正面（1H）
3. 当前连续 2 个正面（2H）

如果当前是 0H，那么 p 的概率，下一个状态是 1H；1 - p 的概率维持在 0H。

如果当前是 1H，那么 p 的概率，下一个状态为 2H（达到条件，任务完成）；1 - p 的概率回到 0H。

假设期望 x 次后，得到 2H，则有 $$x = (1 − p)(1 + x) + p^2 × 2 + p(1 − p)(2 + x)$$，可解得 x = 6。

Q5.一米长的绳子，随机剪两刀，最长的一段有多长

A5.假设三段的长度从小到大依次为 a，a + b，a + b + c，并且满足 a + a + b + a + b + c = 3a + 2b + c = 1 以及 a > 0，b ≥ 0，c ≥ 0

则可以得到 a ≤ 1/3，b ≤ 1/2，c ≤ 1，不妨可以认为 a ∼ U(0, 2k)，b ∼ U(0, 3k)，c∼U(0, 6k)。

绳子最长的一段的期望为 k + 1.5k + 3k = 5.5k，绳子长度的期望为 3k + 3k + 3k = 9k。因为 9k = 1，所以 5.5k = 11/18 = 0.61111

Q6.给定一个 0 到 1 的均匀分布，如何近似地生成一个标准正态分布。即用 numpy.random.uniform() 这个函数， 得到 numpy.random.normal()

A6.本题考点为中心极限定理和均匀分布。中心极限定理即一组相互独立的随机变量的均值符合正态分布。

`np.random.uniform()`生成的是 (0, 1) 之间均匀分布的随机数，则`2 * np.random.uniform() - 1`生成的是 (-1, 1) 之间均匀分布的随机数。

已知 U(a, b) 方差是 (a - b)^2 / 12，则含有 n 个样本的样本均值的方差是 (a - b)^2 / 12 / n。代码如下：

```
import numpy as np
normal_rv = 30 * np.mean(2 * np.random.uniform(size=300) - 1)
```

具体步骤是先产生 300 个 (-1, 1) 随机变量，它们的均值的标准差是 1 / 30，要得到标准正态分布，所以要乘以 30。

Q7.假设一段公路上，1 小时内有汽车经过的概率为96%，那么，30分钟内有汽车经过的概率为

A7.一小时有车的概率 = 1 - 一小时没车的概率 = 1 - 两个半小时都没车的概率 = 1 - (1 - 半小时有车的概率)^2

Q8.一枚不均匀硬币，抛了 100 次，有 70 次朝上，第 101 次朝上的概率是多少，公式是如何推导

A8.7/10。二项分布的极大似然估计，可参考[此链接](https://www.zhihu.com/question/24124998)。

Q9.4个人，52张扑克牌，红桃 A 和黑桃 A 同时被一个人拿到的概率

A9.解法一：C(1,4) * C(11,50) / C(13,52)，C(1,4) = 从四个人中任选 1 人为红桃 A + 黑桃 A，C(11,50) = 从剩余 50 张牌中抽取 11 张给指定人，C(13,52) = 从 52 张牌中随机抽取 13 张

解法二：对于抓到红桃 A 的人，再抓黑桃 A 的概率就是 12/51 = 4/17

#### Algorithm

Q1.有10个排好序的数据库，那么我要找整个的中位数，怎么找

A1.最简单的思路是合并数据库，然后再定位长度，时间复杂度为 O(n)，空间复杂度是 O(n)；但实际上只需要借鉴这个合并的过程，当合并到中位数的时候输出中位数即可，时间复杂度为 O(n)，空间复杂度是 O(1)。这思路十分简单，但并不是最佳算法，有序数组让我们想到的会是二分查找，因此我们可以利用二分查找来使复杂度降至 O(logn)，具体可参考：

1. https://www.douban.com/note/177571938/
2. https://stackoverflow.com/questions/6182488/median-of-5-sorted-arrays

Q2.无序整数数组中找第 k 大的数

A2.[链接](https://blog.csdn.net/wangbaochu/article/details/52949443)

Q3.不用库函数求一个数的立方根

A3.[链接](https://blog.csdn.net/sjpz0124/article/details/47726275)

#### Operating Systems

Q1.为什么要用时间复杂度来描述算法，而不是运行时间

A1.操作系统调度，所以运行时间不一定相同

#### Database Systems

Q1.count(1)、count(*) 和 count(列名)的区别

A1.[链接](https://blog.csdn.net/qq_15037231/article/details/80495882)

Q2.数据库三范式

A2.[链接](https://www.zhihu.com/question/24696366)

#### Machine Learning

Q1.You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)

A1.Processing a high dimensional data on a limited memory machine is a strenuous task, your interviewer would be fully aware of that. Following are the methods you can use to tackle such situation:

1. Since we have lower RAM, we should close all other applications in our machine, including the web browser, so that most of the memory can be put to use.
2. We can randomly sample the data set. This means, we can create a smaller data set, let’s say, having 1000 variables and 300000 rows and do the computations.
3. To reduce dimensionality, we can separate the numerical and categorical variables and remove the correlated variables. For numerical variables, we’ll use correlation. For categorical variables, we’ll use chi-square test.
4. Also, we can use PCA and pick the components which can explain the maximum variance in the data set.
5. Using online learning algorithms like Vowpal Wabbit (available in Python) is a possible option.
6. Building a linear model using Stochastic Gradient Descent is also helpful.
7. We can also apply our business understanding to estimate which all predictors can impact the response variable. But, this is an intuitive approach, failing to identify useful predictors might result in significant loss of information

Q2.You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?

A2.If you have worked on enough data sets, you should deduce that cancer detection results in imbalanced data. In an imbalanced data set, accuracy should not be used as a measure of performance because 96% (as given) might only be predicting majority class correctly, but our class of interest is minority class (4%) which is the people who actually got diagnosed with cancer. Hence, in order to evaluate model performance, we should use Sensitivity (True Positive Rate), Specificity (True Negative Rate), F measure to determine class wise performance of the classifier. If the minority class performance is found to to be poor, we can undertake the following steps:

1. We can use undersampling, oversampling or SMOTE to make the data balanced.
2. We can alter the prediction threshold value by doing [probability caliberation](https://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/) and finding a optimal threshold using AUC-ROC curve.
3. We can assign weight to classes such that the minority classes gets larger weight.
4. We can also use anomaly detection.

Know more: Imbalanced Classification

Q3.While working on a data set, how do you select important variables? Explain your methods.

A3.Following are the methods of variable selection you can use:

1. Remove the correlated variables prior to selecting important variables
2. Use linear regression and select variables based on p values
3. Use Forward Selection, Backward Selection, Stepwise Selection
4. Use Random Forest, Xgboost and plot variable importance chart
5. Use Lasso Regression
6. Measure information gain for the available set of features and select top n features accordingly.

Q4.逻辑回归为什么不能用均方误差计算损失函数

A4.非凸

Q5.K-means 中我想聚成100类 结果发现只能聚成98类，为什么

A5.因为聚类过程中可能会产生空簇，可见[例子](https://blog.csdn.net/shwan_ma/article/details/80096408)

### Stack

#### Algorithm

- 字符串
  - [KMP 算法](https://www.zhihu.com/question/21923021)
- 数组
  - [链表](http://wuchong.me/blog/2014/03/25/interview-link-questions/)
  - [KSum 问题](https://lpq29743.github.io/redant/algorithm/2018/10/29/KSum/)
- 树
- 查找
- 动态规划
- [海量数据处理](https://lpq29743.github.io/redant/algorithm/2017/02/20/MassiveData/)

#### Machine Learning

##### Regression

- Least Squares
- Linear Regression
- Logistic Regression
- Ridge Regression
- Lasso Regression

##### Classification

- kNN
- Naive Bayes
- [SVM](https://lpq29743.github.io/redant/artificialintelligence/2018/09/12/SVM/)

##### Clustering

- Hierarchical Methods
  - BIRCH
  - CHAMELEON
- Partitioning Methods
  - K-Means
- Density-based Methods
  - DBSCAN
- EM

##### Decision Tree

- CART
- ID3
- C4.5

##### Dimension Reducing

- PCA
- SVD

##### Emsemble Learning

- Boosting
- Bagging
- AdaBoost
- Random Forest
- GBDT
- XGBoost

#### Deep Learning

- BP
- MLP
- RBM
- CNN
- LSTM
- GRU
- Attention Mechanism
- Memory Network
- GAN
- VAE

#### Natural Language Processing

##### Feature Engineering

- Bag of Words
- Bag of N-Grams
- TF-IDF
- TextRank
- LDA
- Word2Vec
- Doc2Vec

##### Named Entity Cognition

- HMM
- CRF
- Viterbi Algorithm

### Subjective Question

Q1.告诉我一个你曾经做过的产品，大脑中设想过的也可以

Q2.熟悉哪个算法

Q3.哪门课学的比较好

Q4.你的优缺点

Q5.你用过我们的产品吗？怎么样

### Ask Back

Q1.What will I do here

Q2.What can you offer to help and guide me