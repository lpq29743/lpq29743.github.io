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

Q12.有 n 名球员参加比赛，第一轮抽签两两对决，如果 n 为奇数，则剩下一人轮空，后面的轮次采取类似做法。问总共有几场比赛，轮空次数为多少？

A12.因为总共要淘汰 n - 1 人，所以比赛场数为 n - 1 场。将 n 表示为二进制，则可得轮空次数，如 27 表示为 11011，则轮空次数为除最高位外的 1 的个数，即为 4。

Q13.一瓶可乐两元钱，喝完后两个空瓶可以换一瓶可乐，假设你有40块，请问你最多可以喝到几瓶汽水？

A13.此题与上题类似，每一次换可乐之后都会损失一个瓶盖，所以 20 瓶可乐可以换到 19 瓶可乐，因此总共可以可以喝到 39 瓶可乐。如果开放借瓶盖的话，由于最后我们只剩一个瓶盖，所以最多只能借一个，而且只有当可乐数不是 2 的幂的时候借，因此，如果可以借的话总共可以喝 40 瓶可乐。

Q14.黑色硬币问题

A14.[链接](https://www.bilibili.com/video/av19584232)

Q15.难铺的瓷砖

A15.[链接](https://wenku.baidu.com/view/8605c11452d380eb62946d70.html)

Q16.共 10 瓶药丸。（1）其中一瓶每颗超重 10 毫克；（2）其中多瓶每颗超重 10 毫克。用最少称重数目给出错误的瓶号。

A16.（1）从 1 到 10 瓶，每瓶各拿出 1、2、3、...、10 颗；（2） 从 1 到 10 瓶，每瓶各拿出 1、2、4、...、1024 颗。

Q17.捡麦穗问题

A17.[链接](https://www.zhihu.com/question/66465943)

##### Probability Theory and Mathematical Statistics

Q1.为什么样本方差（sample variance）的分母是 $$n - 1$$？

A1.如果期望已知，分母就是 $$n$$，如果未知，分母是 $$n - 1$$ 是为了保证方差的估计是无偏的。如果直接使用 $$n$$ 为分母作为估计，那么会倾向于低估方差（可用数学方法证明），所以为了正确的估计方差，所以可以把原先的估计值稍微放大一点，即把分母 $$n$$ 改为 $$n - 1$$。

这里也可以用自由度（随机变量中可以同时自由随机变化的变量数目）的角度进行分析。对于 $$n$$ 个样本，由于已经根据这些样本估计了样本均值，因此只剩下 $$n - 1$$ 个样本的值是可以变化的。换句话说，样本中原有的 $$n$$ 个自由度，有一个被分配给计算样本均值，剩下自由度即为 $$n - 1$$，所以用 $$n - 1$$ 作为分母来计算样本方差。

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

Q5.假设有某种病毒，每个病毒每秒钟会以 1/3 的概率分裂成两个病毒，1/3 的概率不分裂（还是一个），1/3 的概率消亡（变成 0 个）。在最初时刻，玻璃罩中有一个病毒，那么最终玻璃罩内没有活着的病毒概率是多大？

A5.[链接](http://sofasofa.io/forum_main_post.php?postid=1000666#)

Q6.一米长的绳子，随机剪两刀，最长的一段有多长

A6.假设三段的长度从小到大依次为 a，a + b，a + b + c，并且满足 a + a + b + a + b + c = 3a + 2b + c = 1 以及 a > 0，b ≥ 0，c ≥ 0

则可以得到 a ≤ 1/3，b ≤ 1/2，c ≤ 1，不妨可以认为 a ∼ U(0, 2k)，b ∼ U(0, 3k)，c∼U(0, 6k)。

绳子最长的一段的期望为 k + 1.5k + 3k = 5.5k，绳子长度的期望为 3k + 3k + 3k = 9k。因为 9k = 1，所以 5.5k = 11/18 = 0.61111

Q7.给定一个 0 到 1 的均匀分布，如何近似地生成一个标准正态分布。即用 numpy.random.uniform() 这个函数， 得到 numpy.random.normal()

A7.本题考点为中心极限定理和均匀分布。中心极限定理即一组相互独立的随机变量的均值符合正态分布。

`np.random.uniform()`生成的是 (0, 1) 之间均匀分布的随机数，则`2 * np.random.uniform() - 1`生成的是 (-1, 1) 之间均匀分布的随机数。

已知 U(a, b) 方差是 (a - b)^2 / 12，则含有 n 个样本的样本均值的方差是 (a - b)^2 / 12 / n。代码如下：

```
import numpy as np
normal_rv = 30 * np.mean(2 * np.random.uniform(size=300) - 1)
```

具体步骤是先产生 300 个 (-1, 1) 随机变量，它们的均值的标准差是 1 / 30，要得到标准正态分布，所以要乘以 30。

Q8.假设一段公路上，1 小时内有汽车经过的概率为96%，那么，30分钟内有汽车经过的概率为

A8.一小时有车的概率 = 1 - 一小时没车的概率 = 1 - 两个半小时都没车的概率 = 1 - (1 - 半小时有车的概率)^2

Q9.一枚不均匀硬币，抛了 100 次，有 70 次朝上，第 101 次朝上的概率是多少，公式是如何推导

A9.7/10。二项分布的极大似然估计，可参考[此链接](https://www.zhihu.com/question/24124998)。

Q10.4个人，52张扑克牌，红桃 A 和黑桃 A 同时被一个人拿到的概率

A10.解法一：C(1,4) * C(11,50) / C(13,52)，C(1,4) = 从四个人中任选 1 人为红桃 A + 黑桃 A，C(11,50) = 从剩余 50 张牌中抽取 11 张给指定人，C(13,52) = 从 52 张牌中随机抽取 13 张

解法二：对于抓到红桃 A 的人，再抓黑桃 A 的概率就是 12/51 = 4/17

Q11.假设有一副被打乱的扑克牌，52张，其中13张黑桃，一个人从这副牌里随机的抽牌，每次抽一张，并且不放回，假设在第X次抽牌的时候，第一次抽到黑桃。请问X的数学期望是多少

A11.[链接](http://sofasofa.io/forum_main_post.php?postid=1000445)

Q12.三个人告诉你深圳下雨了，每个人说谎概率是1/3，那么深圳下雨概率是多少

A12.8/9

Q13.蒲丰投针问题

A13.[链接](https://baike.baidu.com/item/%E8%92%B2%E4%B8%B0%E6%8A%95%E9%92%88%E9%97%AE%E9%A2%98/10876943?fromtitle=%E5%B8%83%E4%B8%B0%E6%8A%95%E9%92%88&fromid=5919098)

Q14.中餐馆过程

A14.[链接](http://sofasofa.io/forum_main_post.php?postid=1003110)

Q15.两个人轮流抛硬币，规定第一个抛出正面的人可以吃到苹果，请问先抛的人能吃到苹果的概率多大？

A15.先抛的人吃到苹果的概率：$$1/2 + 1/2^3 + 1/2^5 + ...$$，求得结果为 $$2/3$$。另一种解法是设先抛先吃的概率为 $$p_1$$， 后抛先吃的概率为 $$p_2$$，有：$$p_1 = 1/2 + 1/2 * p_2$$ 且 $$p_1 + p_2 = 1$$，解方程可得，$$p_1 = 2/3$$。如果题目说是只抛一次的话，则概率为 $$1/2$$。 

Q16.一个骰子，6 面，1 个面是 1， 2 个面是 2， 3 个面是 3， 问平均掷多少次能使 1、2、3 都至少出现一次？

A16.[链接](https://blog.csdn.net/wongson/article/details/7974587)

Q17.什么是点估计，什么是区间估计

A17.点估计是预测参数的值，区间估计是预测参数所处的区间

Q18.什么是置信区间，什么是置信水平/置信度

A18.置信区间是一个带着置信度的估计区间。若置信区间为 $$[a, b]$$，则置信水平 Y% 表示 $$P(a < \mu < b) = Y%$$。常见的置信度为 95%（$$2\sigma$$），95% 置信度表示的是 100 次区间估计，其中约有 95 次区间估计得到的区间结果包含正确的参数值

根据大数定律和中心极限定律，样本均值 $$M \sim N(\mu, \sigma^2/n)$$，其中 $$\mu$$ 为抽样总体分布期望，$$\sigma^2$$ 为抽样总体分布方差，$$n$$ 为样本数目

求置信区间的方式是先计算抽样样本的均值和方差，然后再根据设置的置信区间查表就可以得到置信区间的上下界

Q19.频率派概率和贝叶斯概率有什么区别

A19.频率派概率是最大似然估计，贝叶斯概率是最大后验估计。频率派从自然角度出发，直接为事件建模，即事件 A 在独立重复试验中发生的频率趋于概率 p。贝叶斯派则认为概率是不确定的，需要结合先验概率和似然概率来得到后验概率。随着数据量的增加，参数分布会向数据靠拢，先验的影响越来越小

#### Algorithm

Q1.有10个排好序的数据库，那么我要找整个的中位数，怎么找

A1.最简单的思路是合并数据库，然后再定位长度，时间复杂度为 O(n)，空间复杂度是 O(n)；但实际上只需要借鉴这个合并的过程，当合并到中位数的时候输出中位数即可，时间复杂度为 O(n)，空间复杂度是 O(1)。这思路十分简单，但并不是最佳算法，有序数组让我们想到的会是二分查找，因此我们可以利用二分查找来使复杂度降至 O(logn)，具体可参考：

1. https://www.douban.com/note/177571938/
2. https://stackoverflow.com/questions/6182488/median-of-5-sorted-arrays

Q2.无序整数数组中找第 k 大的数

A2.[链接](https://blog.csdn.net/wangbaochu/article/details/52949443)

Q3.不用库函数求一个数的立方根

A3.[链接](https://blog.csdn.net/sjpz0124/article/details/47726275)

Q4.二进制中 1 的个数

A4.把一个整数减去 1，再和原整数做与运算，会把该整数最右边的 1 变成 0。那么一个整数的二进制表示中有多少个 1，就可以进行多少次这样的操作。具体解题思路可参见《剑指 Offer》

Q5.数值的整数次方

A5.[链接](https://zhuanlan.zhihu.com/p/38715645)

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

Q5.K-means 中我想聚成 100 类 结果发现只能聚成98类，为什么

A5.因为聚类过程中可能会产生空簇，可见[例子](https://blog.csdn.net/shwan_ma/article/details/80096408)

Q6.最小二乘法为什么可以解决线性回归问题

A6.残差满足正态分布时，用最大似然估计法可以证明最小二乘法是合理的

Q7.描述一下最小二乘法的几何意义

A7.最小二乘法中的几何意义是高维空间中的一个向量在低维子空间的投影。$WX$ 实际上是当前样本形成的线性组合空间 $$S$$，最小化的过程是找到一个合适的 $$W$$，使得不在 $$S$$ 上的 $$Y$$ 到 $$S$$ 的投影距离最小

Q8.什么是机器学习

A8.机器学习的目标是让机器能够将非形式化的人类操作传达给机器，从而让机器能够模拟人的一些简单的活动：如识别文本情况、识别图片内容等

Q9.什么是深度学习

A9.深度学习是机器学习的一种，它的目标是试图用神经网络来模拟大脑中的神经元活动，用更深的网络来模拟人的思考

Q10.强化学习和监督学习有什么区别

A10.监督学习中每一个决策（预测标签）是独立的，它对决策的优化取决于标签，强化学习每一个决策是相互影响的，它对决策的优化取决于延时标签（奖励）

Q11.单层感知机为什么不能解决异或问题

A11.因为异或操作需要两条线来划分边界，而单层感知机可以理解为一个线性分类器，只能解决与、或、非问题

Q12.如何对单层感知机进行改进，使其能够解决异或问题

A12.多层感知机，或在进入激活函数前加一个多项式模块，从而添加非线性成分

Q13.KNN 算法的 k 值应该如何选择

A13.k 值太小，模型复杂度较高，容易过拟合；k 值太大，模型复杂度不够，较远的点也可能影响分类结果，分类模糊，导致分类结果不理想。当 k 取训练集大小时，分类结果都为训练集中最多的类。k 值一般选取较小的值，且要低于训练样本数的平方根，可以使用交叉验证法选取

Q14.KNN 算法可以根据距离加权吗

A14.可以用反函数或高斯函数进行距离加权，前者为近邻样本赋予较大权重，稍远的会衰减地很快，因此对噪声数据比较敏感，后者能解决这个问题，但比较复杂

Q15.常见的距离度量方法有哪些

A15.$$L_p$$ 距离 / Minkowski 距离 / 闵式距离是最常规的距离度量方式，其公式为 $$(|x-y|^p)^{1/p}$$。当 $$p = 1$$ 时为曼哈顿距离，$$p = 2$$ 时为欧式距离，$$p$$ 为无穷大时为各个坐标距离的最大值，即切比雪夫距离

Q16.决策树中的特征选择方法有哪些

A16.信息增益、信息增益比和基尼系数

Q17.支持向量机可以用来做回归吗

A17.支持向量机分类是使两类的点在各自的支持向量外，而支持向量机回归是把所以的点都看成一类，并要求在支持向量内

Q18.上溢和下溢是什么，softmax 函数会出现哪种情况，该怎么解决

A18.上溢即大量级的数被近似为正负无穷时，发生上溢。发生上溢后，这些数值会变为非数值。下溢即有些逼近零的数，如零除或者对零取对数时，得到负无穷，如果对负无穷进一步运算，则会得到非数字。softmax 函数中有指数运算，如果要运算的数过小或过大，则会下溢或上溢。解决上溢的方式是让每一个值都减去最大分量的值，由于这样做之后分母有一项为 1，所以不会出现下溢。同样对于取对数，可以让所有数都加 1

### Stack

#### Algorithm

- 字符串
  - [KMP 算法](https://www.zhihu.com/question/21923021)
  - [Edit Distance](https://github.com/youngwind/blog/issues/106)
  - 正则表达式
- 数组
  - [链表1](https://wuchong.me/blog/2014/03/25/interview-link-questions/) [链表2](https://www.jianshu.com/p/1361493e4f31)
  - 前缀、中缀、后缀表达式
- 树
- 查找
  - [KSum 问题](https://lpq29743.github.io/redant/algorithm/2018/10/29/KSum/)
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
- Regularization
- Optimization
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

##### Text Processing

- Word Normalization and Stemming: Normalization, Case folding, Lemmatization, Morphology, Stemming and Porter's algorithm
- Tokenization
- Stop words
- Part-of-Speech Tagging
- Named Entity Recognition

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

##### Application

- Text Classification
- Text Matching: VSM, BM25
- Dialogue System
- Machine Translation

### Subjective Question

Q1.熟悉哪个算法

Q2.哪门课学的比较好

Q3.看过哪些书

Q4.你的优缺点

Q5.你用过我们的产品吗？怎么样

Q6.IAN

Q7.RAM

Q8.意图识别

- [类别不平衡（人为数据增强、损失加权、阈值控制、评价指标）](https://lpq29743.github.io/redant/artificialintelligence/2018/09/26/UnbalancedData/)
- 类别过多
- 数据量过少
- 评价指标（准确率、召回率、F1 值、宏平均、微平均等）
- 过拟合（训练和测试的不一样，允许丢失部分准确率）
- 实体替换、数据预处理、数据读取（pandas）
- 数据增强、同义词替换、回译、文档裁剪、迁移学习
- 规则（规则学习、规则权重、规则严格过滤）和深度结合
- 选择怎样的深度模型
- K-fold 实验、数据集分割、模型融合
- 错误分析

Q9. 指代消解

### Ask Back

Q1.What will I do here

Q2.What can you offer to help and guide me

Q3.转正