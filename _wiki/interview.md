---
layout: wiki
title: Interview
categories: Algorithm
description: 面试题
keywords: 面试题
---

### Math

#### Linear Algebra

1. 行列式（Determinant）

    答：在方块矩阵（n * n）上计算得到的标量。性质：单位矩阵（Identity matrix）行列式为 1；交换行列式两列反号；某列元素扩大 n 倍，行列式扩大 n 倍；矩阵的转置的行列式等于本身的行列式；矩阵相乘的行列式等于两者行列式相乘。矩阵的行列式的几何意义是矩阵对应的线性变换前后的面积比。

2. 计算三个稠密矩阵 A,B,C 的乘积 ABC ,假设三个矩阵的尺寸分别为 m ∗ n，n ∗ p，p ∗ q，且 m <n < p < q，计算顺序效率最高的是 (AB)C 还是 A(BC)？

    答：在 (AB)C 中，m ∗ n 的矩阵 A 和 n ∗ p 的矩阵 B 的乘积，得到 m ∗ p 的矩阵 A * B ，而 A ∗ B 的每个元素需要 n 次乘法和 n-1 次加法，忽略加法，共需要 m ∗ n ∗ p 次乘法运算。同样情况分析 A * B 之后再乘以 C 时的情况，共需要 m ∗ p ∗ q 次乘法运算。因此， (AB)C 需要的乘法次数是 m ∗ n ∗ p + m ∗ p ∗ q 。同理分析 C 选项 A (BC) 需要的乘法次数是 n ∗ p ∗ q + m ∗ n ∗ q。

3. 给一个 3 * 3 的矩阵，求 row-wise cosine similarity

    答：
    ```python
    import numpy as np
    
    X = np.random.randn(3, 3)
    
    # default ord=2
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8  # shape (3,1)
    
    # 归一化矩阵，使每行向量单位化
    X_normalized = X / norms
    
    # 计算行向量之间的余弦相似度（相当于点积）
    cosine_sim = X_normalized @ X_normalized.T
	```

4. 特征值（eigenvalue）和特征向量（eigenvector）

    答：对于方块矩阵（n * n），如果存在数 m 和非零 n 维列向量 x，使得 $$Ax=mx$$ 成立，则称 m 是 A 的一个特征值，而 x 是特征向量。如果把矩阵看作是位移，那么特征值 = 位移的速度，特征向量 = 位移的方向。对称矩阵的秩 = 非零特征值的个数；特征值的乘积 = 矩阵的行列式。上（下）三角矩阵对角线上的值为特征值。如果有特征值为 0，矩阵不可逆（singular，non-invertible）。特征向量不能为 0。

5. Python 相关实现

    答：
    ```python
	import numpy as np
	from numpy.linalg import norm
	from scipy import linalg as la  # 可选：更强大的分解/求解
	
	# 向量与矩阵
	a = np.array([1., 2., 3.])
	b = np.array([4., 5., 6.])
	A = np.array([[1., 2.], [3., 4.]])
	B = np.array([[2., 0.], [1., 2.]])
	
	# 基本运算
	dot_ab = a @ b                # 点积
	matmul = A @ B                # 矩阵乘
	AT = A.T                      # 转置
	fro = norm(A, 'fro')          # Frobenius 范数
	l2 = norm(a, 2)               # 向量L2范数
	
	# 线性方程组 Ax = y
	y = np.array([1., 0.])
	x = np.linalg.solve(A, y)     # 解 x
	
	# 行列式 / 逆
	detA = np.linalg.det(A)
	Ainv = np.linalg.inv(A)
	
	# 特征分解（对称阵优先用 eigh）
	w, V = np.linalg.eig(A)       # A V = V diag(w)
	w_sym, U = np.linalg.eigh((A + A.T)/2)  # 对称阵更稳
	
	# SVD
	U_svd, S, VT = np.linalg.svd(A, full_matrices=False)
	
	# QR / LU / Cholesky（SciPy）
	Q, R = la.qr(A)
	P, L, Ulu = la.lu(A)
	C = la.cholesky(np.array([[4.,1.],[1.,3.]]))  # 正定矩阵
	```

#### Probability and Statistics

1. 给一枚硬币，但扔（flip）出正反（head and tail）的概率未知，如何得到等概率的二元随机数？

    答：扔两次，00、11 时无输出重扔，01 输出 0，10 输出 1。

2. 抛的硬币直到连续出现两次正面为止，平均要扔多少次？

    答：用马尔可夫链，可做图求解递归方程。
    
    假定扔出正面 (H) 的概率为 p，扔出反面 (T) 的概率为 1 - p。我们需要扔出连续 2 个 H。在整个过程有这么几种状态：
    
    a. 当前连续 0 个正面（0H）；
    
    b. 当前连续 1 个正面（1H）；
    
    c. 当前连续 2 个正面（2H）。
    
    如果当前是 0H，那么 p 的概率，下一个状态是 1H；1 - p 的概率维持在 0H。
    
    如果当前是 1H，那么 p 的概率，下一个状态为 2H（达到条件，任务完成）；1 - p 的概率回到 0H。
    
    假设期望 x 次后，得到 2H，则有 $$x = (1 − p)(1 + x) + p^2 × 2 + p(1 − p)(2 + x)$$，当 p = 0.5，可解得 x = 6。

3. 抛的硬币直到可以丢出正反面为止，平均要扔多少次？

    答：第一次抛肯定出现一个面，期望为 1 + 1/另一个面的概率。如果概率为 0.5，期望为 3。

4. 一枚不均匀硬币，抛了 100 次，有 70 次朝上，第 101 次朝上的概率是多少，公式是如何推导？

    答：7/10。二项分布的极大似然估计，可参考[此链接](https://www.zhihu.com/question/24124998)。

5. 两个人轮流抛硬币，规定第一个抛出正面的人可以吃到苹果，请问先抛的人能吃到苹果的概率多大？

    答：先抛的人吃到苹果的概率：$$1/2 + 1/2^3 + 1/2^5 + ...$$，求得结果为 $$2/3$$。另一种解法是设先抛先吃的概率为 $$p_1$$， 后抛先吃的概率为 $$p_2$$，有：$$p_1 = 1/2 + 1/2 * p_2$$ 且 $$p_1 + p_2 = 1$$，解方程可得，$$p_1 = 2/3$$。如果题目说是只抛一次的话，则概率为 $$1/2$$。 

6. 如何用一个骰子（dice）等概率地生成 1 到 7 的随机数？

    答：将一个筛子扔两次可以得到 36 种组合（进制思维），每五种组合代表一个数字，剩下的一种表示重扔。

7. 一个骰子，6 面，1 个面是 1， 2 个面是 2， 3 个面是 3， 问平均掷多少次能使 1、2、3 都至少出现一次？

    答：加权 Coupon Collector 问题，期望次数公式为：
    
    $$E[T] = \sum_{i=1}^{3} \frac{1}{p_i} - \sum_{i<j} \frac{1}{p_i + p_j} + \frac{1}{p_1 + p_2 + p_3}$$，代入数值得到 E[T] = 11 - 4.7 + 1 = 7.3。

8. 假设有某种病毒，每个病毒每秒钟会以 1/3 的概率分裂成两个病毒，1/3 的概率不分裂（还是一个），1/3 的概率消亡（变成 0 个）。在最初时刻，玻璃罩中有一个病毒，那么最终玻璃罩内没有活着的病毒概率是多大？

    答：假设从一个病毒最终到没有病毒的概率为p。根据马尔可夫链，p=1/3 * p^2 +1/3 * p + 1/3。解上面的方程，可得p=1。概率是1，也就是病毒最终肯定会都死光。

9. 一米长的绳子，随机剪两刀，最长的一段有多长？

    答：假设三段的长度从小到大依次为 a，a + b，a + b + c，并且满足 a + a + b + a + b + c = 3a + 2b + c = 1 以及 a > 0，b ≥ 0，c ≥ 0。

    则可以得到 a ≤ 1/3，b ≤ 1/2，c ≤ 1，不妨可以认为 a ∼ U(0, 2k)，b ∼ U(0, 3k)，c∼U(0, 6k)。

    绳子最长的一段的期望为 k + 1.5k + 3k = 5.5k，绳子长度的期望为 3k + 3k + 3k = 9k。因为 9k = 1，所以 5.5k = 11/18 = 0.61111。

10. 一根绳子切成三段，组成三角形的几率是多少？

    答：设绳长为12，分成的三段分别为x，y，12-x-y，且x > y > 12-x-y，则 x，y 应满足以下 5 条关系：x+y < 12，x>0，y>0，x > y, y > 12-x-y，在平面直角坐标系中是以(12,0), (6,6), (4,4)为顶点的三角形区域，易求出面积等于12。
    
    由于 x > y > 12-x-y，只需再满足 x<6，这三段就能构成三角形，即在上述 5 条关系后再加上第 6条：x<6，组成了以 (6,3), (6,6), (4,4) 为顶点的三角形区域,易求出面积等于 3。构成三角形的概率是 3/12=1/4。

11. 假设一段公路上，1 小时内有汽车经过的概率为 96%，那么，30 分钟内有汽车经过的概率为？

    答：一小时有车的概率 = 1 - 一小时没车的概率 = 1 - 两个半小时都没车的概率 = 1 - (1 - 半小时有车的概率)^2。

12. 4 个人，52 张扑克牌，红桃 A 和黑桃 A 同时被一个人拿到的概率？

    答：解法一：C(1,4) * C(11,50) / C(13,52)，C(1,4) = 从四个人中任选 1 人为红桃 A + 黑桃 A，C(11,50) = 从剩余 50 张牌中抽取 11 张给指定人，C(13,52) = 从 52 张牌中随机抽取 13 张；

    解法二：对于抓到红桃 A 的人，再抓黑桃 A 的概率就是 12/51 = 4/17。

13. 假设有一副被打乱的扑克牌，52 张，其中 13 张黑桃，一个人从这副牌里随机的抽牌，每次抽一张，并且不放回，假设在第X次抽牌的时候，第一次抽到黑桃。请问 X 的数学期望是多少?

    答：Negative Hypergeometric Distribution：N 个元素中有 K 个正元素，不放回地（without replacements）随机取一个元素，直到有 r 个负元素被取出，其中有 k 个正元素。在这个问题里N=52，负元素是黑桃，K=52-13=39，r=1，k+1 是步数。按均值公式 E(k)=rK/(N−K+1)=39/14，最后步数的期望是 E(k)+1=53/14≈3.79。

14. 三个人告诉你深圳下雨了，每个人说谎概率是1/3，那么深圳下雨概率是多少?

    答：8/9。

15. 幼儿园 10 个小朋友排成一列，其中 3 个小朋友是女孩，求女孩排在一起的概率？

    答：所有人排列是 10!，三个小女孩看做整体排列是 8!，三个小女孩排列是 3!，最后结果就是 (8! \* 3!) / 10! = 1/15。

16. 村长带着 4 对父子参加爸爸去哪儿第三季第二站某村庄的拍摄。村里为了保护小孩不被拐走有个前年的规矩，那就是吃饭的时候小孩左右只能是其他小孩或者自己的父母。那么 4 对父子在圆桌上共有几种坐法？

    答：FFFFBBBB 模式 $$8*4*3*2*2=384$$，FFBBFFBB 模式,这种模式下旋转 4 个位置还是这种模式，该模式下总数为 $$4*4*3*2=96$$，总和为 $$384*96=480$$。

17. 一个点用偶数步（2n）从原点出发回到原点有几种走法？

    答：想象把坐标轴旋转 45 度，那么每一步相当于在水平方向和竖直方向各自独立移动一个单位。那么走 2n 步回到原点的方案数为 (C(2n，n))^2。

18. 离散型随机变量

    答：伯努利分布  (Bernoulli Distribution)：伯努利试验，只有两个结果
    $$X = 0,1$$
	$$P(x) = 
	\begin{cases}
	p, \quad \quad \quad  x= 1\\
	1-p , \quad \quad x = 0
	\end{cases}
	\\
	 = p^x(1-p)^{1-x}
	$$
	
	期望
	$$
	E(X) = p*2 + (1-p)*0 = p
	$$
	
	方差
	$$
	Var(X) = E(X^2) - (E(X)^2) = p(1-p)
	$$
	
	二项分布 (Binomial Distribution)：n 次独立相同试验，结果只有两种可能，每次概率相同为 p
	
	$$P(X=k) = C^k_np^k(1-p)^{n-k}, k=0,1,2,...$$
	
	期望
	$$E(X)=np$$
	
	方差
	$$Var(X)=np(1-p)$$
	
	泊松分布 (Possion Distribution)：泊松分布测定在单位空间或时间中某事件发生的次数。如某一服务设施在一定时间内受到的服务请求的次数，电话交换机接到呼叫的次数、汽车站台的候客人数、机器出现的故障数、自然灾害发生的次数、DNA序列的变异数、放射性原子核的衰变数、激光的光子数分布等等。
	$$
	X=0,1,2...
	\\
	P(X=k)=\frac{\lambda^{k}e^{-\lambda}}{k!},k=0,1,2...
	$$
	其中 $$\lambda$$ 是单位时间或空间中事件发生次数的数学期望。记作 $$X\sim Possion(\lambda)$$
	
	期望
	$$E(X)=\lambda$$
	
	方差
	$$Var(X)=\lambda$$

19. 连续型随机变量

    答：连续型随机变量一般用概率密度函数 f（Probability Density Function，PDF）表示，其值大于等于 0，其在负无穷到正无穷之间取积分为 1。对于连续型随机变量，某点概率没有意义，或者说某点的概率为 0，一般求区间概率，即 PDF 从 a 到 b 的积分。另外一种常用的表示连续型随机变量的方式是累积分布函数 F（Cumulative Distribution Function，CDF），其每一个点的值表示从负无穷大到该点的概率，因此是非减函数，其求区间概率为 F(b) - F(a)。常见分布有均匀分布、指数分布和正态分布。

20. 两个 (0, 1) 均匀分布的和是怎样子的？

    答：[链接](https://www.statlect.com/fundamentals-of-probability/sums-of-independent-random-variables?utm_source=chatgpt.com)

21. 频率派概率（Frequentist）和贝叶斯概率（Bayesian）有什么区别？

    答：频率派概率是最大似然估计，贝叶斯概率是最大后验估计。频率派从自然角度出发，直接为事件建模，即事件 A 在独立重复试验中发生的频率趋于概率 p。贝叶斯派则认为概率是不确定的，需要结合先验概率和似然概率来得到后验概率。随着数据量的增加，参数分布会向数据靠拢，先验的影响越来越小。

22. 先验概率（Prior）是什么？后验概率（Posterior）是什么？

    答：$$p(\theta \| x)=\frac{p(x \| \theta)p(\theta)}{p(x)}$$。$$x$$ 为观察得到的数据（结果），$$\theta$$ 为决定数据分布的参数（原因），$$p(\theta \| x)$$ 为后验分布，$$p(\theta)$$ 为先验分布，$$p(x \| \theta)$$ 为似然。

23. 高维空间下的点有什么特点？

    答：
    - 随着维度升高，点的分布会变得稀疏，任意两点的距离会变大。
    - 距离会变得差不多。距离的相对方差为 1/d，d 为维度，随着维度升高，方差趋于 0。
    - 集中在表面。从低维推到高维空间，体积会越来越由表面附近区域贡献。
    - 几乎正交

24. 为什么样本方差（sample variance）的分母（denominator）是 $$n - 1$$？

    答：如果期望已知，分母就是 $$n$$，如果未知，分母是 $$n - 1$$ 是为了保证方差的估计是无偏的（unbiased）。如果直接使用 $$n$$ 为分母作为估计，那么会倾向于低估方差（可用数学方法证明），所以为了正确的估计方差，所以可以把原先的估计值稍微放大一点，即把分母 $$n$$ 改为 $$n - 1$$。
    
    这里也可以用自由度（随机变量中可以同时自由随机变化的变量数目，Degree of Freedom）的角度进行分析。对于 $$n$$ 个样本，由于已经根据这些样本估计了样本均值，因此只剩下 $$n - 1$$ 个样本的值是可以变化的。换句话说，样本中原有的 $$n$$ 个自由度，有一个被分配给计算样本均值，剩下自由度即为 $$n - 1$$，所以用 $$n - 1$$ 作为分母来计算样本方差。

25. 给定一个 0 到 1 的均匀分布，如何近似地生成一个标准正态分布。即用 numpy.random.uniform() 这个函数， 得到 numpy.random.normal()？

    答：本题考点为中心极限定理（Central Limit Theorems）和均匀分布。中心极限定理即一组相互独立的随机变量的均值符合正态分布。

    `np.random.uniform()`生成的是 (0, 1) 之间均匀分布的随机数，则`2 * np.random.uniform() - 1`生成的是 (-1, 1) 之间均匀分布的随机数。

    已知 U(a, b) 方差是 (a - b)^2 / 12，则含有 n 个样本的样本均值的方差是 (a - b)^2 / 12 / n。代码如下：

    ```python
    import numpy as np
    normal_rv = 30 * np.mean(2 * np.random.uniform(size=300) - 1)
    ```

    具体步骤是先产生 300 个 (-1, 1) 随机变量，它们的均值的标准差是 (1 - (-1))^2 / 12 * 300 = 1 / 30，要得到标准正态分布，所以要乘以 30。

26. 请解释协方差矩阵的含义

    答：协方差矩阵描述多变量间的线性关系，矩阵的每个元素代表两个变量的协方差。协方差大于 0，两个变量正相关，反之负相关，为 0 时不相关。Pearson 系数是协方差除以两个变量的标准差的标准化版本。

27. 什么是点估计，什么是区间估计？

    答：点估计是预测参数的值，区间估计是预测参数所处的区间。

28. 什么是置信区间，什么是置信水平/置信度？

    答：置信区间是一个带着置信度的估计区间。若置信区间为 $$[a, b]$$，则置信水平 Y% 表示 $$P(a < \mu < b) = Y\%$$。常见的置信度为 95%（$$2\sigma$$），95% 置信度表示的是 100 次区间估计，其中约有 95 次区间估计得到的区间结果包含正确的参数值。

    根据大数定律和中心极限定律，样本均值 $$M \sim N(\mu, \sigma^2/n)$$，其中 $$\mu$$ 为抽样总体分布期望，$$\sigma^2$$ 为抽样总体分布方差，$$n$$ 为样本数目。

    求置信区间的方式是先计算抽样样本的均值和方差，然后再根据设置的置信区间查表就可以得到置信区间的上下界。

29. 什么是卡方检验（Chi-squared test）？

    答：卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，如果卡方值越大，二者偏差程度越大；反之，二者偏差越小；若两个值完全相等时，卡方值就为0，表明理论值完全符合。
    
    常用类型 1：适合度检验，即检验单个分类变量的分布是否符合某种理论分布，如掷骰子 60 次，观测 6 个面的出现次数是否均匀。
    $$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$，其中：$$O_i$$ 表示第 i 类的观察频数，$$E_i$$ 表示第 i 类的期望频数，$$k$$ 表示类别数，$$chi^2$$ 为卡方统计量。
    
    常用类型 2：独立性检验，即检验两个分类变量是否独立，如性别（男/女）和是否喜欢某品牌（喜欢/不喜欢）是否相关。
    $$chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$。期望频数计算公式：$$E_{ij} = \frac{R_i \cdot C_j}{N}$$，其中 $$O_{ij}$$ 表示第 i 行、第 j 列的观察频数，$$E_{ij}$$ 表示第 i 行、第 j 列的期望频数，$$R_i$$ 为第 i 行合计 $$C_j$$ 为第 j 列合计，$$N$$ 为总样本数，$$r$$ 为行数，$$c$$ 为列数，$$\chi^2$$ 为卡方统计量。

30. 如何利用计算机求 π？

    答：蒙特卡洛随机法（monte carlo simulation）。随机生成点 (x, y)，分布在单位正方形内；统计落在单位圆内的点的比例；利用公式估算 π。

31. 蒲丰投针问题

    答：[链接](https://baike.baidu.com/item/%E8%92%B2%E4%B8%B0%E6%8A%95%E9%92%88%E9%97%AE%E9%A2%98/10876943?fromtitle=%E5%B8%83%E4%B8%B0%E6%8A%95%E9%92%88&fromid=5919098)

32. 中餐馆过程

    答：[链接](http://sofasofa.io/forum_main_post.php?postid=1003110)

33. Python 相关实现

    答：
    ```python
	import numpy as np
	import math
	
	# === Discrete ===
	def bernoulli_pmf(x, p):
	    return p**x * (1-p)**(1-x)
	
	def bernoulli_sample(p, size=1):
	    return np.random.binomial(1, p, size)
	
	def binomial_pmf(k, n, p):
	    return math.comb(n, k) * (p**k) * ((1-p)**(n-k))
	
	def binomial_sample(n, p, size=1):
	    return np.random.binomial(n, p, size)
	
	def poisson_pmf(k, lam):
	    return math.exp(-lam) * lam**k / math.factorial(k)
	
	def poisson_sample(lam, size=1):
	    return np.random.poisson(lam, size)
	
	# === Continuous ===
	def normal_pdf(x, mu=0, sigma=1):
	    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
	
	def normal_sample(mu=0, sigma=1, size=1):
	    return np.random.normal(mu, sigma, size)
	
	def exponential_pdf(x, lam):
	    return lam * np.exp(-lam * x) if x >= 0 else 0
	
	def exponential_sample(lam, size=1):
	    return np.random.exponential(1/lam, size)
	
	# === Experiments ===
	def law_of_large_numbers(p=0.5, N=100000):
	    samples = bernoulli_sample(p, N)
	    return samples.mean()
	
	def central_limit_theorem(M=10000):
	    sums = np.sum(np.random.rand(M, 12), axis=1) - 6  # ≈ N(0,1)
	    return sums.mean(), sums.var()
	
	def monte_carlo_pi(N=1000000):
	    xy = np.random.rand(N, 2)*2 - 1
	    inside = (xy[:,0]**2 + xy[:,1]**2 <= 1).sum()
	    return 4 * inside / N
	```

#### Calculus and Optimization

1. 什么情况下，局部最优为全局最优？

    答：凸优化。

2. 一阶优化和二阶优化

    答：一阶优化（如梯度下降）对应 Jacobian 矩阵，只能告诉你下降方向，但无法告诉你到底走多远最合适，收敛速度通常是线性的（误差大约每次乘一个常数）。二阶优化（如牛顿法）对应 Hessian 矩阵，Hessian 告诉你曲率（曲面的“陡峭程度”），更新步长自动调整，不需要手动选择学习率，迭代时误差平方级（quadratic）下降 → 二阶收敛。当矩阵正定（positive definite，特征值全部大于 0 的实对称矩阵或复 Hermitian 矩阵），梯度为 0 处为极小值。当 Hessian 为常数时，牛顿法可一次求解（如二次方程）。

3. 深度学习为什么不用二阶优化？

    答：有些方法可用，但总体不适用。原因：计算量大，训练慢；求导复杂；深度学习不需要高精度解；稳定性差。

4. 什么是 Jensen 不等式？

     答：Jensen 不等式刻画了凸函数的一个性质。假如 $$f(X)$$ 是个凸函数，对于一个随机变量 $$X$$，那么$$E[f(X)]≥f(E[X])$$，当等号成立时，函数是一条直线。

5. Python 实现的梯度下降和牛顿法示例

     答：以 $$f(x)=x^2 - 4x + 4$$ 为例。
     ```python
	import numpy as np
    
	# === 目标函数 ===
	def f(x):
	    return x**2 - 4*x + 4
	
	# 一阶导数
	def grad_f(x):
	    return 2*x - 4
	
	# 二阶导数 (Hessian)
	def hessian_f(x):
	    return 2  # 对一维函数就是常数
	
	# === 梯度下降 ===
	def gradient_descent(x0, lr=0.1, tol=1e-6, max_iter=100):
	    x = x0
	    for _ in range(max_iter):
	        x_new = x - lr * grad_f(x)
	        # 参数几乎不再变化了，算法可以认为已经收敛，提前停止迭代。
	        if abs(x_new - x) < tol:
	            break
	        x = x_new
	    return x, f(x)
	
	# === 牛顿法 ===
	def newton_method(x0, tol=1e-6, max_iter=100):
	    x = x0
	    for _ in range(max_iter):
	        grad = grad_f(x)
	        hess = hessian_f(x)
	        x_new = x - grad / hess
	        # 参数几乎不再变化了，算法可以认为已经收敛，提前停止迭代。
	        if abs(x_new - x) < tol:
	            break
	        x = x_new
	    return x, f(x)
	
	# === 测试 ===
	x0 = 0.0
	x_gd, f_gd = gradient_descent(x0)
	x_newton, f_newton = newton_method(x0)
	
	print("梯度下降结果: x =", x_gd, ", f(x) =", f_gd)
	print("牛顿法结果: x =", x_newton, ", f(x) =", f_newton)
	```

#### Information Theory

1. 互信息是什么？

     答：$$I(X; Y) = \sum_{y \in Y}\sum_{x \in X} {p(x,y)log{\frac{p(x,y)}{p(x)p(y)}}}$$。当变量相互独立时，互信息为 0。

2. KL 散度（KL divergence）和交叉熵（cross-entropy）的区别？

     答：自信息（一个事件的信息量）：$$I(x)=-logP(x)$$；

     信息熵（一个分布的信息量）：$$H(x)=E_{X \sim P}[I(x)]=-E_{X \sim P}[P(x)]$$；

     交叉熵（在给定的真实分布下，使用非真实分布所指定的策略需要的代价）：$$-\sum_{k=1}^N{p_klog_2{q_k}}$$；

     交叉熵越低，策略越好，当交叉熵 = 信息熵，表示使用了真实分布；

     相对熵 / KL 散度（两个分布之间的差异）：$$KL(f(x) \| g(x))=\sum_{x \in X} {f(x) * log_2{\frac{f(x)}{g(x)}}}$$；

     相对熵 = 交叉熵 - 信息熵。

3. 为什么 KL 散度不对称（non-symmetric）？

     答：KL 散度不对称的根本原因在于其定义中的对数函数位置（分子 numerator 和分母 denominator 位置）的非对称性。它本质上是一个“相对熵”：衡量一个分布在用另一个分布进行近似时，所造成的信息损失。

4. Python 相关实现

     答：对大于 0 的判断是为了避免分母为0，log(0) 等非法操作，保证数值稳定性。
     ```python
    import numpy as np

	# === 1. 熵 (Entropy) ===
	def entropy(p):
	    """
	    p: 概率分布数组，p_i >=0 且 sum(p)=1
	    """
	    p = np.array(p)
	    p = p[p > 0]  # 避免 log(0)
	    return -np.sum(p * np.log2(p))
	
	# === 2. 联合熵 (Joint Entropy) ===
	def joint_entropy(p_xy):
	    """
	    p_xy: 联合概率分布矩阵，sum(p_xy)=1
	    """
	    p_xy = np.array(p_xy)
	    p_xy = p_xy[p_xy > 0]
	    return -np.sum(p_xy * np.log2(p_xy))
	
	# === 3. 条件熵 (Conditional Entropy) H(Y|X) ===
	def conditional_entropy(p_xy):
	    """
	    p_xy: 联合概率分布矩阵
	    H(Y|X) = H(X,Y) - H(X)
	    """
	    p_xy = np.array(p_xy)
	    p_x = p_xy.sum(axis=1)
	    H_xy = joint_entropy(p_xy)
	    H_x = entropy(p_x)
	    return H_xy - H_x
	
	# === 4. 互信息 (Mutual Information) ===
	def mutual_information(p_xy):
	    """
	    I(X;Y) = H(X) + H(Y) - H(X,Y)
	    """
	    p_xy = np.array(p_xy)
	    p_x = p_xy.sum(axis=1)
	    p_y = p_xy.sum(axis=0)
	    H_x = entropy(p_x)
	    H_y = entropy(p_y)
	    H_xy = joint_entropy(p_xy)
	    return H_x + H_y - H_xy
	
	# === 5. KL 散度 (Kullback-Leibler Divergence) ===
	def kl_divergence(p, q):
	    """
	    D_KL(p || q) = sum p_i * log(p_i / q_i)
	    """
	    p = np.array(p)
	    q = np.array(q)
	    mask = (p > 0) & (q > 0)
	    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
	
	# === 示例 ===
	if __name__ == "__main__":
	    p = [0.2, 0.5, 0.3]
	    q = [0.1, 0.7, 0.2]
	    print("Entropy H(p):", entropy(p))
	    print("KL(p||q):", kl_divergence(p, q))
	
	    # 联合概率矩阵
	    p_xy = np.array([[0.1, 0.2],
	                     [0.3, 0.4]])
	    print("Joint Entropy H(X,Y):", joint_entropy(p_xy))
	    print("Conditional Entropy H(Y|X):", conditional_entropy(p_xy))
	    print("Mutual Information I(X;Y):", mutual_information(p_xy))
    ```

#### Discrete Mathematics

1. 有 1000 个一模一样的瓶子，其中有 999 瓶是普通的水，有一瓶是毒药。任何喝下毒药的生物都会在一星期之后死亡。现在，你只有 10 只小白鼠和一星期的时间，如何检验出哪个瓶子里有毒药？

    答：可以用二进制编码的思维来解决。先对 1000 个瓶子以二进制数的形式进行编号，则至少需要十位的二进制数（Binary number）进行表示。再用这10只小白鼠分别对应二进制数的 10 个数位，让每只小白鼠喝下编号对应数位值为 1 的瓶子。最后根据小白鼠的死亡情况得到一个十位二进制数，编号与其相等的瓶子里面有毒药。

2. 1000 桶水，其中一桶有毒，猪喝毒水后会在 15 分钟内死去，想用一个小时找到这桶毒水，至少需要几头猪？

    答：由于一只猪总共有五种状态，所以可以用五进制的方法解决，将 1000桶水表示成五进制至少需要 5 位，所以至少需要五头猪。

3. 1000 桶水中两桶有毒，猪喝毒水后会在 15 分钟内死去，想用一个小时找到毒水，至少需要几只猪？如何实现？

    答：从理论上来讲，1000 桶水两桶有毒，有 499500 种状态，而一头猪有 5 种状态，求解 5^l > 499500，求得 l > 8.15，所以 9 只猪就可以找到毒水：
    
    a. 我们先把1000桶水分为10组，每只猪试验一组，一组空白；
    
    b. 如果只有一只猪死，那么说明两桶毒水都在那 100 桶水中，再把这 100 桶水分为 10 组，剩下的 9 只猪试验 9 组，另外 10 桶不实验，这样问题可一步一步缩小；
    
    c. 如果有两只猪死，则说明有两组 100 桶水各有一桶毒水，每组用 3 头猪即可实验。

4. 一个村子里有很多户人家，每家都养了一条狗。现在发现村子里面出现了 n 只疯狗，村里规定，谁要是发现了自己的狗是疯狗，就要将自己的狗枪毙。但问题是，村子里面的人只能看出别人家的狗是不是疯狗，而不能看出自己的狗是不是疯的，如果看出别人家的狗是疯狗，也不能告诉别人。于是大家开始观察，第一天晚上，没有枪声，第二天晚上，没有枪声，直到第 n 天晚上，这 n 只狗才被同时枪毙，请问这是为什么？

    答：具体分析流程如下：
    
    a. 首先从只有一条疯狗分析起，如果只有一条疯狗，那么疯狗的主人在第 1 天就会发现其他家庭一只疯狗都没有，从而枪毙自己家的狗；
    
    b. 如果有两条疯狗，那么拥有疯狗的家庭在第一天由于看到别人家有疯狗，就不会枪毙自己家的狗，但发现第一晚没人枪毙自己的狗后，他会知道村子里有两条疯狗，其中一条就是自己的。实际上，整个村子的人都会在第一晚没人枪毙自己的狗后，知道整个村子至少有两条疯狗；
    
    c. 继续分析，如果第二晚还没有枪声，那说明拥有疯狗的人都看到了至少两条疯狗，所以他不会认为自己拥有疯狗，但经过没有枪声的第二晚后，全村人便达成了村子至少有三条疯狗的事实；
    
    d. 同理可得，拥有疯狗的家庭由于能看到 n - 1 条狗，他需要 n 天才能判断村子里至少有 n 条疯狗，其中一条就是自己家的狗，从而在第 n 天晚上枪毙自己家的狗。

5. 五个同事决定计算他们的平均工资，在大家互相不告诉薪水的情况下，如何才能做到这一点？

    答：这道题的方法有很多，比如有：
    
    a. 每个人把自己的工资随意拆成四个数的和，分别把四个数字告诉自己以外的四个人；每个人手里收到四个数字，加起来，报出；五个人的数字相加，即可得到五个人的总收入，除以5得到平均工资；
    
    b. 找个计算器，叫第一个人输入一个巨大的不规则的数，然后把自己的收入加上去，之后依次传给其他人，每人都把自己的收入加上之前的数。最后传回第一个人。第一个人再把最后的总和减去他选中的那个不规则数，然后除以人数，即可得到大家的平均。

6. 圆桌上有 1 到 1000 号，1 号右手边是 2 号，左手边是 1000 号。1 号开枪打死 2 号，把枪交给 3 号，3 号打死 4 号交给5号。。999 号打死 1000 号后把枪交给 1 号，继续循环。最后留下来的是几号？

    答：约瑟夫环问题，套公式 $$f(n) = 2(n - 2^{log_2n}) + 1$$ 直接得到结果为 977。

7. 一幢 200 层的大楼，给你两个鸡蛋。如果在第 n 层扔下鸡蛋，鸡蛋不碎，那么从前 n-1 层扔鸡蛋都不碎。这两只鸡蛋一模一样，不碎的话可以扔无数次。已知鸡蛋在 0 层扔不会碎。提出一个策略，要保证能测出鸡蛋恰好会碎的楼层，并使此策略在最坏情况下所扔次数最少？

    答：这是一道非常经典的面试题，我们用两种方法来解决：
    
    a. 分析法：对于每一次扔鸡蛋，都可以看作是一次决策，所以最终扔的方案应该是构成一棵决策树，问题就可以转换成求最矮决策树的高度。假设第一次扔的楼层是第 k 层楼，则碎子树的高度为 k - 1，如果第一次扔鸡蛋没碎，则设第二次扔的高度为 m，则对于 m 来讲，其碎子树高度为 m - k - 1，相对根节点高度则为 m - k。由于要尽可能保证子树的高度一致，所以得 m - k = k - 1，故可得第二次扔的高度要比前一次高 k - 1 层。从而得到方程 k(k + 1) / 2 = 200，从而解得高度为 14；
    
    b. 动态规划法：这道题是很经典的动态规划问题，设楼层次数为 n，我们可以得到状态转移方程`f(n) = min(max(k-1, f(n - k))) + 1 (0 < k <= n)`，如果我们再加入鸡蛋数目变量 m，则状态转移方程为`f(n, m) = min(max(f(k - 1, m - 1), f(n - k, m))) + 1 (0 < k <= n)。`

8. 16 个硬币，A 和 B 轮流拿走一些，每次拿走的个数只能是 1，2，4 中的一个数。谁最后拿硬币谁输。请问：A 或 B 有无策略保证自己赢？

    答：B 可以保证自己赢。 如果 A 拿 1 个，则 B 拿2个； 如果 A 拿 2 个，则 B 拿 1 个； 如果 A拿 4 个，则 B 拿 2 个。这样每次 AB 加起来都是 3 或者 6，所以最后会剩下 1 个或 4 个。 如果是 1 个则 A 直接输了； 如果剩下 4 个，A 全拿则输了，如果不全拿，B继续采取上面的策略，最后还是剩下 1 个，还是 A 输。

9. Nim 问题?

    答：必败态当且仅当所有堆硬币的数量都异或起来结果为 0。

10. 海盗博弈

    答：[链接1](https://www.zhihu.com/question/20014343)、[链接2](https://zhuanlan.zhihu.com/p/27388049)、[链接3](https://www.zhihu.com/question/47973941)

11. 五个囚犯先后从100颗绿豆中抓绿豆。抓得最多和最少的人将被处死，不能交流，可以摸出剩下绿豆的数量，谁的存活几率最大？

    答：[链接](https://www.zhihu.com/question/19912025)

12. 三个极度嫉妒的人分一个蛋糕，采用什么策略，能让三人都觉得公平？

    答：[链接](https://www.zhihu.com/question/20615717)

13. 有 5 个直线排列的洞，兔子每天换一个相邻的洞，每天插一次，怎样做一定能找到兔子，次数不限。

    答：第一天，守住第二个洞，没有的话，第二天还守住第二个洞，还没有的话，一二洞可排除，兔子前两天一定不会在一二两个洞，第三天检查第三个洞，如果没有，此时，兔子只可能在第二四五三个洞，第四天检查第四个洞，如果还没有，说明只可能在一三五三个洞，第五天还检查第四个洞，如果没有的话，说明兔子不可能在第四五个洞，也不可能在第一三个洞，只可能在第二个洞，第六天检查第三个洞，如果没有，第七天一定可以在第二个洞里抓到它的。

14. 64 匹马，8 个赛道，找出跑得最快的 4 匹马，至少比赛几场？ 

    答：[链接](https://blog.csdn.net/madefromme/article/details/81584325)

15. 有 8 个台球，其中一个比其他的 7 个都要重一些。如果仅仅是使用天平而不称出具体的重量，请问最少几次能找出那个最重的台球？

    答：2 次。把所有的球分成 3 组，其中 2 组是 3 个球，最后一组是两个球。首先，把 3 个球的两组放在天平上。如果其中一方比较重，把偏重的那一组球任意拿出来两个放在天平上。如果两组球一样重，那就把剩下的 2 个球放在天平上称重。

16. 有 n 名球员参加比赛，第一轮抽签两两对决，如果 n 为奇数，则剩下一人轮空，后面的轮次采取类似做法。问总共有几场比赛？

    答：因为总共要淘汰 n - 1 人，所以比赛场数为 n - 1 场。

17. 一瓶可乐两元钱，喝完后两个空瓶可以换一瓶可乐，假设你有40块，请问你最多可以喝到几瓶汽水？

    答：此题与上题类似，每一次换可乐之后都会损失一个瓶盖，所以 20 瓶可乐可以换到 19 瓶可乐，因此总共可以可以喝到 39 瓶可乐。如果开放借瓶盖的话，由于最后我们只剩一个瓶盖，所以最多只能借一个，而且只有当可乐数不是 2 的幂的时候借，因此，如果可以借的话总共可以喝 40 瓶可乐。

18. 0， 6， 24，60， 120，（）?

    答：0=0\*1\*2；6=1\*2\*3；24=2\*3\*4；60=3\*4\*5；120=4\*5\*6；210=5\*6\*7。

19. 黑色硬币问题

    答：[链接](https://www.bilibili.com/video/av19584232)

20. 难铺的瓷砖

    答：[链接](https://wenku.baidu.com/view/8605c11452d380eb62946d70.html)

21. 共 10 瓶药丸。（1）其中一瓶每颗超重 10 毫克；（2）其中多瓶每颗超重 10 毫克。用最少称重数目给出错误的瓶号。

    答：（1）从 1 到 10 瓶，每瓶拿出 1、2、3、...、10 颗，超重颗数即为超重瓶号；（2） 从 1 到 10 瓶，每瓶各拿出 1、2、4、...、512 颗，根据超重颗数的二进制表示得到超重瓶号。

22. 捡麦穗问题

    答：[链接](https://www.zhihu.com/question/66465943)

23. 鹰鸽博弈

    答：[链接](https://www.bilibili.com/video/av12414632?from=search&seid=6712463178550853494)

24. Python 相关实现

    答：
    ```python
	import itertools
	import math
	from collections import deque
	
	# =======================
	# 集合运算
	# =======================
	def set_intersection(A, B):
	    return A & B
	
	def set_union(A, B):
	    return A | B
	
	def set_difference(A, B):
	    return A - B
	
	def set_symdiff(A, B):
	    return A ^ B
	
	# =======================
	# 排列与组合
	# =======================
	def permutations(lst):
	    return list(itertools.permutations(lst))
	
	def combinations(lst, k):
	    return list(itertools.combinations(lst, k))
	
	def comb(n, k):
	    return math.comb(n, k)
	
	def perm(n, k):
	    return math.perm(n, k)
	
	# =======================
	# 逻辑运算与真值表
	# =======================
	def truth_table(n):
	    return list(itertools.product([False, True], repeat=n))
	
	# =======================
	# 模运算与数论
	# =======================
	def mod_add(a, b, m):
	    return (a + b) % m
	
	def mod_mul(a, b, m):
	    return (a * b) % m
	
	def mod_pow(a, b, m):
	    return pow(a, b, m)
	
	def egcd(a, b):
	    if a == 0:
	        return (b, 0, 1)
	    g, x, y = egcd(b % a, a)
	    return (g, y - (b // a) * x, x)
	
	def modinv(a, m):
	    g, x, _ = egcd(a, m)
	    if g != 1:
	        raise Exception("No modular inverse")
	    return x % m
	
	# =======================
	# 示例使用
	# =======================
	if __name__ == "__main__":
	    A = {1, 2, 3}
	    B = {2, 3, 4}
	    print("交集:", set_intersection(A,B))
	    print("并集:", set_union(A,B))
	    print("差集 A-B:", set_difference(A,B))
	    print("对称差:", set_symdiff(A,B))
	
	    lst = [1,2,3]
	    print("排列:", permutations(lst))
	    print("组合:", combinations(lst,2))
	    print("C(5,2):", comb(5,2))
	    print("P(5,2):", perm(5,2))
	
	    graph = [[0,1,1,0],
	             [1,0,0,1],
	             [1,0,0,1],
	             [0,1,1,0]]
	    print("BFS from 0:", bfs(graph,0))
	    print("DFS from 0:", dfs(graph,0))
	
	    print("2变量真值表:", truth_table(2))
	
	    print("模运算: (17+13)%5 =", mod_add(17,13,5))
	    print("模运算: (17*13)%5 =", mod_mul(17,13,5))
	    print("模运算: 17^13 % 5 =", mod_pow(17,13,5))
	    print("17 在模 5 下的逆元:", modinv(17,5))
    ```


### Algorithm

1. KSum

    答：[KSum 问题](https://lpq29743.github.io/algorithm/2018/10/29/KSum/)
    
    Two Sum：遍历过程中，使用哈希表记录已访问数字及其索引，查找 target - nums[i] 是否存在，时间复杂度 O(n)。此方法不需要排序。
    
    Three Sum：对数组排序，固定一个数，剩余部分用双指针从两端逼近，跳过重复项，时间复杂度 O(n²)。
    
    Three Sum Closest：类似 3Sum，固定一个数后使用双指针，记录当前和与 target 的最小差值，更新最接近值。
    
    Three Sum Smaller：排序后，固定一个数，对剩余部分用双指针统计满足和小于 target 的组合数量。
    
    Four Sum：排序后，固定两个数，对剩余数组使用双指针。可看作 2Sum 在外面套两层循环，时间复杂度 O(n³)。为避免重复访问，第二个数索引应比第一个数大，左指针也应比第二个索引大。
    
    Four Sum II（给定四个整数数组，统计所有元素和为零的四元组个数）：利用哈希表预存前两数组的和，再在后两数组中查找 target - sum 是否存在，时间复杂度可降为 O(n²)。
    
    General K-Sum：排序 + 递归：固定一个数，将 KSum 降维为 (K-1)Sum，最终降为 2Sum 使用双指针，时间复杂度 O(n^(k-1))。
    
    K-Sum II（输出所有组合）：不要求去重时，可使用回溯（DFS）遍历所有长度为 K 的组合，判断其和是否等于目标，适合小规模数据。
    
    K-Sum with Constraints（带上下界或特定索引）：  
    在递归或双指针基础上加入条件判断或跳过无效值，需特别注意剪枝。
    
    K-Sum Variant in Interview（求组合数 / 有多少组）：  
    可以用记忆化递归或多维动态规划（DP[k][i][j]），适合计数问题而非列出所有组合。

2. 双指针

    答：一、对撞型双指针（Two Pointers Opposite Direction）
    
    - KSum
	- Container With Most Water：双指针从两端向内收缩，面积取决于较短的那条边，移动短板。时间复杂度：O(n)
	- Valid Palindrome：双指针从两端向中间检查字符是否匹配，跳过非字母数字字符。可扩展为“最多删一个字符”场景。
	- 接雨水：对于每个 i，其可接雨水量为`min(leftmax[i], rightmax[i]) - height[i]`。双指针分别从两端向中间移动，维护 `left_max`、`right_max`。每次移动较低的一侧，并计算可接的水量。时间复杂度：O(n)，空间复杂度：O(1)
	
	二、滑动窗口型双指针（Same Direction）
	
	- Longest Substring Without Repeating Characters：右指针滑动加入字符，左指针缩小窗口直到无重复，哈希表记录字符位置。
	- Minimum Window Substring：右指针扩展直到满足条件，左指针缩小窗口以求最小子串，哈希计数维护需求。
	- Longest Subarray with Sum ≤ K（正整数数组的子数组和）：滑动窗口维护当前和，超过 K 就移动左指针。
	- Fruit Into Baskets（最多两个不同元素的最长子数组）：滑动窗口 + 哈希表记录窗口内元素类型，超过两种时左指针前移。
	
	```python
	left = 0
	for right in range(len(arr)):
	    # 扩大窗口（通常加入arr[right]）
    
	    while 窗口不满足条件:
	        # 缩小窗口（通常移动left）
        
	    # 此时窗口满足条件，更新结果（比如最大值、最小长度等）
	```
	
	三、快慢指针（Fast and Slow Pointers）
	
	- Linked List Cycle Detection（环检测）：快慢指针在链表上移动，相遇即有环。环入口可通过重新遍历得到。
	- Middle of Linked List：快指针一次两步，慢指针一步，快指针到末尾时，慢指针即为中点。
	- Remove N-th Node From End of List：快指针先走 n 步，然后快慢指针同时移动，快到末尾时慢指向待删除前一节点。

3. 子数组

    答：最大/最小和/积
    
    - 最大子数组和（Maximum Subarray）：Kadane 算法，维护当前最大和和全局最大和：若当前和为负数，则丢弃，从当前元素重新开始；否则就累加当前元素。
    - 最小子数组和：与最大子数组类似，取 min。
    - 最大子数组乘积（Maximum Product Subarray）：遍历数组时，同时维护当前位置结尾的最大乘积 (`max_prod`) 和最小乘积 (`min_prod`)；遇到负数时交换二者，因为负数可能把最小变最大；每一步更新全局最大值 `res`。状态转移方程为`max_prod[i] = max(nums[i], nums[i] * max_prod[i-1], nums[i] * min_prod[i-1])`，`min_prod[i] = min(nums[i], nums[i] * max_prod[i-1], nums[i] * min_prod[i-1])`，时间复杂度为 O(n)。
    
    ```python
    def maxProduct(nums):
	    max_prod = min_prod = result = nums[0]
	    for num in nums[1:]:
	        # 遇到负数时，最大和最小会交换
	        if num < 0:
	            max_prod, min_prod = min_prod, max_prod
	        
	        max_prod = max(num, num * max_prod)
	        min_prod = min(num, num * min_prod)
	        
	        result = max(result, max_prod)
	    return result
    ```
    
	- 子数组最大和（二维扩展）：枚举上边界 `top` 和下边界 `bottom` 行。对每一对 `(top, bottom)`，把矩阵列压缩成一个一维数组 `sums`，其中 `sums[c]` 表示列 `c` 在 `top..bottom` 行的累加和。在 `sums` 上使用一维最大子数组和（Kadane），得到这一行段组合的最大矩阵和。遍历所有 `(top, bottom)`，更新全局最大值。时间复杂度为 O(rows^2 * cols)。
    
    ```python
    def maxSumSubmatrix(matrix):
	    if not matrix or not matrix[0]:
	        return 0
	    rows, cols = len(matrix), len(matrix[0])
	    max_sum = float('-inf')
	
	    for top in range(rows):
	        # 初始化列累加数组
	        col_sums = [0] * cols
	        for bottom in range(top, rows):
	            # 累加当前行到 col_sums
	            for c in range(cols):
	                col_sums[c] += matrix[bottom][c]
	
	            # 在 col_sums 上运行 Kadane 算法
	            cur_sum = col_sums[0]
	            cur_max = col_sums[0]
	            for i in range(1, cols):
	                cur_sum = max(col_sums[i], cur_sum + col_sums[i])
	                cur_max = max(cur_max, cur_sum)
	
	            max_sum = max(max_sum, cur_max)
	
	    return max_sum
    ```
    
    滑动窗口（适用于全正数或符合单调性问题）
    
    - 长度为 K 的最大平均值子数组：固定长度滑动窗口，维护当前窗口总和。
    - 最多包含 K 个不同元素的最长子数组：变长滑动窗口 + 哈希表统计窗口内字符种类。
    - 最多有两个不同字符的最长子字符串：滑动窗口 + 哈希表记录字符频次。
    
    计数
    
    - 等差子数组个数：枚举子数组判断是否等差，可使用 dp 优化。
    ```python
    def numberOfArithmeticSlices(nums):
	    n = len(nums)
	    if n < 3:
	        return 0
	
	    dp = [0] * n
	    total = 0
	
	    for i in range(2, n):
	        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
	            dp[i] = dp[i-1] + 1
	            total += dp[i]
	        # 否则 dp[i] 保持 0
	
	    return total
    ```
    - 子数组最大值 - 最小值 ≤ K 的个数：使用两个单调队列维护区间最大最小值的 index，维护一个变长滑动窗口。 
    
    ```python
    def count_subarrays(nums, k):
	    n = len(nums)
	    max_q, min_q = deque(), deque()
	    left = 0
	    count = 0
	
	    for right in range(n):
	        # 维护单调递减队列 max_q
	        while max_q and nums[right] > nums[max_q[-1]]:
	            max_q.pop()
	        max_q.append(right)
	
	        # 维护单调递增队列 min_q
	        while min_q and nums[right] < nums[min_q[-1]]:
	            min_q.pop()
	        min_q.append(right)
	
	        # 当最大值 - 最小值 > k 时，移动左边界
	        while nums[max_q[0]] - nums[min_q[0]] > k:
	            left += 1
	            if max_q[0] < left:
	                max_q.popleft()
	            if min_q[0] < left:
	                min_q.popleft()
	
	        # 窗口内每个子数组都是合法的
	        count += right - left + 1
	
	    return count
    ```

4. 区间问题

    答：一、基础区间合并与排序
    
	- 合并区间/最少区间合并次数/合并后区间个数：将区间按起点排序，依次比较是否重叠并合并。
	- 插入区间（Insert Interval）：先添加所有在新区间左边的区间，再合并所有与新区间重叠的区间，最后添加所有在新区间右边的区间。时间复杂度为 O(n)。
	- 区间交集（Interval Intersection）：双指针遍历两个有序区间列表，找重叠区间。
	
	二、覆盖与选择类
	
	- 用最少数量的区间覆盖整段区间：按起点排序，使用贪心策略选择右端最大值。
	- 用最少区间覆盖点集 / 时间段：按右端排序，每次选择能覆盖最多未覆盖点的区间。
	- 判断是否能用区间完全覆盖目标区间：贪心扫描，判断最大可达右端是否覆盖到终点。
    
    三、区间动态操作类
    
    - 区间更新 + 区间查询（如数值加减、最大最小值等）：线段树（将每个长度不为 1 的区间划分成左右两个区间递归求解） / 树状数组 / 差分数组。
    - 区间加法，单点查询：使用前缀和或差分数组优化区间操作。
    - 区间异或 / 取反操作 + 查询：线段树支持 lazy propagation 或位运算。
    
    四、区间重叠与冲突检测类
    
    - 判断是否存在重叠区间：按起点排序，检查当前起点是否小于上一个区间的终点。
    - 会议室问题（最多需要几个会议室）：扫描线（处理区间、时间段、二维平面上的覆盖问题）：把所有开始时间当作 +1 事件（需要新会议室），把所有结束时间当作 -1 事件（释放会议室），然后从小到大扫描时间：每遇到一个开始时间，就 `+1`，每遇到一个结束时间，就 `-1`。在扫描过程中，记录当前会议室数量的最大值，就是需要的会议室数。另一种方法是用最小堆维护当前会议的结束时间。
    - 区间调度问题（最多安排多少个互不重叠的区间）：贪心，按结束时间排序，每次选择最早结束的非重叠区间。
    - 最少区间移除使所有区间不重叠：贪心选择最多不重叠的区间，剩下的就是需要移除的。
    
    五、其他综合变种
    
    - 区间染色、区间合并维护颜色段数：线段树 + 标记合并逻辑。
    - 二维区间合并 / 查询（如矩形区域）：扫描线 + 树状数组/线段树 + 离散化。
    - 区间差值或重叠个数统计：差分数组 + 前缀和 或 扫描线算法。
    - 区间最长连续可用时间段：对不可用区间排序合并，剩余为可用时间段。

5. 前缀和（Prefix Sum）

    答：得到前缀和之后可以知道任何区间的和，所以很适合求解要求区间和为某个数的问题。前缀和是一种通用思想，它也可以是前缀积、前缀异或、前缀最大值。
    
    一维前缀和：时间复杂度为 O(n)
    
	- 区间和快速查询：构建前缀和数组 `prefix[i+1] = sum(nums[0..i])`，任意区间和为 `prefix[r+1] - prefix[l]`。
	- 固定大小窗口的最小/最大和：用前缀和快速计算任意长度为 k 的区间和，再遍历或滑动窗口取最小/最大。
    - 和为 K 的子数组数量（Subarray Sum Equals K）：滑动窗口只适用于所有数为正；否则使用前缀和 + 哈希表（初始值为 {0: 1}），在计算前缀和的时候，同时记录前缀和出现次数，当前 prefix_sum - k 在哈希表里的次数为以当前结点为结尾的和为 k 的子数组个数。
    - 和为 K 的最长子数组：同样使用前缀和 + 哈希表，记录每个前缀和第一次出现的位置。
    - 不超过 K 的最大子数组和：通过维护一个有序前缀和集合，利用二分查找快速定位满足不超过 K 限制的最优子数组前缀差，进而求出最大和。
    - 不超过 K 的最大子数组长度：前缀和 + 有序集合（或单调队列变体），找到最左边的前缀使得子数组和 ≤ K，然后更新长度。
    - 最长平衡子数组（0 和 1 个数相同）：将 0 转为 -1，问题转为和为 0 的最长子数组，使用前缀和+哈希。
    - 判断是否存在和为 k 的连续子数组：计算前缀和并对 k 取模，若同一余数出现两次，则存在满足条件的子数组。
    
    二维前缀和（矩阵）
    
	- 矩阵区域和查询：预处理二维前缀和 `pre[i][j]=pre[i−1][j]+pre[i][j−1]−pre[i−1][j−1]+matrix[i−1][j−1]`，查询时用`sum=pre[row2+1][col2+1]−pre[row1][col2+1]−pre[row2+1][col1]+pre[row1][col1]`快速获取矩形和。`+1` 是因为前缀和数组比原矩阵多一行一列，方便处理边界。
    - 最大子矩阵和：枚举上下边界，将列压缩为一维数组，转化为一维最大子数组和问题 + 前缀和。
    - 统计满足条件的子矩阵个数：将二维压缩为一维后，用前缀和 + 哈希/树状数组处理子数组和满足条件的数量。
    
    前缀和与频率/计数统计
    
	- 区间内某值出现的次数：对每个字符/数值构建出现次数的前缀和数组，查询时用 `cnt[r+1] - cnt[l]` 获取某值出现频率。
    - 最多变 k 次后使所有数相等：排序 + 前缀和 + 双指针，维护区间和与最大值的关系判断是否满足变化次数。
    - 子数组的平均值比较：用前缀和计算子数组和避免重复计算平均值，再转为乘法避免浮点误差。
    
    特殊技巧：差分、异或前缀和
    
	- 差分数组：通过差分数组只更新区间端点，最后用前缀和还原整体变化。
    - 异或前缀和：异或满足结合律，使用 `prefix[i] = A[0]^...^A[i]`，子数组异或为 `prefix[r] ^ prefix[l-1]`。
    - 字符串中子串统计：构建每个字符的出现次数前缀和，查询区间内特定字符频率。
    
    综合技巧与高难度题
    - 子矩阵和为目标的个数：枚举行区间压缩为一维数组，然后用子数组前缀和 + 哈希统计目标值出现次数。
    - 最大连续 1 的数量（允许翻转 k 个 0）：用前缀和记录 0 的累计个数，滑动窗口找到最多翻转 k 次后的最长区间。
    - 滑动窗口平均值/中位数优化：前缀和用于快速计算平均值，配合单调队列或多重集合求中位数。

6. 并查集（Union Find）

    答：解决步骤为：初始化父节点为本身；寻找父节点；合并父节点；判断集合数量。
    - 省份数量：并查集判断连通块数。
    - 岛屿数量：并查集合并相邻陆地格子。
    - 冗余连接：并查集找成环的边。
    - 连通网络的操作次数：统计冗余边与连通块数量。
    - 等式方程的可满足性：合并等式再判断不等式是否冲突。
    - 除法求值：带权并查集维护比值。
    - 账户合并：邮箱为节点并查集合并账户。
    - 交换字符串中的元素：归并交换块后排序。
    - 寻找图中是否存在路径：并查集判断是否连通。
    - 字典序最小等效字符串：并查集维护字典序最小根。
    - 最长连续序列：连续数归为同一集合。
    - 相似字符串组：相似字符串构图归类。
    - 冗余连接 II：有向图中判断成树条件。
    - 由斜杠划分区域：每格四块并查集合并内部与相邻块。
    - 最小时间传递信息：Kruskal 过程判断连通。
    
    简单模版
    ```python
    class UnionFindSimple:
	    def __init__(self, n):
	        self.parent = [i for i in range(n)]
	    
	    def find(self, x):
	        if self.parent[x] != x:
	            self.parent[x] = self.find(self.parent[x])  # 路径压缩
	        return self.parent[x]
	    
	    def union(self, x, y):
	        px, py = self.find(x), self.find(y)
	        if px != py:
	            self.parent[py] = px
	    
	    def connected(self, x, y):
	        return self.find(x) == self.find(y)
	```
	高阶模版（支持动态元素；路径压缩，即在 find 的过程中，把节点直接连到根节点，减少树的高度；按秩合并；打印所有集合）
	```python
	from collections import defaultdict

	class UnionFind:
	    def __init__(self):
	        self.parent = {}
	        self.rank = {}
	    
	    def find(self, x):
	        if x not in self.parent:
	            self.parent[x] = x
	            self.rank[x] = 0
	        # 路径压缩
	        if self.parent[x] != x:
	            self.parent[x] = self.find(self.parent[x])
	        return self.parent[x]
	    
	    def union(self, x, y):
	        px, py = self.find(x), self.find(y)
	        if px == py:
	            return
	        if self.rank[px] < self.rank[py]:
	            self.parent[px] = py
	        elif self.rank[px] > self.rank[py]:
	            self.parent[py] = px
	        else:
	            self.parent[py] = px
	            self.rank[px] += 1
	    
	    def connected(self, x, y):
	        return self.find(x) == self.find(y)
	    
	    def groups(self):
	        """返回所有集合"""
	        g = defaultdict(list)
	        for x in self.parent:
	            g[self.find(x)].append(x)
	        return list(g.values())
	```

7. 原地旋转数组（即右移 m 个元素）

    答：
    - step 1：因为 m 可能大于 n，因此需要对 n 取余，因为每次长度为 n 的旋转数组相当于没有变化。
    - step 2：第一次将整个数组翻转，得到数组的逆序，它已经满足了右移的整体出现在了左边。
    - step 3：第二次就将左边的 m 个元素单独翻转，因为它虽然移到了左边，但是逆序了。
    - step 4：第三次就将右边的 n−m 个元素单独翻转，因此这部分也逆序了。

8. 顺时针旋转矩阵（90 度）

    答：先进行对角线的交换，再翻转。

9. 链表

    答：一、基础操作类
    
    - 创建链表
    
    ```python
    class ListNode:
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	        
	def create_linked_list(arr):
	    dummy = ListNode()  # 虚拟头节点
	    curr = dummy
	    for val in arr:
	        curr.next = ListNode(val)
	        curr = curr.next
	    return dummy.next  # 返回真正的头节点
    ```
    
	- 反转链表（Reverse Linked List）
    
    迭代（iteration）
    
    ```python
    def reverse_list(head):
	    prev = None
	    curr = head
	    while curr:
	        next_temp = curr.next   # 保存下一个节点
	        curr.next = prev        # 反转指针
	        prev = curr             # prev 前进
	        curr = next_temp        # curr 前进
	    return prev  # prev 是新头节点
    ```
    
    递归（recursion）
    
    ```python
    def reverse_list_recursive(head):
	    if not head or not head.next:
	        return head
	    
	    # 假设已经成功反转了 head.next 之后的链表
	    # new_head 会指向反转后的新头节点
	    new_head = reverse_list_recursive(head.next)
	    
	    # 关键操作：
	    # head.next 是反转后链表的尾节点
	    # 把它的 next 指针指回当前节点 head
	    head.next.next = head
	
	    # 把当前节点的 next 断开，否则会形成环
	    head.next = None
	    
	    # new_head 始终指向新的头节点（反转后的链表头）
	    return new_head
    ```
    
    - 合并两个有序链表：双指针逐节点比较；递归也可实现。
    - 给定单向链表的头指针和一个结点指针，定义一个函数在O(1)时间删除该节点：如果待删节点不是尾节点，可以通过将待删节点的下一个节点的值和指针复制过来，然后删除下一个节点来实现 O(1) 删除。如果待删节点是尾节点，且只有头指针，不知道前驱节点，无法在 O(1) 时间内删除该节点。只能遍历找到前驱，时间是 O(n)。
    - 删除链表中的节点（如删除倒数第 n 个节点）：快慢指针找到待删节点的前一个节点。
    - 删除排序链表中的重复元素（只保留一个）：遍历有序链表，当前节点与下一个节点值相同则跳过下一个节点，保留第一个重复节点。
    - 删除排序链表中的所有重复元素（重复节点全部删除）：遍历有序链表，遇到重复节点全部跳过，只保留不重复的节点。
    - 删除无序链表中的重复元素：使用哈希集合记录出现过的值，遍历链表时删除已经出现过的节点。
    - 删除无序链表中所有重复元素（只保留无重复元素的节点）：先遍历统计每个元素出现次数，再遍历链表删除所有出现次数大于1的节点。
    - 删除链表中指定值的所有节点：遍历链表，删除值等于目标值的节点。
    - 删除链表中重复的连续节点（保留第一个节点）：只删除连续重复的后续节点，保留第一个出现节点。
    - 删除环形链表中的重复元素：先检测环，确定环入口，再处理环内重复元素。
    - 递归删除排序链表中的重复元素：使用递归方法实现有序链表中重复元素的删除。
    - 双指针法删除排序链表重复元素：快慢指针遍历，慢指针记录唯一节点，快指针跳过重复节点。
    - 删除链表重复元素后保持链表结构的稳定性：确保删除过程中链表链接正确，避免断链或遗漏节点。
    - 查找链表中点：快慢指针，快走两步、慢走一步。
    
    二、循环与判断类
    
    - 链表是否有环：可以使用快慢指针（Floyd判圈算法），慢指针每次走一步，快指针每次走两步，如果链表有环，那么当快慢指针同时进入环的时候，每次快指针都会在慢指针后追一步，所以如果环长为 r，最多 r - 1次，两者就会相遇。时间复杂度为 O(n)，空间复杂度为 O(1)。
    - 链表环的入口节点：设链表从头到环入口的长度为 `a`；环入口到相遇点的距离为 `b`；环的总长度为 `r`；k 为快指针多走的圈数。在它们相遇时，快指针走的总路程是慢指针的两倍：`快指针走的距离 = 慢指针走的距离 + n 圈环长（n 是正整数） 2(a + b) = a + b + k * r ⇒ a + b = k * r`。所以：`a = k * r - b`。相遇后一指针回头从头出发走 a 步，另一指针从相遇点出发，同样速度，则会弥补 r - b 到入口，并绕 k - 1 圈，因此再次相遇点即入口。
    - 判断两个链表是否相交：尾节点是否相同；或长度对齐后同时遍历。
    - 找到两个链表的交点：双指针分别遍历两链表，尾对接。
    
    三、翻转与重排类
    
    - k 个一组翻转链表：递归或迭代，每次翻转 k 个节点。
    - 按奇偶位置重排链表：奇偶指针分别连接各自序列，最后拼接。
    - 重排链表（L0→Ln→L1→Ln-1→...）：中点+反转后半+合并两个链表。
    
    四、复杂结构类
    
    - 复制带随机指针的链表：三步法（原节点后插拷贝 → 建 random → 拆分）。
    - 多级双向链表扁平化：递归处理 child 指针，调整 prev/next。
    
    五、排序与划分类
    
    - 链表排序（归并排序）：快慢指针找中点 + 递归归并。
    - 分隔链表（小于 x 放前面）：构造两个链表（小于和不小于），最后拼接。
    
    六、数字处理类
	
	- 两个链表表示的数字相加：从低位往高位逐位相加，处理进位。
	- 链表相加 II（高位在前）：用栈反转顺序或递归模拟。
	
	七、其他技巧类
	
	- LRU 缓存机制（链表 + 哈希表）：使用双向链表 + 哈希表快速定位和移动节点。
	- 将二叉搜索树转为排序双向链表：中序遍历 + 链接节点。
	- 设计单链表/双链表数据结构（LeetCode 707）：实现基本的 add/delete/get 接口，注意边界处理。

10. 栈

    答：
    - 有效的括号：利用栈匹配左右括号，检测括号是否成对出现且顺序正确。
    - 最小栈：设计一个支持 push、pop、top 操作，并能在O(1)时间内检索到最小元素的栈。可以使用两个栈实现，一个保存最小值，一个常规操作。
    - 用队列实现栈：两个队列，queue1 保存当前栈元素，queue2 为辅助队列，用于调整顺序。push 直接入 `queue1`，pop / top：把 `queue1` 前面 `n-1` 个元素依次放入 `queue2`，剩下的就是栈顶元素。用一个队列做的时候，push 时，将元素入队，然后将前面 `n-1` 个元素依次出队再入队，这样新加入的元素就“旋转到队头”，模拟栈顶。
    - 逆波兰表达式求值：使用栈计算后缀表达式的值。前缀表达式：运算符在前，从右向左处理，无需括号，如 + 3 * 4 5；中缀表达式：运算符在操作数中间，如 3 + 4 * 5；后缀表达式：运算符在后，从左向右用栈处理，无需括号，如 3 4 5 * +。
    - 基本计算器：用栈处理带括号的表达式计算。
    - 栈排序：只用一个额外栈实现对栈元素排序。
    
    ```python
    stack = []
    
    # 入栈
    stack.append(1)
    stack.append(2)
    
    # 出栈
    top = stack.pop()
    
    # 查看栈顶元素（不弹出）
    top_peek = stack[-1]
    
    # 判断栈是否为空
    is_empty = len(stack) == 0
    ```

11. 单调栈

    答：单调递增（递减）栈内保持从栈底到栈顶递增（递减）；入栈前，弹出不满足单调性的元素；弹栈过程中，当前元素就是被弹元素的下一个更小（更大）元素；结束弹栈后，栈顶元素就是当前元素的上一个更小（更大）元素。
    
    ```python
    def monotonic_stack(nums):
	    stack = []  # 栈里存索引
	    res = [-1] * len(nums)  # 记录每个元素左边最近小于它的元素索引

	    for i, num in enumerate(nums):
	        while stack and nums[stack[-1]] >= num:
	            stack.pop()
	        if stack:
	            res[i] = stack[-1]  # 栈顶就是左边最近更小的元素索引
	        stack.append(i)  # 当前元素入栈
	    return res
    ```
    
    应用例子：
    - 接雨水：单调递减栈。每次遇到比栈顶元素高的柱子，就开始结算面积（此时不会考虑当前柱子），直到栈为空或栈顶柱子高度大于当前柱子高度为止。
    - 柱状图中最大的矩形面积：单调递增栈。每次遇到比栈顶元素矮的柱子，就开始结算面积（此时不会考虑当前柱子），直到栈为空或栈顶柱子高度小于当前柱子高度为止。末尾追加一个 0（哨兵）以触发所有结算。

12. 队列

    答：
    - 用栈实现队列：用两个栈模拟队列的入队和出队操作，stact_in 负责入队，stack_out 负责出队。入队：直接压入 stact_in，出队：如果 stack_out 为空，就把 stack_in 所有元素弹出并压入 stack_out（反转顺序），然后从 stack_out 弹出或读取元素。
    - 循环队列设计：实现固定大小的循环队列，支持入队、出队及满/空状态判断。
    - 最近请求次数（计数器设计）：用队列记录请求时间，统计指定时间窗口内的请求数。
    - 数据流中的中位数：使用两个堆模拟双端队列，支持动态维护中位数。
    - 课程表 BFS 拓扑排序：用队列实现广度优先搜索，检测是否有环。
    - 短est路径（BFS）：在无权图中用队列实现最短路径搜索。
    - 滑动窗口平均值：用队列维护窗口元素，实现动态均值计算。
    - 聊天室限流器：用队列控制单位时间内的请求频率。
    - 猫狗队列：用两个队列分别存储猫和狗，实现按顺序取出最早进入的宠物。
    
    ```python
    from collections import deque
    
    queue = deque()
    
    # 入队
    queue.append('a')
    queue.append('b')
    
    # 出队
    first = queue.popleft()  # 'a'
    
    # 查看队首元素
    peek = queue[0]
    
    # 判断是否为空
    empty = not queue
    ```
    
	用 list 实现出队`first = queue.pop(0)`时间复杂度为 O(n)，但用 deque实现头尾出队时间复杂度都为 O(1)。
	
	优先队列是抽象概念，用 heapq（堆）实现

13. 单调队列

    答：队列内保持单调（递增 / 递减）；入队列前，弹出队尾不满足单调性的元素。
    
    应用例子：
    - 固定区间/滑动窗口的最值：滑动窗口最大值：用双端队列存储数组下标，且保证对应的值在队列中单调递减。遍历过程中，如果队列右侧元素小于当前元素，弹出直到队列为空或者队列右侧元素大于当前元素，紧接着判断左侧第一个元素是否不在窗口内，是则移除，最后如果当前遍历满足窗口大小，则将左侧第一个元素加入结果中。也可用最大堆维护当前窗口最大值（需处理过期元素）。
    - 串口 DP
    - 0-1 BFS。

14. 树

    答：二叉树的构建
    
    节点定义
    
    ```python
    class TreeNode:
	    def __init__(self, val):
	        self.val = val
	        self.left = None
	        self.right = None
    ```
    
    从数组/列表构建二叉树（按层次顺序）：假设输入 `[1,2,3,None,4,5,6]` 表示按层次序排列的节点（`None` 表示空节点）：
    
    ```python
    from collections import deque

	def build_binary_tree(values):
	    if not values:
	        return None
	    root = TreeNode(vaues[0])
	    queue = deque([root])
	    i = 1
	    while queue and i < len(values):
	        node = queue.popleft()
	        if values[i] is not None:
	            node.left = TreeNode(values[i])
	            queue.append(node.left)
	        i += 1
	        if i < len(values) and values[i] is not None:
	            node.right = TreeNode(values[i])
	            queue.append(node.right)
	        i += 1
	    return root
    ```
	
	二叉树的前序、中序、后序遍历：用递归/非递归+栈实现节点访问顺序。对于非递归+栈，在先序中，压入顺序为根节点，右节点，左节点，出栈顺序对应为（根、左、右）。在中序中，进入循环的条件为栈是否为空或根节点是否为NULL，首先将左节点入栈，重复该过程，直到左节点不存在，然后依次出栈，出栈的同时判断当前节点的右节点是否存在，若存在则再次进入循环。在后序遍历中，可以借助前序遍历，前序遍历为根左右，后序遍历为左右根，只需将前序遍历顺序调整为根右左，将最终结果reverse就可以得到后序遍历。
	
    二叉树的层序遍历（BFS）：用队列实现按层访问所有节点。
    
    恢复二叉树
    - 前序 + 中序：前序第一个是根 → 中序划分左右子树 → 递归构建左右子树 → 可生成 Post/Level
    - 后序 + 中序：后序最后一个是根 → 中序划分左右子树 → 递归构建左右子树 → 可生成 Pre/Level
    - 前序 + 后序：不能唯一确定（只有单子树结构会有歧义）
    - 前序 + 层次：层次第一个是根 → 前序可判断左右子树节点顺序 → 递归构建 → 可生成 In/Post
    - 后序 + 层次：层次第一个是根 → 后序可确定左右子树节点 → 递归构建 → 可生成 In/Pre
    - 中序 + 层次：层次第一个是根 → 中序划分左右子树 → 递归构建 → 可生成 Pre/Post
    
    判断二叉树是否是平衡二叉树：递归计算子树高度并判断平衡性。
    
    找到二叉树的最近公共祖先（LCA）：递归遍历左右子树，遇到左右子树都包含目标节点时当前节点就是最近公共祖先，否则返回非空子树的结果。或借助父指针寻找公共祖先。
    
    二叉搜索树（BST）插入和查找：利用BST性质快速定位节点。
    
    树的最大深度和最小深度计算：递归遍历叶节点计算深度。
    
    N叉树的遍历：扩展二叉树遍历思路，适应多子节点结构。
    
    求树中两个节点的距离：利用LCA和深度信息计算距离。
    
    判断一棵树是否是子树：递归比较结构和节点值。
    
    树的序列化与反序列化：将树结构转为字符串及还原树形结构。

15. 堆

    答：堆只保证父子节点的局部有序，不保证整个树有序。
    
    删除节点 O(logn)：先将最后一个元素移到堆顶，再从堆顶开始，将该节点与其左右子节点中较小（大）的一个进行比较交换,直到满足最小（大）堆的性质
    
    用 Python 的 `heapq` 库实现，默认为小顶堆；实现最大堆时可将数值取负。
    
    ```python
    import heapq
    
    heap = []
    heapq.heapify(heap)  # 将列表转为堆结构（可省略，空列表即堆）
    # 往堆中插入一个元素，item 可以是 tuple，heapq 会默认根据 tuple 元素顺序排序
    # 时间复杂度：O(log N)
    heapq.heappush(heap, item) 
    min_item = heapq.heappop(heap) #弹出并返回最小元素，时间复杂度：O(log N)
    min_item = heap[0] # O(1) 时间获取最小元素
    heapq.nsmallest(k, nums) # 获取 nums 最小的 K 个元素
    heapq.nlargest(k, nums) # 获取 nums 最大的 K 个元素
    ```
    
    - 前 K 个高频元素：使用小顶堆（优先队列）维护出现频率最高的 K 个元素。
    - 合并 K 个有序链表：使用小顶堆维护每个链表的头节点，实现逐步合并。
    - 数组中第 K 大的元素：使用大小为 K 的小顶堆，维护当前最大的 K 个元素。
    - 数据流中的中位数：使用最大堆和最小堆分别存储数据流的左右两半，动态维护中位数。
    - 寻找和最小的 K 对数字：在两个有序数组中，使用最小堆生成最小的 K 个数对。
    - 接雨水（二维）：使用最小堆从边界向内扩展，维护当前高度边界。
    - 石头碰撞问题：使用最大堆模拟石头之间的碰撞过程，取出最大两块进行处理。
    - 构建哈夫曼编码树：使用最小堆每次合并频率最小的两个节点，构建最优前缀树。

16. Trie

    答：
    - 实现 Trie（前缀树）：使用字典或数组构建多叉树，支持 `insert`, `search`, `startsWith` 操作。
    
    ```python
    class TrieNode:
	    def __init__(self):
	        self.children = {}      # 存储子节点
	        self.is_end_of_word = False  # 标记单词结束
	
	class Trie:
	    def __init__(self):
	        self.root = TrieNode()
	    
	    def insert(self, word: str) -> None:
	        node = self.root
	        for char in word:
	            if char not in node.children:
	                node.children[char] = TrieNode()
	            node = node.children[char]
	        node.is_end_of_word = True
	    
	    def search(self, word: str) -> bool:
	        node = self.root
	        for char in word:
	            if char not in node.children:
	                return False
	            node = node.children[char]
	        return node.is_end_of_word
	    
	    def startsWith(self, prefix: str) -> bool:
	        node = self.root
	        for char in prefix:
	            if char not in node.children:
	                return False
	            node = node.children[char]
	        return True
    ```
    - 替换词根：将句子中的词替换为其词根；利用 Trie 存储所有词根，遍历句子中每个词，查找最短匹配词根。
    - 单词搜索 II：在二维字符网格中找多个单词；构建 Trie 存储单词表，然后结合 DFS 和 Trie 剪枝遍历。
    - 添加与搜索单词：支持通配符 '.' 的词典查询；在 Trie 上做 DFS，遇到 '.' 时递归尝试所有子节点。
    - 最大异或对：给定整数数组，找最大异或值；用 Trie 按位（高到低）存储每个数字的二进制表示，寻找最大异或路径。
    - 回文对：寻找两个单词拼接后为回文的组合；Trie 结合字符串反转和回文判定处理。
    - 前缀和统计：实现 MapSum 类，支持 `insert(key, val)` 和 `sum(prefix)`；Trie 节点记录每个前缀的值和子树和。
    - 统计单词出现频率（面试类问题）：构造 Trie 并在每个叶节点或终止节点记录出现次数，实现高效统计。
    - 敏感词过滤器：构建 Trie 存储敏感词集合，配合 AC 自动机或双指针，进行文本过滤与替换。
    - DNA 序列查重（变种问题）：构建 Trie 存储基因序列（A/C/G/T），检测是否存在重复序列（如长度为 10 的重复子串）。

17. 完全二叉树是什么？

    答：除了最后一层，所有层都必须被填满，且最后一层节点必须尽量靠左排列（因此二叉平衡树不一定是完全二叉树）。

18. 二叉搜索树是什么？

    答：每个节点的值都满足：左子树 < 根节点 < 右子树。其中序遍历单调递增。

19. 二叉平衡树是什么？

    答：二叉平衡树（Balanced Binary Tree）是对普通二叉搜索树（BST）的优化，目的是解决 BST 在极端情况下可能退化为链表（如连续插入有序数列）的问题，从而保证查找、插入、删除等操作的时间复杂度维持在 O(log n)。二叉平衡树要求任意节点的左右子树高度差控制在一定范围内（如 AVL 树是 ≤1，红黑树是最多两倍）。

20. AVL 树是什么？

    答：AVL 树是一种自平衡的二叉搜索树（Binary Search Tree，BST），它在插入和删除节点时，能自动调整自身结构以保持树的“平衡”，从而保证查找、插入、删除操作的时间复杂度始终为 `O(log n)`。

21. 红黑树是什么？

    答：红黑树性能要好于平衡二叉树。

    红黑树是每个节点都带有颜色属性的二叉查找树，颜色或红色或黑色。在二叉查找树强制一般要求以外，对于任何有效的红黑树我们增加了如下的额外要求:

    性质1. 节点是红色或黑色。

    性质2. 根节点是黑色。

    性质3 每个叶节点（NIL节点，空节点）是黑色的。

    性质4 每个红色节点的两个子节点都是黑色。(从每个叶子到根的所有路径上不能有两个连续的红色节点)

    性质5. 从任一节点到其每个叶子的所有路径都包含相同数目的黑色节点。

22. B/B- 树是什么？

    答：B 树是一种自平衡的多路搜索树，是对二叉搜索树的推广，适合磁盘或大规模数据存储系统中高效读取。
    - 每个节点可以有多个键值（key）和子树指针
    - 所有键值按顺序排列
    - 每个节点的子树数在一个范围内（阶的约束）
    - 所有叶子节点在同一层
    - 查找路径短，I/O 次数少
    
    查找过程类似多路查找：从根节点开始，依次判断键值大小，决定走哪个子树，直到叶子或命中。

23. B+ 树是什么？

    答：B+ 树是在 B 树基础上演化而来的，数据库系统中应用最广泛的索引结构。
    
    - 所有值只存在于叶子节点
    - 内部节点（非叶子）只存键，不存数据
    - 所有叶子节点之间用链表连接，天然支持范围查询
    
	- 更高的扇出（fan-out），因为内部节点更“轻”，树更矮 → 减少磁盘访问
	- 范围查找特别高效，只需找到区间起点，后续通过链表遍历即可

24. 图的表示方法

    答：邻接表：每个节点维护一个列表，存储它所有相邻的节点。邻接矩阵：二维数组。

25. 判断图存在环？

    答：无向图：深度遍历；并查集。
    有向图：深度遍历；广度遍历/拓扑排序：时间复杂度为 O(n＋e)。

26. 最短路径算法及复杂度？

    答：Dijkstra 算法，用于解决边权非负的单源最短路径问题（可多次调用变成多源），是贪心算法，时间复杂度为 O(V^2)。如果是稀疏图，可用堆进行优化，时间复杂度为 O((V + E) lgV)。Dijkstra 算法每次选择当前已知最短路径中最小的节点 u（初始时，源节点到自身距离为 0，到其他任意节点为无穷大），设置为已访问，添加到路径中（可用前驱数组实现）。更新它的邻居的距离（松弛操作，Relaxation），即对当前被选择节点 u 相邻的未访问节点 v，如果经过 u 到 v 的路径比当前已知到 v 的最短路更短：`dist[v] > dist[u] + w`，就更新它，如需保存路径。重复，直到所有节点处理完。
    
    ```python
    import heapq

	def dijkstra(n, edges, start):
	    """
	    n: 节点数
	    edges: 邻接表 {u: [(v, w), ...]}
	    start: 起点
	    """
	    dist = [float('inf')] * n
	    dist[start] = 0
	
	    pq = [(0, start)]  # (当前距离, 节点)
	
	    while pq:
	        d, u = heapq.heappop(pq)
	        # 避免处理过期的队列元素
	        # 当松弛操作发现 `dist[v]` 被更新更小的距离时，会再次把 `(dist[v], v)` 入队。
			# 队列里可能存在旧的、较大的距离值，也就是“过期元素”。
	        if d > dist[u]:
	            continue  # 已经有更短的路径，跳过
	
	        for v, w in edges[u]:
	            if dist[u] + w < dist[v]:
	                dist[v] = dist[u] + w
	                heapq.heappush(pq, (dist[v], v))
	
	    return dist
    ```
    
    0-1 BFS，用于解决边权为 0 或 1（可推广为 0 或任意正整数，不可为两个整数，因为这样子双端队列无法保证单调）的单源最短路径问题（可多次调用变成多源），用邻接矩阵表示图，时间复杂度为 O(V + E)。0-1 BFS 是 Dijkstra 算法的特例，用双端队列 (deque) 替代优先队列：如果边权 = 0，把新节点放到队首。如果边权 = 1，把新节点放到队尾。这样保证 deque 始终按最短路顺序扩展节点。
    
    ```python
    from collections import deque

	def zero_one_bfs(n, edges, start):
	    """
	    n: 节点数
	    edges: 邻接表形式, edges[u] = [(v, w), ...], w ∈ {0, 1}
	    start: 起点
	    """
	    dist = [float('inf')] * n
	    dist[start] = 0
	
	    dq = deque([start])
	
	    while dq:
	        u = dq.popleft()
	        for v, w in edges[u]:
	            if dist[u] + w < dist[v]:
	                dist[v] = dist[u] + w
	                if w == 0:
	                    dq.appendleft(v)  # 权重 0 → 优先
	                else:
	                    dq.append(v)      # 权重 1 → 放队尾
	    return dist
    ```
    
    Floyd 算法，用于解决有负权（无负环）的多源最短路径问题，是动态规划算法，时间复杂度为 O(V^3)。
    
    ```python
    def floyd_warshall(n, edges):
	    INF = float('inf')
	    # 初始化邻接矩阵
	    dist = [[INF] * n for _ in range(n)]
	    for i in range(n):
	        dist[i][i] = 0
	    for u, v, w in edges:
	        dist[u][v] = min(dist[u][v], w)  # 处理重边
	    
	    # 核心 DP 三层循环
	    for k in range(n):
	        for i in range(n):
	            for j in range(n):
	                if dist[i][k] + dist[k][j] < dist[i][j]:
	                    dist[i][j] = dist[i][k] + dist[k][j]
	    
	    return dist
    ```
    
    Bellman-Ford 算法，用于解决有负权的单源最短路径问题（可多次调用变成多源），是动态规划算法，时间复杂度 O(VE)。
    
    ```python
    def bellman_ford(n, edges, src):
	    INF = float('inf')
	    dist = [INF] * n
	    dist[src] = 0
	    
	    # 松弛 V-1 次
	    for _ in range(n - 1):
	        updated = False
	        for u, v, w in edges:
	            if dist[u] + w < dist[v]:
	                dist[v] = dist[u] + w
	                updated = True
	        if not updated:
	            break
	    
	    # 检测负环
	    for u, v, w in edges:
	        if dist[u] + w < dist[v]:
	            raise ValueError("Graph contains negative weight cycle")
	    
	    return dist
    ```
    
    SPFA 算法，用于解决有负权的单源最短路径问题（可多次调用变成多源），是 Bellman-Ford 算法的队列优化版，也是动态规划算法，平均时间复杂度 O(E)，最坏时间复杂度 O(VE)。
    
    ```python
    from collections import deque

	def spfa(n, edges, src):
	    INF = float('inf')
	    dist = [INF] * n
	    in_queue = [False] * n
	    count = [0] * n   # 统计每个点入队次数（用于负环检测）
	    
	    dist[src] = 0
	    q = deque([src])
	    in_queue[src] = True
	    
	    graph = [[] for _ in range(n)]
	    for u, v, w in edges:
	        graph[u].append((v, w))
	    
	    while q:
	        u = q.popleft()
	        in_queue[u] = False
	        for v, w in graph[u]:
	            if dist[u] + w < dist[v]:
	                dist[v] = dist[u] + w
	                if not in_queue[v]:
	                    q.append(v)
	                    in_queue[v] = True
	                    count[v] += 1
	                    if count[v] > n:  # 超过 n 次 → 负环
	                        raise ValueError("Graph contains negative weight cycle")
	    return dist
    ```

27. 无向图最小生成树算法及复杂度？

    答：Prim 算法，是贪心算法，每一步从当前生成树出发，选择权值最小的边，并且这条边连接的是树内节点和树外节点，逐步扩展生成树，直到包含所有节点。时间复杂度 O(V^2)；Kruskal 算法，时间复杂度 O(ElgE)。

28. 农夫过河问题？

    答：[链接](https://www.zhihu.com/question/29968331)

29. KMP 算法

    答：[KMP 算法](https://www.zhihu.com/question/21923021)
    
	1. 构建部分匹配表（next 数组）：next[i] 记录每个位置之前 pattern[0:i] 的前缀后缀最长公共长度。
    
    2. 主串匹配时使用 next 数组跳过重复匹配。
    
    时间复杂度为 O(m + n)

30. 了解 Hamming 距离吗？

    答：两个等长字符串之间的汉明距离是两个字符串对应位置的不同字符的个数。换句话说，它就是将一个字符串变换成另外一个字符串所需要替换的字符个数。

31. 如何求两个数的二进制表示的 Hamming 距离？

    答：先求两个数的异或结果 res，再依次求 res 每一位与 1 与操作的结果，不为 0，则 Hamming 距离加一；每判断完一位，res 右移一位继续判断下一位。

32. 哈希冲突

    答：开放地址法（当当前位置被占用时，寻找下一个可用位置；容易产生聚集现象，负载因子高时效率下降）：线性探测（从当前位置开始，逐个向后找）；二次探测（间隔逐步增大，避免连续冲突）；双重哈希（使用第二个哈希函数计算偏移）。
    
    链地址法：每个哈希桶（地址）用一个链表（或其它结构）存储所有哈希到该地址的元素。需要额外的链表或结构，内存碎片。
    
    再哈希：当负载因子过高时，扩大哈希表，重新计算所有元素的哈希值。扩容代价较高。

33. 排序

    答：冒泡排序：时间复杂度 O(n^2)，稳定
    
    插入排序：时间复杂度 O(n^2)，稳定
    
    选择排序：时间复杂度 O(n^2)，不稳定
    
    归并排序：时间复杂度 O(nlogn)，稳定。可用来解决计算逆序对数目的问题。
    
    ```python
    def merge_sort(arr):
	    if len(arr) <= 1:
	        return arr
	    
	    mid = len(arr) // 2
	    left = merge_sort(arr[:mid])
	    right = merge_sort(arr[mid:])
	    
	    return merge(left, right)
	
	def merge(left, right):
	    result = []
	    i = j = 0
	    while i < len(left) and j < len(right):
	        if left[i] <= right[j]:
	            result.append(left[i])
	            i += 1
	        else:
	            result.append(right[j])
	            j += 1
	    # 添加剩余元素
	    result.extend(left[i:])
	    result.extend(right[j:])
	    return result
	
	# 示例
	arr = [5, 2, 9, 1, 5, 6]
	sorted_arr = merge_sort(arr)
	print(sorted_arr)
    ```
    
    基数排序：按位排序，从低位到高位依次进行分配和收集；稳定
    
    快速排序：时间复杂度 O(nlogn)，空间复杂度最好情况是 O(logn)，最坏情况可能达到 O(n),但平均情况是 O(logn)，不稳定
    
    ```python
	def quick_sort(arr, low, high):
	    if low < high:
	        pivot_index = partition(arr, low, high)
	        quick_sort(arr, low, pivot_index - 1)
	        quick_sort(arr, pivot_index + 1, high)
	
	def partition(arr, low, high):
	    pivot = arr[low]  # 选择第一个元素作为基准
	    i = low + 1
	    j = high
	
	    while True:
	        while i <= j and arr[i] <= pivot:
	            i += 1
	        while i <= j and arr[j] >= pivot:
	            j -= 1
	        if i <= j:
	            arr[i], arr[j] = arr[j], arr[i]
	        else:
	            break
	
	    arr[low], arr[j] = arr[j], arr[low]  # 把基准放到正确位置
	    return j
	```
    
    希尔排序：插入排序的改进版本，其核心思想是先将整个待排序序列分割成若干个子序列分别进行直接插入排序，待整个序列基本有序时，再对全体记录进行一次直接插入排序。不稳定
    
    堆排序：堆排序包含两个主要步骤：建堆和排序。建堆过程：从最后一个非叶子节点开始，自上而下进行堆化。排序过程：每次将堆顶元素与末尾元素交换，并重新对剩余元素进行堆化。需要执行 n-1 次，每次堆化的时间复杂度为 O(logn)，因此排序阶段总的时间复杂度为 O(nlogn)。空间复杂度 O(1)，可原地排序，不稳定
    
    桶排序：将元素分布到若干桶中分别排序，最后再合并。时间复杂度 O(n)。可用来计算一个未排序数组中排序后相邻元素的最大差值。

34. 原地排序与非原地排序？

    答：原地排序就是指在排序过程中不申请多余的存储空间，只利用原来存储待排数据的存储空间进行比较和交换的数据排序。非原地排序就是要利用额外的数组。

35. BFS vs DFS

    答：BFS（广度优先搜索）通常用队列解决，适用于求最短路径、层级遍历和状态步数问题，因为它逐层扩展，能保证最早到达目标。
    
    BFS 模版
	```python
	from collections import deque
	
	def bfs(graph, start):
	    visited = set()
	    queue = deque([start])
    
	    while queue:
	        node = queue.popleft()
	        if node in visited:
	            continue
	        visited.add(node)
	        
	        # 处理当前节点 node
	        print(node)
	        
	        for neighbor in graph[node]:
	            if neighbor not in visited:
	                queue.append(neighbor)
	```
	
	应用场景
    
    - 最短路径（无权图）：BFS 可以找到起点到各节点的最短路径。
    ```python
    def bfs_all_distances(graph, start):
	    visited = set()
	    dist = {start: 0}
	    queue = deque([start])
	    visited.add(start)
	
	    while queue:
	        node = queue.popleft()
	        for neighbor in graph.get(node, []):
	            if neighbor not in visited:
	                visited.add(neighbor)
	                dist[neighbor] = dist[node] + 1
	                queue.append(neighbor)
	    return dist
    ```
    - 拓扑排序（课程表问题）：用 BFS 统计入度为 0 的节点，逐步处理图中的节点，检测环。
    - 层次遍历二叉树：使用队列按层访问节点，可用于打印或计算每层节点值。
    - 连通分量/岛屿问题：用 BFS 找到所有相连节点或岛屿面积。
    - 滑动窗口/最小步数问题：如最少操作次数到达目标状态。
    
    优化技巧：
    - 双端队列：可在两端高效插入/删除，提高 BFS 扩展性能。
    - 记录层数/距离：队列元素可以存 `(node, step)`，方便统计步数。
    - 避免重复访问：用 `visited` 或修改原数组标记节点。
    
    DFS（深度优先搜索）通常用栈或递归解决，更适合遍历所有路径、回溯剪枝、拓扑排序和连通块问题，因其先深入到底便于穷举和递归处理。简言之，找最短用 BFS，找所有/存在性用 DFS。
    
	DFS 模版，递归版
	```python
	def dfs_recursive(graph, node, visited=None):
	    if visited is None:
	        visited = set()
	    
	    visited.add(node)
	    # 处理当前节点 node
	    print(node)
	    
	    for neighbor in graph[node]:
	        if neighbor not in visited:
	            dfs_recursive(graph, neighbor, visited)
	```
	
	DFS 模版，非递归版
	```python
	def dfs_iterative(graph, start):
	    visited = set()
	    stack = [start]
	    
	    while stack:
	        node = stack.pop()
	        if node in visited:
	            continue
	        visited.add(node)
	        
	        # 处理当前节点 node
	        print(node)
	        
	        for neighbor in reversed(graph[node]):  # reversed 保持一致遍历顺序
	            if neighbor not in visited:
	                stack.append(neighbor)
	```
	
	应用场景
	- 连通分量/岛屿问题：找图中所有相连节点或岛屿面积。
	- 拓扑排序（课程表 DFS）：用 DFS 检测环并记录访问顺序。
	- 路径搜索：找到从起点到终点的所有路径或最优路径。
	- 二叉树/多叉树遍历：前序、中序、后序遍历都属于 DFS 的特殊情况。
	- 组合/排列问题：如全排列、子集问题，用 DFS 回溯生成所有解。
	
	优化技巧
	- 剪枝：在递归或栈操作前判断条件，减少不必要的搜索。
	- 标记已访问：用 `visited` 集合或修改原数组，防止重复访问。
	- 递归 + 路径记录：便于回溯时知道当前路径或状态。
	- 如果搜索空间为树状结构，则可考虑回溯。

36. 数组最大最小值最优算法？

    答：同时找到最大值和最小值的话，一般方法（遍历两遍）是 O(2n)，改进方法是每次读两个数，这两个数比一次，然后大的和当前最大值比，小的和当前最小值比。这样每两个数比了三次，故复杂度是 O(3/2n)，不过两种方法都是 O(n)。

37. 无序整数数组中找第 k 大的数？

    答：方法一：最小堆：建一个大小为 k 的最小堆。遍历数组，将元素加入堆中，如果堆大小超过 k，就弹出堆顶（最小元素）。最终堆顶就是第 k 大的数。时间复杂度：O(nlogk)，空间复杂度：O(k)
    
    方法二：快速选择（Quickselect），类似快速排序的分区思想（partition）；每次将数组划分为两个区间，选择一边递归；平均时间复杂度 O(n)，最坏 O(n²)。
    
    方法三：内置排序

38. 从一个几乎排序好的数组中找出第 k 小的元素，时间复杂度尽量低。  

    答：利用“插入排序”特性，数组接近有序，插入排序接近线性。也可以用快速选择算法，平均 O(n)。

39. 在 n 个数中取最大的 k 个数，时间复杂度是？

    答：nlogk。堆的大小为 k，总共要调整 n 次。

40. 旋转数组的最小值

    答：二分法
    ```python
    def find_min(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            # 最小值在 mid 右边
            left = mid + 1
        else:
            # 最小值在 mid 或左边
            right = mid
    return nums[left]
    ```

41. 有 10 个排好序的数据库，那么我要找整个的中位数，怎么找？

    答：最简单的思路是合并数据库，然后再定位长度，时间复杂度为 O(n)，空间复杂度是 O(n)；但实际上只需要借鉴这个合并的过程，当合并到中位数的时候输出中位数即可，时间复杂度为 O(n)，空间复杂度是 O(1)。这思路十分简单，但并不是最佳算法，有序数组让我们想到的会是二分查找，因此我们可以利用二分查找（binary search）来使复杂度降至 O(logn)，具体可参考：
    
    https://stackoverflow.com/questions/6182488/median-of-5-sorted-arrays

42. 埃拉托色尼筛法（Sieve of Eratosthenes）

    答：
    ```python
    def sieve(n):
	    """
	    n: 最大整数
	    返回: is_prime 列表, is_prime[i]=True 表示 i 是素数
	    """
	    is_prime = [True] * (n + 1)
	    is_prime[0] = is_prime[1] = False
	    for i in range(2, int(n**0.5)+1):
	        if is_prime[i]:
	            for j in range(i*i, n+1, i):
	                is_prime[j] = False
	    return is_prime
	```
	第二层遍历 $$n/2 + n/3 + n/5 + n/7$$ 根据调和级数性质，约为 $$ln ln n$$，所以时间复杂度为 O(nloglogn)。

43. 海量数据处理

    答：[海量数据处理](https://lpq29743.github.io/algorithm/2017/02/20/MassiveData/)

44. 汉诺塔

    答：假设移动 n 个圆盘需要 f(n) 次移动
    
    首先考虑一个圆盘，只需一步就可以了 f(1) = 1 …… ①
    
    现在考虑 n 个圆盘，假设开始圆盘在 A 柱，可以先把 A 柱的上面 n - 1个圆盘移到 B，再将 A 剩下的一个移到 C，最后将 B 的 n - 1 个移到 C。总共需要 f(n) = 2f(n-  1) + 1 …… ②
    
    ```python
    def hanoi(n, source, auxiliary, target):
	    """
	    打印将 n 个盘子从 source 移动到 target 的步骤。
	    source: 起始柱子
	    auxiliary: 辅助柱子
	    target: 目标柱子
	    """
	    if n == 1:
	        print(f"Move disk 1 from {source} to {target}")
	    else:
	        hanoi(n - 1, source, target, auxiliary)
	        print(f"Move disk {n} from {source} to {target}")
	        hanoi(n - 1, auxiliary, source, target)
	        
	    # 示例：移动 3 个盘子从 A 到 C，B 为辅助柱子
	    hanoi(3, 'A', 'B', 'C')
    ```
    
    根据 ①② 两式，可求出 f(n) = 2^n - 1 所以 O(n) = 2^n

45. 尾递归（Tail Call）有什么危害，如何避免？

    答：栈溢出（Stack Overflow）。尾递归事实上和循环是等价的。

46. 动态规划

    答：动态规划的时间复杂度和空间复杂度不强相关（如完全背包）。
    
    线性 DP
    - 爬楼梯/斐波那契数列（每步可以走 1 或 2 级台阶，求总方法数）：`dp[i] = dp[i-1] + dp[i-2]`，初始值 `dp[0]=1, dp[1]=1`
    - 零钱兑换（用最少的硬币凑出金额）：`dp[i] = min(dp[i - coin] + 1)`，遍历所有 coin。注意金额 i - coin >= 0
    - 最大子数组和（连续子数组和最大）：Kadane - `dp[i] = max(dp[i-1] + nums[i], nums[i])`
    - 打家劫舍（相邻房子不能同时偷）：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
    
    背包 DP
    - [01 背包问题](https://lpq29743.github.io/algorithm/2017/08/21/Pack1/)（每个物品最多选一次，求最大价值）：`dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])`（放物品 i 和不放物品 i 两种选择的最优选），两个 for 循环可以换顺序。可压缩到一维`dp[w] = max(dp[w], dp[w−weight[i]] + value[i])`（把`dp[i-1]`那一层拷贝到`dp[i]`上，二维`dp[i][w]`和`dp[i-1][w]`有区分度，而一维没有），但需倒序遍历 w（`dp[w−weight[i]]`是`dp[i-1][w−weight[i]]`），因为正序遍历（`dp[w−weight[i]]`是`dp[i][w−weight[i]]`）会出现一个物品被选多次的现象，两个 for 循环不可以换顺序，如果换了问题等价于背包里只能放一个物品（因为当遍历`w_i`时，小于`w_i`的`d[w_j]`都为 0，因此相当于选择价值最大的物品）。
    - [完全背包问题](https://lpq29743.github.io/algorithm/2017/08/22/Pack2/)（每个物品可无限次）：`dp[w] = max(dp[w], dp[w - weight[i]] + value[i])`（一维）
    - [物品冲突问题](https://lpq29743.github.io/algorithm/2017/08/25/Pack3/)
    - 分割等和子集：转化为 0-1 背包求 `dp[sum/2]` 是否为 `True`
    - 子集和问题（subset sum）：`dp[i][s] = dp[i-1][s] or dp[i-1][s - nums[i]]`
    
    序列类 DP
    - 最长上升子序列（LIS）：`dp[i] = max(dp[j] + 1 if nums[i] > nums[j])`，需要遍历 0-n 和 0 到 i，因此时间复杂度为 O(n^2)。另一种解法是贪心 + 二分：用一个数组记录上升子序列末尾最小值，遍历时对每个元素用二分找到合适位置替换，最终数组长度即为最长上升子序列长度，时间复杂度为 O(nlogn)。
    
    ```python
    import bisect
    
    i = bisect.bisect_left(sub, x)
    if i == len(sub):
        sub.append(x)
    else:
        sub[i] = x
    ```
    
	- 两个子序列的最长公共子序列（LCS）：`dp[i][j] = dp[i-1][j-1]+1 if match else max(dp[i-1][j], dp[i][j-1])`，可压缩成一维数组`dp[j] = dp[j-1]+1 if match else max(dp[j], dp[j-1])`。
    - 跳跃游戏（能否到达终点）：从后向前判断每个点能否跳到最后一个点
    - 买卖股票的最佳时机：状态转移 + 决策（持有/卖出）
    
    区间/字符串 DP
    - 区间 DP（如戳气球）：枚举区间分割点 `k`，`dp[i][j] = max(dp[i][k] + dp[k][j] + ...)`
    - 回文子串个数：中心扩展或 `dp[i][j] = s[i]==s[j] && dp[i+1][j-1]`
    - 编辑距离（字符串最少操作次数）：`当 word1[i] == word2[j]，dp[i][j] = dp[i-1][j-1]；当 word1[i] != word2[j]，dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1`。初始化：`dp = [[0]*(n+1) for _ in range(m+1)]`及`for i in range(m+1): dp[i][0]=i; for j in range(n+1): dp[0][j]=j`，最后返回`dp[-1][-1]`
    - 正则表达式匹配：状态表示文本和模式的位置，处理 `*` 和 `.`
    - 最长回文子序列：`dp[i][j] = dp[i+1][j-1] + 2 if s[i]==s[j] else max(dp[i+1][j], dp[i][j-1])`
    - 打字机（最小按键次数）：状态包含剪切板，粘贴次数等
    
    矩阵 DP
    - 矩阵路径和最小值：`dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]`

47. 贪心

    答：贪心技巧：排序 + 遍历；优先队列 / 堆维护可选元素；单调栈 / 队列。关键判断是局部选择能否保证全局最优。
    
    区间/选择类
    - 活动选择问题：选出最多不重叠活动。按结束时间排序，每次选择最早结束且不冲突的活动
    - 跳跃游戏 II（最少跳跃次数）：最少步数跳到末尾。每次选择能到达最远位置的跳跃
    - 区间合并/覆盖问题：每次选择覆盖范围最大的区间
    
    排序/数组类
    - 分发饼干/资源分配问题：尽量用小的资源满足小需求。先排序需求和资源，再从小到大匹配。
    - 股票/利润类问题（无限次买卖）：每次遇到上涨就卖出。局部上涨累积收益即可。
    
    堆/优先队列类
    - 最小开销问题（如合并石子/棒材）：每次选最小/最大元素合并。局部最小成本保证全局最优。
    - 调度/任务分配：优先选择可行范围内最优（最早结束、最小代价）任务。
    
    字符串/栈类
    - 重排字符避免重复：每次选择当前出现次数最多且不冲突的字符
    - 单调栈/队列问题：如接雨水、移掉 k 位数字得到最小数、柱状图最大矩形。保持局部单调性，保证全局最优。

48. 不用库函数求一个数的立方根？

    答：[链接](https://blog.csdn.net/sjpz0124/article/details/47726275)

49. 二进制中 1 的个数？

    答：把一个整数减去 1，再和原整数做与运算，会把该整数最右边的 1 变成 0。那么一个整数的二进制表示中有多少个 1，就可以进行多少次这样的操作。具体解题思路可参见《剑指 Offer》。

50. 位运算加法

    答：
    ```python
    function add(a, b):
	    while b ≠ 0:
	        sum = a XOR b            # 不带进位的和
	        carry = (a AND b) << 1   # 进位
	        a = sum
	        b = carry
	    return a
    ```

51. 数值的整数次方？

    答：[链接](https://zhuanlan.zhihu.com/p/38715645)

52. 有两个未知整数，你可以不断询问某个数与这两个数的大小关系（每次询问一个数），该如何查找这两个数？

    答：[链接](https://www.zhihu.com/question/310970538)

53. 一群木板，一开始有一条线把它们固定在一条水平线上，现在抽掉这条线，有的木板往下掉落，有的木板位置上升，问怎么移动才能使移动距离最小，让它们继续在一条水平线上？

    答：中位数（Median）。

54. 给定两个数，求他们无限次相加中第 k 小的数？

    答：[链接](https://www.zhihu.com/question/41809896)

55. 什么是水塘抽样（Reservoir sampling）？

    答：一种在数据量未知或数据流形式下，以等概率从 n 个元素中采样 k 个的算法，适用于内存受限的场景。

56. 如何从数据流中以等概率选取一个元素（k=1）？

    答：初始化：`result = None`，遍历第 i 个元素时，以 `1/i` 的概率替换 result，所有元素最终被选中的概率都是 `1/n`。

57. 如何扩展到选取 k 个元素？

    答：初始化：前 k 个元素入 reservoir，对第 i (>k) 个元素：以 k/i 的概率随机替换 reservoir 中的一个元素。

58. 链表中如何随机返回一个节点？（单次遍历，O(1) 空间）

    答：遍历链表，对第 i 个节点，以 `1/i` 的概率更新当前候选节点，最终返回的节点是等概率选中的。

### Computer Science

1. 为什么要用时间复杂度来描述算法，而不是运行时间？

   答：操作系统调度，所以运行时间不一定相同。

2. 死锁的条件

   答：a.互斥：某种资源一次只允许一个进程访问，即该资源一旦分配给某个进程，其他进程就不能再访问，直到该进程访问结束。
   b.占有且等待：一个进程本身占有资源（一种或多种），同时还有资源未得到满足，正在等待其他进程释放该资源

   c.不可抢占：别人已经占有了某项资源，你不能因为自己也需要该资源，就去把别人的资源抢过来。

   d.循环等待：存在一个进程链，使得每个进程都占有下一个进程所需的至少一种资源。

3. 死锁避免方法

   答：产生死锁需要四个条件，那么，只要这四个条件中至少有一个条件得不到满足，就不可能发生死锁了。

4. 银行家算法

   答：[链接](https://www.cnblogs.com/chuxiuhong/p/6103928.html)

5. 集线器和交换机有什么区别？

   答：集线器在物理层，所有端口是一个冲突域，交换机在数据链路层，每个端口是一个冲突域。

6. 三次握手是怎样的？

   答：第一次握手：建立连接时，客户端发送 syn 包（syn = j）到服务器，并进入 SYN_SENT 状态，等服务器确认；

   第二次握手：服务器收到 syn 包，必须确认客户的 SYN（ack = j + 1），同时自己也发送一个 SYN 包（syn = k），即 SYN + ACK 包，此时服务器进入 SYN_RECV 状态；

   第三次握手：客户端收到服务器的 SYN + ACK 包，向服务器发送确认包 ACK (ack = k + 1），此包发送完毕，客户端和服务器进入 ESTABLISHED（TCP 连接成功）状态，完成三次握手。

7. 四次挥手是怎样的？

   答：TCP 客户端发送一个 FIN，用来关闭客户到服务器的数据传送。

   服务器收到这个 FIN，它发回一个 ACK，确认序号为收到的序号加 1。和 SYN 一样，一个FIN 将占用一个序号。

   服务器关闭客户端的连接，发送一个 FIN 给客户端。

   客户端发回 ACK 报文确认，并将确认序号设置为收到序号加 1。
   
8. 8 位二进制整型补码表示取值范围是？

    答：-2^32 到 2^32 - 1。补码中正负 0 统一用正 0 表示，所以负数多出一个数。

9. count(1)、count(\*) 和 count(列名) 的区别？

    答：[链接](https://blog.csdn.net/qq_15037231/article/details/80495882)

10. 数据库的三级模式是什么？

    答：数据库领域公认的标准结构是三级模式结构，包括外模式、概念模式、内模式。
    
    用户级对应外模式。它是某个或某几个用户所看到的数据库的数据视图，是与某一应用有关的数据的逻辑表示。
    
    概念级对应概念模式（又称逻辑模式），反映了数据库的整体观。它是由数据库设计者综合所有用户的数据，按照统一的观点构造的全局逻辑结构，是对数据库中全部数据的逻辑结构和特征的总体描述，是所有用户的公共数据视图。
    
    物理级对应内模式（又称存储模式），反映了数据库的存储观。它是数据库中全体数据的内部表示或底层描述，它描述了数据在存储介质上的存储方式和物理结构，对应着实际存储在外存储介质上的数据库。

11. 数据库三范式？

    答：[链接](https://www.zhihu.com/question/24696366)

### Programming

#### C/C++

1. 字节对齐？

    答：先看以下程序：

   ```c
   struct A{
       int    a;
       char   b;
       short  c;
   };
   struct B{
       char   b;
       int    a;
       short  c;
   };
   ```

   已知 32 位机器上各数据类型的长度为：char 为 1 字节、short 为 2 字节、int 为 4 字节、long 为 4 字节、float 为 4 字节、double 为 8 字节。那么上面两个结构体大小如何呢？

   结果是：sizeof(strcut A) 值为 8；sizeof(struct B) 的值却是 12。 

   结构体 A 中包含一个 4 字节的 int 数据，一个 1 字节 char 数据和一个 2 字节 short 数据；B 也一样。按理说 A 和 B大小应该都是 7 字节。之所以出现上述结果，就是因为编译器要对数据成员在空间上进行对齐。

   a.数据类型自身的对齐值：char 型数据自身对齐值为 1 字节，short 型数据为 2 字节，int / float 型为 4 字节，double 型为 8 字节。

   b.结构体或类的自身对齐值：其成员中自身对齐值最大的那个值。

   c.指定对齐值：#pragma pack (value) 时的指定对齐值 value。

   d.数据成员、结构体和类的有效对齐值：自身对齐值和指定对齐值中较小者，即有效对齐值 = min{自身对齐值，当前指定的 pack 值}。

2. 面对对象的三大特性是什么？

    答：封装、继承和多态。

3. 定义一个空类时，C++ 到底会默默为我们编写哪些函数？

    答：4 个函数：一个 default 构造函数、一个析构函数、一个 copy 构造函数、一个等号 “=” 重构函数。

4. C++ 的构造函数可以是虚的嘛？析构函数呢？

    答：构造函数不能为虚函数，而析构函数可以且常常是虚函数。

5. C++ 中的抽象类是什么？

    答：定义了纯虚函数的类称为抽象类，抽象类不能被实例化，抽象方法只能声明于抽象类中，且不包含任何实现，派生类必须覆盖它们。抽象类也可派生自一个抽象类。

6. C++ 中的虚函数是什么？

    答：C++ 的多态性是通过虚函数实现的，虚函数必须存在于继承环境下。虚函数是重载的一种表现形式。只有类的普通成员函数可以定义为虚函数，全局函数及静态成员函数（类拥有）不能声明为虚函数。

7. C++ 中的纯虚函数是什么？

    答：virtual 函数类型 函数名（形参表列）= 0。抽象类中定义的，为了派生类中的使用而声明定义的。

8. C++ 中的抽象类和接口有什么区别？

    答：一个类一次可以实现若干个接口，但是只能扩展一个父类。

9. C++ 中的 public、private 和 protected 是什么？

    答：a. 公有（public）成员可以在类外访问。
    
    b. 私有（private）成员只能被该类的成员函数访问。
    
    c. 保护（protected）成员只能被该类的成员函数或派生类的成员函数访问。

#### Python

1. Python 中的 \*args 和 \*\*kwargs 是什么意思？

    答：\*args 表示可变参数（variadic arguments），它允许你传入 0 个或任意个无名参数，这些参数在函数调用时自动组装为一个 tuple； \*\*kwargs 表示关键字参数（keyword arguments），它允许你传入 0 个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个 dict。同时使用 \*args 和 \*\*kwargs 的时候，必须保证 \*args 在 \*\*kwargs 之前。 \*args 是变量 list，\*\*kwargs 是指针 list。

2. \_\_XXX\_\_ 是什么？

    答：\_\_XXX\_\_ 在 Python 中通常指双下划线开头和结尾的特殊方法或属性，也叫魔法方法/dunder method（double underscore）。常见的包括可通过内置函数调用的 \_\_str\_\_（输出用户可读），\_\_repr\_\_（输出开发者可读）和 \_\_len\_\_（长度调用），可通过运算符调用的 \_\_add\_\_，\_\_sub\_\_，\_\_getitem\_\_ 和 \_\_setitem\_\_。此类函数也可直接调用。

3. \_\_new\_\_ 了解吗？

    答：\_\_init\_\_ 作用是类实例进行初始化，第一个参数为 self，代表对象本身，可以没有返回值。\_\_new\_\_ 则是返回一个新的类的实例，第一个参数是 cls 代表该类本身，必须有返回值。\_\_new\_\_ 先执行，然后再 \_\_init\_\_，实际上，只要 \_\_new\_\_ 返回的是类本身的实例，它会自动调用 \_\_init\_\_ 进行初始化。但是有例外，如果 \_\_new\_\_ 返回的是其他类的实例，则它不会调用当前类的 \_\_init\_\_。

4. \_\_name\_\_ 的值怎么确定？

    答：[链接](https://blog.csdn.net/iamoldpan/article/details/78077983)

5. map() 返回的对象是什么？

    答：map() 函数返回的是一个迭代器，并对返回结果使用了 yield，所以其返回结果只能使用一次，这样做在于节省内存。

6. \[\[1, 2\], \[3, 4\], \[5, 6\]\] 一行代码展开该列表，得出 \[1, 2, 3, 4, 5, 6\]

    答：`[j for i in [[1,2],[3,4],[5,6]] for j in i]`

7. Python 中怎么转换 ascii 码和字符？

    答：`chr(65)`和`ord('A')`。

8. `len('中文')`和`len('中文'.encode('utf-8'))`的输出分别是多少？

    答：2 和 6，Python 默认是  Unicode 编码，转换成 UTF-8 编码，中文通常占用三个字符，英文一个。

9. 一行代码将字符串 "->" 插入到 "abcdefg" 中每个字符的中间

    答：`"->".join("abcdef")`

10. 如何判断一个字符串是否是另一个字符串的子串

    答：`in`；`s.find()`可以放回 index，如果返回 -1 则说明查询不到；`s.index()`，类似`s.find()`，找不到会抛出异常。

11. 字符串常用函数

    答：大小写转换：lower(), upper(), capitalize(), title(), swapcase()
    
    查找与判断：find(), index(), count(), startswith(), endswith(), in
    
    空白与格式清理：strip(), lstrip(), rstrip(), replace()
    
    分割与拼接：split(), join()
    
    字符判断：isdigit(), isalpha(), islower(), isupper()

12. list

    答：
    ```python
    # 增
    lst = [1, 2, 3]
    lst.append(4)        # 加到末尾
    lst.insert(1, 10)    # 插入到索引 1 位置
    lst.extend([5, 6])   # 扩展多个元素
    # 删
    lst.remove(2)        # 删除指定元素（找不到会报错）
    del lst[0]           # 按索引删除
    lst.pop()            # 删除并返回最后一个元素
    lst.clear()          # 清空列表
    # 改
    lst[0] = 100         # 修改指定索引值
    # 查
    print(lst[0])        # 按索引访问
    print(len(lst))      # 长度
    print(10 in lst)     # 是否包含
    for item in lst:     # 遍历
	    print(item)
	```

13. Python 的 list 在 append 的时候会发生什么？

    答：如果当前容量没满，直接把 `x` 放到下一个空位，已用空间 `+1`，这是 O(1) 操作；如果容量满了，Python 会扩容（reallocate）一块更大的内存区域（通常为当前容量的 1.1 倍），把旧数据拷贝过去，再把 `x` 插进去，这是 O(n) 的操作（因为要复制已有元素）。因此`[None] * n`比 `append`更省时。

14. set

    答：
    ```python
    # 增
    s = {1, 2, 3}
    s.add(4)
    s.update([5, 6])     # 添加多个元素
    # 删
    s.remove(2)          # 删除元素（不存在会报错）
    s.discard(10)        # 安全删除（不存在不报错）
    s.pop()              # 随机删除一个元素
    s.clear()            # 清空集合
    # 改：集合不能直接修改元素，但可以删除后添加新值。
    # 查
    print(3 in s)
    for item in s:
	    print(item)
	```

15. dict

    答：
    ```python
    # 增
    d = {'a': 1, 'b': 2}
    d['c'] = 3           # 新增键值对
    d.update({'d': 4})   # 批量添加/更新
    # 删
    del d['a']           # 删除键
    d.pop('b')           # 删除键并返回值
    d.clear()            # 清空字典
    # 改
    d['c'] = 10          # 修改键对应的值
    # 查
    print(d['c'])        # 获取值（键必须存在）
    print(d.get('x'))    # 安全获取，不存在返回 None
    print('c' in d)      # 是否包含键
    for k, v in d.items():
	    print(k, v)
	```

16. tuple

    答：
    ```python
    # 增/删/改：都不支持
    t = (1, 2, 3)
    t = t + (4,)         # 创建新元组，实现“添加”效果
    # 查
    print(t[0])          # 访问索引
    print(len(t))        # 长度
    print(3 in t)        # 是否包含
    for item in t:
	    print(item)
	```

17. Python 的可变对象和不可变对象

    答：可变对象：可以在原地修改其内容，不改变对象的 id（地址）。不可变对象：一旦创建，内容就不能被改变。修改操作会创建新的对象。
    
    可变对象有 list, dict, set；不可变对象有 str, tuple, int, float, bool。
    
    不可变对象例子：字符串
    ```python
    s = "hello"
    print(id(s))  # 比如 123456
    s = s + " world"
    print(id(s))  # 改变了（新对象）
    ```
    
    可变对象例子：列表
    ```python
    lst = [1, 2, 3]
    print(id(lst))  # 比如 789123
    lst.append(4)
    print(id(lst))  # 没变（原地修改）
    ```
    
    可变对象在函数中修改，会影响原对象，在函数内进行重新绑定，并不会影响外部变量；不可变对象修改，会创建新对象。
    
    作为字典键或集合元素，必须是不可变对象（如字符串、整数、元组）。

18. 自定义排序

    答：sorted_data = sorted(data, key=lambda x: (x[0], x[1]))
    
    data.sort(key=lambda x: (x[0], x[1]))
    
    可以加 reverse=True 或 reverse=False
    
    或
    ```python
    from functools import cmp_to_key
    
    def my_cmp(x, y):
	    # 先比较a字段升序
	    if x[0] < y[0]:
	        return -1
	    elif x[0] > y[0]:
	        return 1
	    else:
	        # a相等时，b字段降序
	        if x[1] > y[1]:
	            return -1
	        elif x[1] < y[1]:
	            return 1
	        else:
	            return 0
	
	lst = [(2, 3), (1, 4), (2, 1), (1, 5)]
	lst.sort(key=cmp_to_key(my_cmp))
	print(lst)
	```

19. Python 的匿名函数是什么

    答：就是 lambda 函数，一般用来实现简单的功能，如加法 `(lambda x, y: x + y)(3, 4)`。

20. @staticmethod 和 @classmethod 的作用与区别

    答：[链接](https://blog.csdn.net/qq_15037231/article/details/77943109)

21. Python 的 getter 和 setter 是什么？

    答：@property 和 @name.setter。

22. Python 赋值、浅拷贝和深拷贝

    答：[链接](https://www.cnblogs.com/wilber2013/p/4645353.html)

23. 装饰器的意义和使用？

    答：用来实现代码复用，增强代码可读性。

    ```python
    def deco(func):
        def warpper(*args, **kwargs):
            print('start')
            func(*args, **kwargs)
            print('end')
        return warpper
          
    @deco
    def myfunc(parameter):
        print("run with %s" % parameter)
        myfunc("something")
    ```

24. 解释一下多态？

    答：多态的好处就是，当我们需要传入`Dog`、`Cat`、`Tortoise`……时，我们只需要接收`Animal`类型就可以了，因为`Dog`、`Cat`、`Tortoise`都是`Animal`类型，然后，按照`Animal`类型进行操作即可。由于`Animal`类型有`run()`方法，因此，传入的任意类型，只要是`Animal`类或子类，就会自动调用实际类型的`run()`方法，这就是多态的意思：
    
    对于一个变量，我们只需要知道它是`Animal`类型，无需确切地知道它的子类型，就可以放心地调用`run()`方法，而具体调用的`run()`方法是作用在`Animal`、`Dog`、`Cat`还是`Tortoise`对象上，由运行时该对象的确切类型决定，这就是多态真正的威力：调用方只管调用，不管细节，而当我们新增一种`Animal`的子类时，只要确保`run()`方法编写正确，不用管原来的代码是如何调用的。

25. 鸭子类型是什么？

    答：对于 Python 这样的动态语言，则不一定需要传入`Animal`类型。我们只需要保证传入的对象有一个`run()`方法就可以了：这就是动态语言的“鸭子类型”，它并不要求严格的继承体系，一个对象只要“看起来像鸭子，走起路来像鸭子”，那它就可以被看做是鸭子。

26. Counter 与运算和或运算

    答：与和或操作分别返回两个 Counter 各元素的最小值和最大值。得到的 Counter 对象将删除小于 1 的元素。

27. Python 多线程？

    答：对于 CPU 密集型，由于全局解释器锁（GIL）的存在，GIL 保证在任何时刻只有一个线程执行 Python 字节码，所以线程是线性执行的；对于 I/O 密集型，由于线程在等待 I/O 操作时会释放全局解释器锁，所以这时多线程有体现作用。

#### Linux

1. 程序如何后台运行？

    答：nohup + &!；tmux。

2. 查看当前训练进程占用的内存和 CPU 资源

    答：top, htop, ps aux 找到进程

3. 查看当前训练进程占用的 GPU 资源

    答：nvidia-smi

### Machine Learning

#### Basic

1. 解释一下奥卡姆剃刀原理（Occam's Razor）？

    答：如果两个模型的预测能力差不多，就选简单的。原因有二，简单模型往往解释性更好；复杂的模型更有可能过拟合。

2. 解释一下没有免费的午餐原理（No free lunch）？

    答：A 算法在某些数据集或任务上表现比 B 算法好，则一定存在一些数据集或任务，B 算法表现比 A 算法好。这告诉我们具体问题具体分析。

3. L1 和 L2 的联系和区别？

   答：都是正则化系数，L1（Lasso）得到稀疏的权值，用于特征选择，L2（Ridge）得到平滑的权值。

4. 为什么 L1 更容易得到稀疏解？

    答：因为损失函数的等高线易与 L1 正则的坐标轴上的点相切。第一象限与正方形相切的圆的中心只能是在第一象限正方形边的垂直区域内，而第一象限的圆都可以与中心点的圆相切。

    The question can be answered in a geometric view. L1 loss can be represented as a diamond, while L2 loss can be represented as a circle. Therefore, the loss is likely to **intersect** with the diamond on only one point, which is in axises.
    
    根据自身函数或梯度，对于 L1，梯度要么是 1，要么是 -1（梯度的正负代表方向，数值代表更新幅度），而对于 L2，权重更大，对应的梯度更大。因此对于 L2，规范数值较大的项导致 L2 范数的减少比对数值较小项的减少要大得多，而对于 L1，规范任意大小的参数带来的收益是相同的。进一步说，当使用 L2，任何东西都不太可能被设置为零，而对于 L1，则有概率导致稀疏性。

5. Huber loss

    答：L2 损失（平方）对异常值敏感（容易被 outlier 拉大），L1 损失（绝对值）更鲁棒（robust），对异常值不敏感。
    
    Huber loss 小误差用 L2，大误差用 L1，兼顾精度和鲁棒性。

6. 结构风险（Empirical Risk）和经验风险（Structural Risk）分别是什么？

    答：经验风险就是使所有训练样本的损失平均值最小，结构风险其实就是加多一个正则项。

7. 什么是生成模型，什么是判别模型？

    答：生成模型：学习得到联合概率分布 P(x, y)，即特征 x 和标记 y 共同出现的概率，然后求条件概率分布。能够学习到数据生成的机制。

    判别模型：学习得到条件概率分布 P(y \| x)，即在特征 x 出现的情况下标记 y 出现的概率。

    数据要求：生成模型需要的数据量比较大，能够较好地估计概率密度；而判别模型对数据样本量的要求没有那么多。

8. 什么是参数模型，什么是非参数模型？

    答：参数模型假设总体服从某分布，该分布由一些参数确定。因此参数数量固定，训练、推理速度快，占用内存小，但可能会欠拟合，例子有线性回归、逻辑回归、神经网络、SVM（核线性除外）。
    
    非参数模型对于总体分布不做任何假设，只有在给定一些样本的条件下，能够依据非参数统计的方法进行推断。因此参数数量随数据量增长，拟合度高，但训练、推理速度慢，占用内存大，例子有 KNN、决策树、核密度估计、高斯过程。

9. 拉普拉斯平滑是什么？

    答：为了解决零概率问题：如果某个量 x，在训练集中没有出现过，会导致概率结果是 0。这是不合理的，不能因为一个事件没有观察到就武断的认为该事件的概率是 0。拉普拉斯平滑即加法平滑。假定训练样本很大时，每个分量 x 的计数加 1 造成的估计概率变化可以忽略不计，但可以方便有效的避免零概率问题。

10. bias 和 variance 的含义是什么？

    答：bias 要求让 Error(train) 尽可能小，variance 要求让 Error(train) 尽可能等于 Error(test)。bias 体现了模型的精确度，variance 体现了模型的泛化能力。两个值都是希望越小越好。

11. 宏平均 F1 和 微平均 F1 的区别？

     答：宏平均 F1 对每个类求 F1 之后取平均，微平均 F1 针对整体求 F1。

12. ROC（Receiver Operating Characteristic）和 AUC（Area under Curve）

     答：ROC 依据阈值，画出曲线，横轴为假正类率 (false positive rate，FPR)，纵轴为真正类率 （true positive rate，TPR）。
     
     AUC 为 ROC 曲线下的面积，介于 0.1 和 1 之间。AUC 作为数值可以直观地评价分类器的好坏，即排序能力（将正样本排在负样本前面的能力），值越大越好。

13. 手撕 AUC

     答：AUC 可以理解为：随机选择一个正样本和一个负样本，正样本预测分数大于负样本的概率。
     
     ```python
     def auc_score(y_true, y_score):
	    # 获取正负样本索引
	    pos_scores = [s for t, s in zip(y_true, y_score) if t == 1]
	    neg_scores = [s for t, s in zip(y_true, y_score) if t == 0]
	    
	    pos_count = len(pos_scores)
	    neg_count = len(neg_scores)
	    
	    if pos_count == 0 or neg_count == 0:
	        return 0.5  # 极端情况
	    
	    count = 0
	    for pos in pos_scores:
	        for neg in neg_scores:
	            if pos > neg:
	                count += 1
	            elif pos == neg:
	                count += 0.5
	    
	    return count / (pos_count * neg_count)
     ```

#### Regression

1. 线性回归的基本假设是？

    答：
    - 线性性（Linearity）：可通过可视化观察；不成立会导致欠拟合；可通过对数据做非线性转换来满足条件。
    - 独立性（Independence of Errors）：可用 DW 检测；不满足会导致自相关性（如时间序列数据）；此种场景需要换成自回归之类的模型解决。
    - 同方差性（Constant Variance）：可视化；异方差性虽然不会影响参数估计，即估计量依旧是无偏的（Unbiased），但会导致标准误差（Standard Error）被低估或高估，从而 t 检验、F 检验不再可靠，置信区间（Confidence interval）不正确，显著性检验可能出现假阳性或假阴性；可更换因变量或使用稳健回归/加权回归。
    - 正态性（Normality of Errors）：Q-Q图/KS 非参数检验；置信区间不稳定；可寻找遗漏的自变量，检验或剔除异常值，对自变量（independent variable）/因变量（dependent variable）进行线性转换。
    - 无多重共线性（No Multicollinearity）：可计算 Pearson score，若 score 过高，则证明相关，VIF 是可替代方法；不成立会导致多重共线性；可剔除、合并相关变量来解决。。

2. 线性回归中特征不小心重复会有影响吗？

    答：会，使用最小二乘法会导致矩阵不可逆，使用梯度下降法会导致训练不稳定，因为两个特征分散了一个权重，提高一个降低另一个对损失的影响是没有的。

3. 最小二乘法（Least Squares）为什么可以解决线性回归问题？

    答：残差（residuals）满足均值为 0 的同方差正态分布时，用最大似然估计法可以证明最小二乘法是合理的。

4. 描述一下最小二乘法的几何意义？

    答：最小二乘法中的几何意义是高维空间中的一个向量在低维子空间的投影。$$WX$$ 实际上是当前样本形成的线性组合空间 $$S$$，最小化的过程是找到一个合适的 $$W$$，使得不在 $$S$$ 上的 $$Y$$ 到 $$S$$ 的投影距离最小。

5. 正规方程（normal equation）是什么？它有什么作用？

    答：$$(X^TX)^{-1}X^Ty$$。可以一次运算得出结果，但特征数目过多时不适用。

6. 用最小二乘法实现线性回归

    答：
    ```python
    import numpy as np
	# 1. 生成数据（y = 2x + 3 + noise）
	np.random.seed(42)
	X = np.random.rand(100, 1)  # 100 个样本，1 个特征
	y = 2 * X[:, 0] + 3 + np.random.randn(100) * 0.1  # 添加噪声
	
	# 2. 添加偏置项 x0 = 1（扩展 X 为 [x, 1]）
	# 这是为了把截距 b 统一进矩阵运算里，便于计算、推导、编程
	X_b = np.c_[X, np.ones((X.shape[0], 1))]  # shape: [100, 2]
	
	# 3. 正规方程解：theta = (X^T X)^(-1) X^T y
	theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
	
	print("权重和偏置（w, b）：", theta)
    ```

7. 用梯度下降实现线性回归

    答：
    ```python
    # 初始化
	w = np.random.randn()
	b = np.random.randn()
	lr = 0.1
	
	for epoch in range(1000):
	    y_pred = w * X[:, 0] + b
	    error = y_pred - y
	    loss = (error ** 2).mean()
	
	    # 手动计算梯度
	    grad_w = 2 * (error * X[:, 0]).mean()
	    grad_b = 2 * error.mean()
	
	    # 更新参数
	    w -= lr * grad_w
	    b -= lr * grad_b
	
	    if epoch % 100 == 0:
	        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
    ```

8. 最小二乘法和梯度下降的区别

    答：最小二乘法可以一次性得到最优解，但当 feature 和 sample 数目过多时，对内存要求过高，且当矩阵不可逆时，无法求解。

#### Classification

1. 逻辑斯特回归（Logistic Regression，LR）为什么不能用均方误差（MSE，Mean Squared Error）计算损失函数？

    答：MSE 对于 Sigmoid 来说是非凸的，而且也会有梯度消失问题，因为导数中有 p(1-p) 部分，当预测值靠近 0 或 1 时，梯度过小，收敛速度很慢。但如果是 soft label 就可以考虑。

2. LR 为什么用交叉熵计算损失函数（二分类）？

    答：假设有训练样本 $$(x_i, y_i)$$，其中 $$y_i$$ 是标签，取值为 0 或 1。
    
    逻辑回归模型对正类的预测概率为：
    
    $$\hat{y}_i = 1 / (1 + exp(-w·x_i))$$
    
    负类概率为：
    
    $$1 - \hat{y}_i$$
    
    极大似然估计的目标是最大化所有样本的联合概率：
    
    $$L(w) = ∏_{i=1}^n P(y_i | x_i; w)$$
    
    因为 $$y_i$$ 只有 0 或 1，服从伯努利分布（二分类），可以写成：
    
    $$L(w) = ∏_{i=1}^n (\hat{y}_i)^{y_i} * (1 - \hat{y}_i)^{1 - y_i}$$
    
    取对数似然：
    
    $$log L(w) = ∑_{i=1}^n [y_i * log(\hat{y}_i) + (1 - y_i) * log(1 - \hat{y}_i)]$$
    
    最大化对数似然等价于最小化负对数似然，即交叉熵损失：
    
    $$Loss = - ∑_{i=1}^n [y_i * log(\hat{y}_i) + (1 - y_i) * log(1 - \hat{y}_i)]$$
    
    因此，逻辑回归的极大似然估计目标函数就是交叉熵损失函数。

3. LR 为什么用交叉熵计算损失函数（多分类）？

    答：假设有样本 $$(x_i, y_i)$$，标签 $$y_i$$ 属于类别集合 {1, 2, ..., K}。
    
    逻辑回归的多分类模型预测第 k 类的概率为：
    
    $$p_{i,k} = exp(w_k · x_i) / ∑_{j=1}^K exp(w_j · x_i)$$
    
    对所有样本的联合概率：
    
    $$L(W) = ∏_{i=1}^n P(y_i | x_i; W) = ∏_{i=1}^n p_{i, y_i}$$
    
    取对数似然，并因为数据服从多项式分布（多分类）：
    
    $$log L(W) = ∑_{i=1}^n log p_{i, y_i} = ∑_{i=1}^n log [ exp(w_{y_i} · x_i) / ∑_{j=1}^K exp(w_j · x_i) ]$$
    
    展开：
    
    $$log L(W) = ∑_{i=1}^n [ w_{y_i} · x_i - log ∑_{j=1}^K exp(w_j · x_i) ]$$
    
    最大化对数似然等价于最小化负对数似然，即交叉熵损失：
    
    $$Loss = - ∑_{i=1}^n log p_{i, y_i} = - ∑_{i=1}^n [ w_{y_i} · x_i - log ∑_{j=1}^K exp(w_j · x_i) ]$$
    
    换一种形式，用 one-hot 标签 $$y_i^k$$：
    
    $$Loss = - ∑_{i=1}^n ∑_{k=1}^K y_i^k * log p_{i,k}$$

4. LR 梯度

    答：Logistic Regression 的预测概率定义为：
    
    $$\hat{y}_i = 1 / (1 + exp(-z_i))，其中 z_i = w · x_i + b$$
    
    损失函数（单样本）是交叉熵：
    
    $$L_i = - [ y_i * log(\hat{y}_i) + (1 - y_i) * log(1 - \hat{y}_i) ]$$
    
    求损失对参数 w 的梯度：
    
    $$∂L_i/∂w = (\hat{y}_i - y_i) * x_i$$
    
    求损失对偏置 b 的梯度：
    
    $$∂L_i/∂b = \hat{y}_i - y_i$$
    
    推导要点：
    
    1. 先对损失函数关于预测概率 $$\hat{y}_i$$ 求导：
    
    $$∂L_i/∂\hat{y}_i = - y_i / \hat{y}_i + (1 - y_i) / (1 - \hat{y}_i)$$
    
	2. 再对预测概率 $$\hat{y}_i$$ 关于 $$z_i$$ 求导：
    
	$$∂\hat{y}_i/∂z_i = \hat{y}_i * (1 - \hat{y}_i)$$
	
	3. 最后对 $$z_i$$ 关于参数 w 求导：
    
	$$∂z_i/∂w = x_i$$
	
	4. 通过链式法则合并以上导数，简化得到：
    
	$$∂L_i/∂w = (\hat{y}_i - y_i) * x_i$$
	
	同理偏置 b 的梯度为：
	
	$$∂L_i/∂b = \hat{y}_i - y_i$$

5. LR 为什么用 sigmoid 函数？

    答：[链接](http://sofasofa.io/forum_main_post.php?postid=1004244)

6. 从贝叶斯的角度解释 LR

    答：LR 可以从贝叶斯角度理解为最大后验估计（MAP）。贝叶斯公式如下：
    后验概率 $$P(θ \mid D)$$ ∝ 似然 $$P(D \mid θ)$$ × 先验 $$P(θ)$$。
    
    其中 $$D$$ 是观测数据；$$θ$$ 是模型参数；$$P(D \mid θ)$$ 表示在参数 $$θ$$ 下，数据 $$D$$ 出现的可能性，即似然函数；$$P(θ)$$ 是对参数 θ 的先验知识；$$P(θ \mid D)$$ 是给定数据 $$D$$ 后，参数 $$θ$$ 的后验分布。
    
    训练 Logistic Regression 的目标就是找到使后验概率 $$P(θ \mid D)$$ 最大的参数 $$θ$$，这就是最大后验估计（MAP）：$$θ_{MAP} = argmax P(D \mid θ) × P(θ)$$
    
    不同情况下有不同的结果：
    
    - 如果不加先验（即 $$P(θ)$$ 为常数），那么最大后验就变成了最大似然估计（MLE）。这对应标准的逻辑回归训练目标，即最小化交叉熵损失。
    - 如果 $$P(θ)$$ 是高斯分布（$$exp(−θ² / 2σ²)$$），那么 MAP 对应在最大似然的基础上加入 L2 正则项（Ridge Regression 形式），防止过拟合。
    - 如果 $$P(θ)$$ 是拉普拉斯分布（$$exp(−\|θ\| / b)$$），则 MAP 对应于最大似然 + L1 正则项（Lasso Regression 形式），这会导致部分参数为 0，实现特征选择的效果。

7. 交叉熵的理论下限

    答：H(y)。对于 hard label，H(x) = 0。

8. LR 为什么要对特征进行离散化（discretization）？

    答：模型是使用离散特征还是连续特征，其实是一个“海量离散特征+简单模型” 同 “少量连续特征+复杂模型”的权衡。既可以离散化用线性模型，也可以用连续特征加深度学习。
    
    a. 离散特征的增加和减少都很容易，易于模型的快速迭代； 
    
    b. 稀疏向量内积乘法运算速度快； 
    
    c. 离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄 >30 是 1，否则 0。如果特征没有离散化，一个异常数据“年龄 300 岁”会给模型造成很大的干扰； 
    
    d. 变量离散化为相当于为模型引入了非线性，能够提升模型表达能力，加大拟合； 
    
    e. 特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20 - 30 作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问； 
    
    f. 特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。

9. 给一个有 m 个样本，n 维特征的数据集，如果用 LR 算法，那么梯度是几维？

    答：n 维。

10. 如何用机器学习算法计算特征重要性？

    答：LR 后看系数。

11. 特征选择

    答：剔除方差比较小的特征，其区分度较低，信息熵很大；单变量特征选择（卡方检验）；递归式特征消除：逐步移除掉最不重要的特征；L1；决策树；依序选择：前向（从少到多）或后向（从多到少）。

12. 单层感知机为什么不能解决异或问题？

    答：因为异或操作需要两条线来划分边界，而单层感知机可以理解为一个线性分类器，只能解决与、或、非问题。

13. 如何对单层感知机进行改进，使其能够解决异或问题？

    答：多层感知机，或在进入激活函数前加一个多项式模块，从而添加非线性成分。

14. 怎么判断分类器是线性分类器还是非线性分类器？

    答：根据决策面是否是线性的。

15. KNN 的训练损失是多少？

    答：KNN 实际上不算训练，损失为 0。

16. KNN 算法的 k 值应该如何选择？

    答：k 值太小，模型复杂度较高，容易过拟合；k 值太大，模型复杂度不够，较远的点也可能影响分类结果，分类模糊，导致分类结果不理想。当 k 取训练集大小时，分类结果都为训练集中最多的类。k 值一般选取较小的值，且要低于训练样本数的平方根，可以使用交叉验证法选取。

17. KNN 怎么更快地找到最近邻点？

    答：KD 树和 Ball 树，两者都是用树结构把点集递归划分，使得查询时可以剪枝，减少距离计算。KD 树根据样本构建，但训练样例远大于特征维度时才适用，适用于低维数据。Ball 树适用于高维数据。

18. KNN 算法可以根据距离加权吗？

    答：可以用反函数或高斯函数进行距离加权，前者为近邻样本赋予较大权重，稍远的会衰减地很快，因此对噪声数据比较敏感，后者能解决这个问题，但比较复杂。

19. KNN 核心代码

    答：
    ```python
	def k_nearest_neighbors(X, y, test_sample, k):
	    distances = np.linalg.norm(X - test_sample, axis=1)
	    nearest_indices = np.argsort(distances)[:k]
	    nearest_labels = y[nearest_indices]
	    return int(np.round(np.mean(nearest_labels)))
    ```

20. 常见的距离度量方法有哪些？

    答：$$L_p$$ 距离 / Minkowski 距离 / 闵式距离是最常规的距离度量方式，其公式为 $$(\|x-y\|^p)^{1/p}$$。当 $$p = 1$$ 时为曼哈顿距离，$$p = 2$$ 时为欧式距离，$$p$$ 为无穷大时为各个坐标距离的最大值，即切比雪夫距离。

21. 衡量相似度的方法？

    答：欧式距离，Jaccard 相似度（两集合交集大小 / 并集大小），余弦相似度，皮尔逊相关系数（数值归一化后计算余弦相似度），汉明距离。

22. 什么是支持向量机？

    答：支持向量机就是构造一个的超平面，使得距离超平面最近的那些点，即支持向量与超平面之间的 margin 最大，从而将两个集合分开。

23. LR 和 SVM 的联系与区别？

    答：[链接](https://www.cnblogs.com/zhizhan/p/5038747.html)

    联系：都可以处理分类问题（一般为线性二分类）；都可以增加不同正则化项。两种算法性能接近；两个方法都增加对分类影响较大的数据点的权重，SVM 是只考虑 support vectors，LR 是通过非线性映射，减小了离分类平面较远的点的权重。

    区别：LR 是参数模型，SVM 是非参数模型；从目标函数来看，LR 采用的是 log loss，SVM 采用的是 hinge loss。

24. 当数据线性可分、接近线性可分以及线性不可分时，分别使用什么 SVM？

    答：硬间隔最大化、软间隔最大化以及核技巧。

25. SVM 为什么采用间隔最大化？

    答：当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。感知机利用误分类最小策略，求得分离超平面，不过此时的解有无穷多个。线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。另一方面，此时的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强。

26. 手推 SVM

    答：
    
    [SVM](https://lpq29743.github.io/artificialintelligence/2018/09/12/SVM/)
    
    超平面：$$y=w^TX+b$$；

    样本点 $$P(x_i, y_i)$$ 到超平面的几何距离：$$\frac{\|w^Tx_i+b\|}{\|w\|}$$；

    样本点 $$P(x_i, y_i)$$ 到超平面的几何间隔：$$y_i\frac{w^Tx_i+b}{\|w\|} \geq 0$$；

    SVM 解决问题：$$\max_{w,b}{\min_{x_i}y_i\frac{w^Tx_i+b}{\|w\|}}$$；

    由于 $$w, b$$ 可缩放，所以令最近点满足 $$y_i(w^Tx_i+b)=1$$，问题转换为 $$\max_{w}{\frac{1}{\|w\|}} \ \ s.t. \ \ y_i(w^Tx_i + b) \geq 1$$；

    定义拉格朗日函数：$$L(w, b, \alpha) = \frac{1}{2} \|w\|^2 + \sum_{i = 1}^m{\alpha_i(1 - y_i(w^Tx_i + b))}$$；

    转换为拉格朗日的极小极大问题：$$\min_{w, b}\max_{\alpha}L(w, b, \alpha)$$，即先求拉格朗日的上界，再最小化上界；

    可进一步转换为极大极小问题，即对偶问题：$$\max_{\alpha}\min_{w, b}L(w, b, \alpha)$$；

    先求极小问题，对 $$w, b$$ 求导，令导数为 0，求解 $$w = \sum_{i=1}^m{\alpha_iy_ix_i}, 0=\sum_{i=1}^m{\alpha_iy_i}$$；

    代回拉格朗日函数，得对偶问题：$$\max_\alpha-{\frac{1}{2}\sum_{i=1}^m{\sum_{j=1}^m{\alpha_i\alpha_jy_iy_jx_i^Tx_j}} + \sum_{i=1}^m\alpha_i} \ \ s.t. \ \ \sum_{i=1}^m{\alpha_iy_i=0,\alpha_i \geq 0}$$；

    求解问题，解出 $$\alpha$$，从而解得 $$w, b$$，$$\alpha_i > 0$$ 对应的样本即为支持向量。

27. 为什么要将求解 SVM 的原始问题转换为其对偶问题？

    答：一是对偶问题往往更易求解。二是可以自然引入核函数，进而推广到非线性分类问题。

28. 为什么 SVM 对缺失数据敏感？

    答：缺失数据是指缺失某些特征数据，向量数据不完整。SVM 没有处理缺失值的策略。而 SVM 希望样本在特征空间中线性可分，所以特征空间的好坏对 SVM 的性能很重要。缺失特征数据将影响训练结果的好坏。

29. 什么是几何间隔，什么是函数间隔？

    答：几何间隔 $$y_i\frac{w^Tx_i+b}{\|w\|}$$，函数间隔 $$y_i(w^Tx_i+b)$$。函数间隔可以无限大，几何间隔不可以。

30. 支持向量机的训练在本质上是在最优化哪个值？

    答：w。w 得到 b 自然可以得到。

31. 如何用支持向量机实现深度学习？

    答：可以用支持向量机作为网络的最后一层，进行分类。

32. 给一组数据，问决策树、LR、NB 以及 SVM 等算法学出来是什么样子的？

    答：[链接](https://www.zhihu.com/question/26726794)

33. 什么是基于核的机器学习算法？

    答：判别式模型需要把正负样本区分开，那势必会遇到区分不开的情形，这时要用到核函数，所以可认为判别式模型都要用核函数的。

34. SVM 有哪些核函数？

    答：线性核和高斯核，即线性核与 RBF（径向基）核。 线性核：主要用于线性可分，参数少，速度快，对于一般数据，分类效果已经很理想了。 RBF 核：主要用于线性不可分，参数多，分类结果非常依赖于参数。 如果 Feature 数量跟样本数量差不多，应选用线性核的 SVM。 如果 Feature 数量比较小，样本数量一般，选用高斯核的 SVM。其他的核函数包括幂指数核、拉普拉斯核以及 Sigmoid 核等等。

35. 高斯核为什么有效？

    答：[链接](https://stats.stackexchange.com/questions/131138/what-makes-the-gaussian-kernel-so-magical-for-pca-and-also-in-general)

36. 支持向量机可以用来做回归吗？

    答：支持向量机分类是使两类的点在各自的支持向量外，而支持向量机回归是把所有的点都看成一类，并要求在支持向量内。

37. SVM 和 softmax 的区别？

    答：SVM 具有附加稳定性，当样例满足边界条件时，该样例不会影响损失函数；而 softmax 将考虑所有的样例。

38. 感知机和 SVM 有什么区别？

    答：[链接](http://sofasofa.io/forum_main_post.php?postid=1003714)

39. One-class SVM？

    答：用于异常检测问题：只用正常样本训练，学习出一个边界包围正常数据，边界外的点被判为异常。

40. 朴素贝叶斯为何如此朴素？

    答：对条件概率分布作了条件独立性（conditional independence）的假设。

41. 朴素贝叶斯中特征不小心重复会有影响吗？

    答：会，破坏了原本的独立性假设。

42. 用 numpy 实现 cross entropy loss（softmax）

    答：
    ```python
    import numpy as np
    
	def softmax(logits):
	    """
	    计算 softmax 概率，确保数值稳定。
	    logits: 形状为 (N, C)，N 是样本数，C 是类别数
	    """
	    shifted = logits - np.max(logits, axis=1, keepdims=True)
	    exps = np.exp(shifted)
	    return exps / np.sum(exps, axis=1, keepdims=True)
	
	def cross_entropy_loss(probs, labels):
	    """
	    计算平均交叉熵损失。
	    probs: softmax 后的概率，形状为 (N, C)
	    labels: 每个样本的真实类别索引，形状为 (N,)
	    """
	    N = probs.shape[0]
	    # 取出每个样本对应真实类别的概率，防止 log(0) 加个小常数
	    log_likelihood = -np.log(probs[np.arange(N), labels] + 1e-15)
	    return np.sum(log_likelihood) / N
	
	# 示例数据
	logits = np.array([
	    [2.0, 1.0, 0.1],
	    [0.5, 2.5, 0.3],
	    [1.2, 0.7, 3.0]
	])
	labels = np.array([0, 1, 2])  # 每个样本的真实标签
	
	# 计算 softmax 概率
	probs = softmax(logits)
	
	# 计算交叉熵损失
	loss = cross_entropy_loss(probs, labels)
    ```

43. 用 numpy 实现 cross entropy loss（log-softmax）

    答：用 log-softmax 数值更稳定，减 max 是防止指数函数输出过大，用减法算 log 是防止 log(0)。
    ```python
    import numpy as np
    
	def log_softmax(logits):
	    """
	    计算 log-softmax，数值稳定版本。
	    logits: 形状为 (N, C)
	    """
	    shifted = logits - np.max(logits, axis=1, keepdims=True)
	    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
	    return shifted - log_sum_exp  # log_softmax 输出
	
	def cross_entropy_loss_from_logits(logits, labels):
	    """
	    直接从 logits 计算交叉熵损失（使用 log-softmax）。
	    logits: shape (N, C)
	    labels: shape (N,)
	    """
	    N = logits.shape[0]
	    log_probs = log_softmax(logits)
	    log_likelihood = -log_probs[np.arange(N), labels]
	    return np.sum(log_likelihood) / N
	
	# 示例数据
	logits = np.array([
	    [2.0, 1.0, 0.1],
	    [0.5, 2.5, 0.3],
	    [1.2, 0.7, 3.0]
	])
	labels = np.array([0, 1, 2])  # 每个样本的真实标签
	
	# 计算 log-softmax 并交叉熵损失
	loss = cross_entropy_loss_from_logits(logits, labels)
    ```

44. You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?

    答：If you have worked on enough data sets, you should deduce that cancer detection results in imbalanced data. In an imbalanced data set, accuracy should not be used as a measure of performance because 96% (as given) might only be predicting majority class correctly, but our class of interest is minority class (4%) which is the people who actually got diagnosed with cancer. Hence, in order to evaluate model performance, we should use Sensitivity (True Positive Rate), Specificity (True Negative Rate), F measure to determine class wise performance of the classifier. If the minority class performance is found to to be poor, we can undertake the following steps:
    
    a. We can use undersampling, oversampling or SMOTE to make the data balanced.
    
    b. We can alter the prediction threshold value by doing [probability caliberation](https://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/) and finding a optimal threshold using AUC-ROC curve.
    
    c. We can assign weight to classes such that the minority classes gets larger weight.
    
    d. We can also use anomaly detection.
    
    Know more: Imbalanced Classification

45. 多分类问题如何转二分类方法？

    答：a. 一对多法（one-versus-rest）。把某类样本归为一类，其他归为另一类，k 个类别的样本就构造出了 k 个 SVM；

    b. 一对一法（one-versus-one）。在任意两类样本间设计一个 SVM，k 个类别需要 k(k - 1) / 2 个 SVM；

    c. 层次支持向量机（H-SVMs）。先将所有类别分成两个子类，再将子类进一步划分成两个次级子类，如此循环。

46. 上溢（overflow）和下溢（underflow）是什么，softmax 函数会出现哪种情况，该怎么解决？

    答：上溢即大量级的数被近似为正负无穷时，发生上溢。发生上溢后，这些数值会变为非数值。下溢即有些逼近零的数，如零除或者对零取对数时，得到负无穷，如果对负无穷进一步运算，则会得到非数字。softmax 函数中有指数运算，如果要运算的数过小或过大，则会下溢或上溢。解决上溢的方式是让每一个值都减去最大分量的值，由于这样做之后分母有一项为 1，所以不会出现下溢。同样对于取对数，可以让所有数都加 1。

#### Clustering

1. 手撕 KMeans

    答：
    ```python
    import numpy as np
    
	class KMeans:
	    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
	        self.n_clusters = n_clusters
	        self.max_iter = max_iter
	        self.tol = tol
	        self.random_state = random_state
	
	    def fit(self, X):
	        np.random.seed(self.random_state)
	        n_samples, _ = X.shape
	
	        # Step 1: 初始化质心（随机选择 K 个样本）
	        initial_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
	        self.centroids = X[initial_idxs]
	
	        for i in range(self.max_iter):
	            # Step 2: 分配每个样本到最近的质心
	            distances = self._compute_distances(X)
	            labels = np.argmin(distances, axis=1)
	
	            # Step 3: 更新质心
	            new_centroids = np.array([
	                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
	                for j in range(self.n_clusters)
	            ])
	
	            # Step 4: 判断收敛（质心移动小于 tol）
	            shift = np.linalg.norm(new_centroids - self.centroids)
	            self.centroids = new_centroids
	            if shift < self.tol:
	                break
	
	        self.labels_ = labels  # 保存训练标签
	
	    def predict(self, X):
	        # 计算每个点到所有质心的距离，返回最近质心的索引
	        distances = self._compute_distances(X)
	        return np.argmin(distances, axis=1)
	
	    def _compute_distances(self, X):
	        # 返回 (n_samples, n_clusters) 的距离矩阵
	        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
    ```

2. KMeans 将 m 条 n 维数据进行聚类，一共迭代了 t 次，其中簇的数目为 K，计算时间复杂度和空间复杂度

    答：使用 KMeans 算法的时候，需要对 m 条数据每一维都与 K 个中心进行计算，一共迭代 t次，即一共是 O(mntK) 的时间复杂度，同时每个维度有 K 个聚类中心，要存 m 个 n 维向量，因此空间复杂度为 O(n(m + K))。

3. KMeans 聚类数目选择方法

    答：肘部法则（Elbow Method）：通过绘制聚类数目与目标函数（通常是簇内平方和）的关系图，寻找图像呈肘状的拐点，该拐点对应的聚类数目被认为是合适的选择。
    
    轮廓系数法（Silhouette Method）：通过计算每个样本的轮廓系数，绘制轮廓系数与聚类数目的关系图，选择轮廓系数最大的聚类数目。
    
    Gap统计量（Gap Statistic）：比较聚类结果与随机数据集的差异，选择Gap统计量最大的聚类数目。

4. KMeans 中我想聚成 100 类 结果发现只能聚成 98 类，为什么？

    答：因为聚类过程中可能会产生空簇，可见[例子](https://blog.csdn.net/shwan_ma/article/details/80096408)。

5. 为什么在高维空间中聚类效果会变差？如何应对？

    答：高维导致“距离集中”，影响距离度量有效性（维度灾难）。可先进行降维（如 PCA、t-SNE、UMAP）再聚类。

6. 讲一下 EM 算法，E 步和 M 步的具体步骤，E 中的期望是什么？

    答：初始化参数 $$\theta^{old}$$；E 步：估计 $$p(Z\|X, \theta^{old})$$，求得样本的后验期望值；M 步：根据极大似然估计求得 $$\theta^{new}$$；根据 $$\theta$$，迭代至收敛。

7. KMeans 和 EM 有什么关系，和 GMM 有什么关系？

    答：KMeans 的目标函数（整个数据集点到各自聚类中心的距离的平方和）可以用 EM 算法求解。K-Means 算法归类为 GMM 的 EM 解法的一个特例。

8. DBSCAN

    答：通过点的密度决定簇，密度足够高的区域形成一类，噪声点会被丢掉。不需要指定 k，但参数敏感。

9. IVF（Inverted File Index，倒排文件索引）

    答：将整个向量空间划分为多个“簇（clusters）”，并构建倒排表（inverted list），从而减少实际要比较的向量数量。

#### Decision Tree

1. 决策树

    答：每个内部结点表示一个属性的测试；每个分支表示一个测试输出；每个叶结点代表一种类别。
    决策树的总体流程是自根至叶的递归过程，在每个中间结点寻找一个划分属性，来降低熵。

2. id3 是什么？

    答：利用信息增益（Information Gain，大的特征优选）的决策多叉树。
    $$H(S) = - \sum_{c=1}^C p_c \log_2 p_c$$；$$Gain(S, A) = H(S) - \sum_{v \in Values(A)} \frac{\|S_v\|}{\|S\|} H(S_v)$$。

3. c4.5 是什么？

    答：信息增益容易倾向选择取值多的属性，所以 c4.5 是利用信息增益比（Gain Ratio，大的特征优选）的决策多叉树。
    $$SplitInfo(S, A) = - \sum_{v \in Values(A)} \frac{\|S_v\|}{\|S\|} \log_2 \frac{\|S_v\|}{\|S\|}$$；$$GainRatio(S, A) = \frac{Gain(S, A)}{SplitInfo(S, A)}$$。

4. cart 是什么？

    答：当为分类树，利用基尼系数（Gini index/Gini impurity/基尼不纯度）：基于某个属性，划分成多个子数据集，从子数据集中随机抽取两个样本，类别标志不一样概率，概率越低，说明越纯，即依据该属性划分更有效，因此基尼系数小的优选。$$Gini(S) = 1 - \sum_{c=1}^C p_c^2$$，$$GiniIndex(S, A) = \sum_{v \in Values(A)} \frac{\|S_v\|}{\|S\|} Gini(S_v)$$。
    
    当为回归树，用均方误差或平方残差和作为划分标准，最后每个叶节点输出为该节点下所有训练样本的平均值。

5. 决策树中的特征选择方法有哪些？

    答：分类：信息增益、信息增益比和基尼系数；回归：误差（一般取叶子结点数值的方差和）。

6. 决策树容易过拟合的原因是什么？如何缓解？

    答：训练数据中噪声或特征太多，树可以完美拟合训练集。
    
    缓解方法：
    - 剪枝：预剪枝（在决策树生长过程中，对每个结点在划分前进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分并将当前结点标记为叶结点），后剪枝（先从训练集生成一颗完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能的提升，则将该子树替换为叶结点）
    - 设置最大深度、最小叶子节点数
    - 使用集成方法（如随机森林）

7. Python 相关实现

    答：
    ```python
    import numpy as np
	from collections import Counter
	from math import log2
	
	# ===== 公共工具函数 =====
	def entropy(y):
	    counter = Counter(y)
	    probs = [c/len(y) for c in counter.values()]
	    return -sum(p*log2(p) for p in probs if p > 0)
	
	def gini(y):
	    counter = Counter(y)
	    probs = [c/len(y) for c in counter.values()]
	    return 1 - sum(p**2 for p in probs)
	
	# ===== ID3 =====
	def id3(X, y):
	    base_entropy = entropy(y)
	    best_gain, best_f = -1, -1
	    for f in range(X.shape[1]):
	        new_entropy = 0
	        for v in np.unique(X[:, f]):
	            idx = (X[:, f] == v)
	            new_entropy += len(y[idx])/len(y) * entropy(y[idx])
	        gain = base_entropy - new_entropy
	        if gain > best_gain:
	            best_gain, best_f = gain, f
	    return best_f
	
	# ===== C4.5 =====
	def c45(X, y):
	    base_entropy = entropy(y)
	    best_ratio, best_f = -1, -1
	    for f in range(X.shape[1]):
	        new_entropy, split_info = 0, 0
	        for v in np.unique(X[:, f]):
	            idx = (X[:, f] == v)
	            p = len(y[idx])/len(y)
	            new_entropy += p * entropy(y[idx])
	            if p > 0:
	                split_info -= p * log2(p)
	        info_gain = base_entropy - new_entropy
	        if split_info > 0:
	            ratio = info_gain/split_info
	            if ratio > best_ratio:
	                best_ratio, best_f = ratio, f
	    return best_f
	
	# ===== CART =====
	def cart(X, y):
	    best_gini, best_f, best_v = 1e9, -1, None
	    for f in range(X.shape[1]):
	        for v in np.unique(X[:, f]):
	            left = y[X[:, f] <= v]
	            right = y[X[:, f] > v]
	            if len(left) == 0 or len(right) == 0: continue
	            g = (len(left)/len(y))*gini(left) + (len(right)/len(y))*gini(right)
	            if g < best_gini:
	                best_gini, best_f, best_v = g, f, v
	    return best_f, best_v
	
	# ===== Train / Test =====
	def train(X, y, method="id3"):
	    # 纯叶子
	    if len(set(y)) == 1:
	        return y[0]
	    # 没特征 → 多数表决
	    if X.shape[1] == 0:
	        return Counter(y).most_common(1)[0][0]
	
	    if method == "id3":
	        f = id3(X, y)
	        tree = {f: {}}
	        for v in np.unique(X[:, f]):
	            idx = (X[:, f] == v)
	            subtree = train(np.delete(X[idx], f, axis=1), y[idx], method)
	            tree[f][v] = subtree
	        return tree
	
	    elif method == "c45":
	        f = c45(X, y)
	        tree = {f: {}}
	        for v in np.unique(X[:, f]):
	            idx = (X[:, f] == v)
	            subtree = train(np.delete(X[idx], f, axis=1), y[idx], method)
	            tree[f][v] = subtree
	        return tree
	
	    elif method == "cart":
	        f, v = cart(X, y)
	        if f == -1:  # 无法继续分
	            return Counter(y).most_common(1)[0][0]
	        left_idx, right_idx = (X[:, f] <= v), (X[:, f] > v)
	        return {
	            f: {
	                "<=": train(X[left_idx], y[left_idx], method),
	                ">": train(X[right_idx], y[right_idx], method)
	            },
	            "value": v
	        }
	
	def test(tree, x):
	    if not isinstance(tree, dict):  # 叶子
	        return tree
	    f = list(tree.keys())[0]
	    if "value" in tree:  # CART
	        v = tree["value"]
	        if x[f] <= v:
	            return test(tree[f]["<="], x)
	        else:
	            return test(tree[f][">"], x)
	    else:  # ID3/C4.5
	        v = x[f]
	        if v in tree[f]:
	            sub = tree[f][v]
	            return test(sub, np.delete(x, f))
	        else:
	            return None  # 未见过的取值
	```

#### Dimension Reducing

1. SVD 算法是什么？

    答：$$A=U \sum V^T$$。把一个矩阵 $$A \in \mathbb{R}^{m \times n}$$分解为正交矩阵 $$U \in \mathbb{R}^{m \times m}$$（经过 $$A$$ 变换后的新标准正交基），对角矩阵 $$\sum \in \mathbb{R}^{m \times n}$$ （$$V$$ 中向量与 $$U$$ 中对应向量之间的比例关系，$$\sum$$ 中的每个 $$\sigma$$ 会从大到小排序，值越大代表该维度重要性越高）和正交矩阵 $$V \in \mathbb{R}^{n \times n}$$（原始域的标准正交基）。正交矩阵满足矩阵与矩阵倒置相乘为单位矩阵。

2. PCA，SVD 和 LDA 有什么区别？

    答：PCA 和 SVD 是无监督算法，不需要类别信息，他们都可以将数据投影到低维空间，以最大程度保留原始数据的方差信息。PCA 基本思想：找到数据方差最大的方向，将高维数据投影到这些方向上以实现降维。主要步骤：中心化 → 计算协方差矩阵 → 特征值分解（eigendecomposition）→ 选择前 k 个主成分 → 投影。LDA 是有监督算法，需要类别信息，他在降维的同时让类间距离尽可能大，类内距离尽可能小。

3. 特征标准化有什么意义？怎么做？

    答：消除不同指标量纲的影响。归一化，正态化。

4. 什么是白化？

    答：输入数据分布变换到 0 均值，单位方差的正态分布。但计算代价大，很少用于神经网络。

#### Emsemble Learning

1. ensemble method 中哪种方法降低 bias，哪种方法降低 variance

    答：Bagging 降低 variance，Boosting 既能降低 bias，又能降低 variance。

    根据中心极限定理：样本平均值将会约等于其所在总体的平均值，取样次数越多，结果就越接近正态分布；而且样本大小越大，样本平均值分布的标准差就越小。

    在 Bagging 中，可以把建立每一个分类器的过程都看作给定数据集下的随机变量，把分类器的组合看作样本，很明显分类器越多，预测结果的 variance 就越小。

    Boosting 的每一轮迭代都是基于前面的残差，不断的去学习残差，从而使模型一步步去逼近真实值，整个过程都在优化 loss function，很明显随着迭代次数增加，预测结果的 bias 就越小。另外，boosting 也属于集成学习，是由若干个若分类器组成的一个强分类器，但由于各阶段分类器之间相关性较强，若把 Boosting 也看成一次抽样，变量之间并不相互独立，也不是完全相关，存在一定的互相依赖关系，因此方差降低得较少。

2. 集成方法的个体学习器有什么要求？

    答：好而不同。

3. 集成方法分为什么？有哪些方法？

    答：个体学习器不存在强依赖关系，可同时生成，有 Bagging、Random Forest；个体学习器存在强依赖关系，须串行生成，有 Boosting，AdaBoost，GBDT。

4. 强可学习和弱可学习是什么？

    答：对于存在一个多项式的学习算法，若正确率很高，则是强可学习的，若仅比随机猜测略好，则是弱可学习的。

5. Bagging 的工作机制是什么？

    答：利用自助采样法得到多套数据集，然后通过多个基分类器分类。

6. Boosting 的工作机制是什么？

    答：通过基学习器的表现对训练样本分布进行调整，使得分错样本得到更多关注。

7. 随机森林是什么？

    答：随机森林是 Bagging 的变种，其以决策树为基学习器，并在决策树的训练过程中加入了随机属性选择。其可以处理离散特征（discrete），又可以处理连续特征（continuous）。

8. 随机森林相比普通的 Bagging 的改进是什么？

    答：不仅对样本随机选择，还对特征随机选择。

9. Bagging 的自动校验是什么？

    答：包外估计（out-of-bag estimation）。

10. AdaBoost 是什么？

    答：此方法通过提高前一轮弱分类器错误分类样本权值来提高集成学习效果，它在预测时采用加权多数表决的方法，即加大分类误差率小的弱分类器的权值，减小分类误差率大的弱分类器的权值。

11. GBDT（Gradient Boosted Decision Trees）相对于随机森林的改进是什么？

    答：随机森林中每棵决策树是独立的，而在 GBDT 中，每棵树都是以前一棵树的残差（真实值跟预测值的差值，刚好是平方损失函数的负梯度）为学习目标去拟合。基于残差的 GBDT 对异常值敏感，可以用绝对损失或 huber 损失来代替平方损失函数。

12. 随机森林和 GBDT 的联系和区别？

    答：相同点：都是由多棵树组成；最终的结果都是由多棵树一起决定

    不同点
    - 组成随机森林的树可以分类树也可以是回归树，而 GBDT 只由回归树组成
    - 组成随机森林的树可以并行生成，而 GBDT 是串行生成
    - 随机森林的结果是多数表决表决的，而 GBDT 则是多棵树累加之和
    - 随机森林对异常值不敏感，而 GBDT 对异常值比较敏感
    - 随机森林是通过减少模型的方差来提高性能，而 GBDT 是减少模型的偏差来提高性能的
    - 随机森林不需要进行数据预处理，即特征归一化，而 GBDT 则需要进行特征归一化。

13. GBDT 和 XgBoost 区别？

    答：GBDT 用一阶梯度（如残差），一般仅限 MSE 等简单损失函数，且无正则项。
    
    XgBoost 用一阶 + 二阶梯度（加快收敛，提高稳定性），支持任意可导损失函数（通过泰勒展开），支持 L1、L2 正则，防止过拟合。
    
    XgBoost 将树模型的复杂度（叶节点的数量和叶节点的得分越高，树就越复杂）作为正则项加在优化目标上，公式推导中用到了二阶导数信息（second derivative），支持并行操作。

14. 提升树是什么？

    答：如果用于分类，那就是特殊的 AdaBoost，基分类器都为决策树，回归则情况有所不同。

#### Deep Learning

1. 权重初始化方法？

     答：[链接](https://lpq29743.github.io/artificialintelligence/2017/12/16/TensorFlowInitialization/)
     
     零初始化，常量初始化，高斯/均匀随机初始化，Xavier 初始化，He 初始化，正交初始化。

2. 为什么不能零初始化或常量初始化？

     答：if the neurons start with the same weights, then all the neurons will follow the same gradient, and will always end up doing the same thing as one another.

3. Xavier / He 初始化的目的是什么？

     答：使每一层输出方差为 1。

4. RNN 系列为什么要正交初始化？

     答：RNN 的反向传播本质是权值矩阵连乘，如果矩阵所有特征值绝对值小于 1，则梯度消失，大于 1，则梯度爆炸。

5. 怎么得到正交初始化？

     答：QR 分解或 SVD。

6. 有哪些激活函数？

     答：sigmoid，softmax，tanh，ReLU，PReLU，Leakly ReLU，Maxout。
    ```python
    import numpy as np
     
	# ===== 激活函数 =====
	def sigmoid(x):
	    return 1 / (1 + np.exp(-x))
	
	def sigmoid_derivative(x):
	    s = sigmoid(x)
	    return s * (1 - s)
	
	def tanh(x):
	    return np.tanh(x)
	
	def tanh_derivative(x):
	    return 1 - np.tanh(x)**2
	
	def relu(x):
	    return np.maximum(0, x)
	
	def relu_derivative(x):
	    return (x > 0).astype(float)
	
	def leaky_relu(x, alpha=0.01):
	    return np.where(x > 0, x, alpha * x)
	
	def leaky_relu_derivative(x, alpha=0.01):
	    dx = np.ones_like(x)
	    dx[x < 0] = alpha
	    return dx
	```

7. 激活函数如何选择？

     答：除了 gate 之类的地方，尽量不要用 sigmoid，可以用 tanh 或者 relu 之类的激活函数。

8. 为什么在 CNN 等结构中将原先的 sigmoid、tanh 换成 ReLU 可以取得比较好的效果？

     答：解决了梯度消失问题。sigmoid 导数在两端趋近于 0，容易导致梯度消失；ReLU 在正区间梯度恒为 1，不会出现梯度爆炸/消失问题，支持更深网络训练。

9. RNN 中只能采用 tanh 而不是 ReLU 作为激活函数么？

     答：ReLU 能解决梯度消失，但对 CNN 有效，对 RNN 无效。因为CNN 每一层使用独立的参数不同，原始的 RNN 在每个阶段都共享一个参数。如果直接把 RNN 的激活函数换成 ReLU 会导致非常大的输出值。

10. ReLU 在 0 点的导数是多少？

     答：[链接](http://sofasofa.io/forum_main_post.php?postid=1003784)

11. dying ReLU？

     答：[链接](http://sofasofa.io/forum_main_post.php?postid=1004214)

12. 为什么 softmax 包含 “soft”？

    答：“soft”表示 softmax 函数是连续可导的，以保证梯度下降可以用来优化损失函数。

    “soft” means that the softmax function is continuous and differentiable so that the gradient descent can be used to optimize the loss function.

13. 怎么得到一个 soft 版本的 argmax？

     答：用 softmax 的结果与 index 的倒置相乘。

14. softmax 相比直接 x/sum 的优势

     答：softmax 强调大的值，弱化小的值，差距会被放大，且 x/sum 会受负数影响。

15. Dropout

     答：以一定的概率随机地使一部分神经元节点失效。应用 Dropout 之后，前向传播生成网络结构的过程可以看做服从的分布是伯努利分布。

16. 矩阵计算：AB=C，y=f(C)，y 对 C 的偏导为 P，求 y 对 A 和 B的偏导。

    答：$$PB^T$$ 和 $$A^TP$$。

17. 挑一种激活函数推导梯度下降的过程?

     答：[链接](https://blog.csdn.net/jediael_lu/article/details/77852060)

18. softmax 求导

    答：[链接](https://zhuanlan.zhihu.com/p/25723112)。$$softmax'(z)=softmax(z)(y_i-softmax(z))$$，其中$$y_i$$为标签。如果表示为 Jacobian 矩阵可为$$J_{softmax}=Diag(p)-pp^T$$，其中$$p=softmax(z)$$，而$$Diag(p)$$是以p为对角线的矩阵。

19. argmax 不可导怎么办？

     答：gumbel softmax。

20. SGD、Momentum、AdaGrad、RMSProp、Adam、AdamW、Lion 区别与联系

    答：优化器输入当前参数值，学习率，梯度，输出更新参数值。
    
    SGD 没动量，全局固定学习率。公式为更新参数值 = 当前参数值 - 学习率 * 梯度。在 SGD 中，L2 正则是加上 $$\frac{\lambda}{2} \theta^2$$，weight decay 是 $$\theta \rightarrow \theta - \lambda\theta$$，两者是等价的。SGD 优化轨迹不受损失函数常数缩放影响。
    
    ```python
    def sgd(w, dw, lr=0.01):
	    return w - lr * dw
    ```
    
    SGD + Momentum，固定学习率，Momentum 是一阶动量，即梯度的指数加权平均（exponentially weighted average of past gradients）/梯度滑动平均（EMA，Exponential Moving Average）。越新的梯度权重越大，越久远的梯度，权重按指数衰减。公式为更新参数值 = 当前参数值 - 学习率 * 一阶动量。
    
    ```python
    def momentum(w, dw, v, lr=0.01, beta=0.9):
	    v = beta * v + (1 - beta) * dw
	    w = w - lr * v
	    return w, v
    ```
    
    AdaGrad 会对学习率加权，权重为过往梯度平方和的根号，开根号是为了单位一致性，且保证过往梯度过大导致的学习旅衰减过大。频繁更新的参数，学习率自动变小；很少更新的参数，学习率保持较大。由于每次迭代都有梯度，导致梯度平方和每次都增加，因此学习率会单调递减。
    
    ```python
    def adagrad(w, dw, h, lr=0.01, eps=1e-8):
	    h += dw**2
	    w = w - lr * dw / (np.sqrt(h) + eps)
	    return w, h
    ```
    
    RMSProp 是二阶动量，即梯度的平方滑动平均（exponentially weighted average of the squares of past gradients）。相比 AdaGrad，它更看中最近的梯度，而且由于老的梯度平方会指数衰减，学习率不会无限减小。
    
    ```python
    def rmsprop(w, dw, h, lr=0.001, beta=0.9, eps=1e-8):
	    h = beta * h + (1 - beta) * dw**2
	    w = w - lr * dw / (np.sqrt(h) + eps)
	    return w, h
    ```
    
    Adam 学习率根据一阶动量（Momentum）+ 二阶动量（RMSProp）调整。Adam 会进行偏差修正：由于一阶动量和二阶动量初始值为 0，导致乘了 $$1 - beta1$$ 或 $$1 - beta2$$ 会使得均值被低估，因此需要根据当前 step $$t$$ 进行一阶动量、二阶动量的修正。在 Adam 中，weight decay（L2）会被加到原始梯度里，然后进行同样的一阶动量和二阶动量调整，从而导致实际的权重衰减效果并不等价于直接对权重进行衰减。
    
    ```python
    def adam(w, dw, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
	    m = beta1 * m + (1 - beta1) * dw
	    v = beta2 * v + (1 - beta2) * (dw**2)
	    m_hat = m / (1 - beta1**t)
	    v_hat = v / (1 - beta2**t)
	    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
	    return w, m, v
    ```
    
    AdamW，同 Adam，但权重衰减更合理，将 L2 正则对应的 weight decay 放到了一阶动量和二阶动量调整后的参数更新上。
    
    ```python
    def adamw(w, dw, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
	    # AdamW 在更新前加入权重衰减
	    w = w - lr * weight_decay * w
	    m = beta1 * m + (1 - beta1) * dw
	    v = beta2 * v + (1 - beta2) * (dw**2)
	    m_hat = m / (1 - beta1**t)
	    v_hat = v / (1 - beta2**t)
	    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
	    return w, m, v
	```
    
    Lion，使用一阶动量，只看更新方向，不考虑更新幅度。
    
    ```python
    def lion(w, dw, m, lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.0):
	    m = beta2 * m + (1 - beta2) * dw
	    update = np.sign(beta1 * m + (1 - beta1) * dw)
	    if weight_decay != 0:
	        update += weight_decay * w
	    w = w - lr * update
	    return w, m
    ```
    
    Muon 方向由一阶动量决定，幅度由二阶动量决定。
    
    ```python
    def muon(w, dw, m, v, lr=1e-3, beta1=0.9, beta2=0.99, beta3=0.999, eps=1e-8):
	    m = beta2 * m + (1 - beta2) * dw
	    v = beta3 * v + (1 - beta3) * (dw**2)
	    update = np.sign(beta1 * m + (1 - beta1) * dw)
	    w = w - lr * update / (np.sqrt(v) + eps)
	    return w, m, v
    ```
    
    如果加上 Scheduler，可以动态调整全局学习率。Warmup 策略可以前期慢慢提供学习率，后期用 Cosine Decay（训练初期快速降学习率，防止剧烈震荡：训练刚开始时，模型参数还比较随机，快速降低学习率能避免过大步长导致训练不稳定或发散；但起始时仍保留较大学习率，帮助模型迅速从随机初始化的参数中找到“正确方向”。中期保持学习率相对平稳，助于稳定收敛：进入训练中期，学习率下降变缓，模型有足够的时间在当前的参数空间“细致探索”；平稳的学习率避免过早降低导致训练停滞，同时不给出过大步长打断已有收敛趋势。后期再次快速下降，微调模型细节：训练末期快速将学习率降低到很小，帮助模型“精细调节”参数，减少振荡，提升泛化性能；类似于在优化曲面上的“爬坡”逐渐变得非常缓慢，避免错过局部极小值。），Linear Decay 等方式进行衰减。

21. 神经网络为什么会产生梯度消失现象（vanishing gradient）？

     答：根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都小于 1 的话，多次链乘之后会接近为 0。常见的触发还有：不合适的损失函数，比如 sigmoid，其导数范围为 0（输入很大或很小）到 0.25（输入为 0）；权重初始化太小；RNN 长序列传递。

22. 梯度消失会导致什么后果？

     答：网络训练停滞，参数不更新；深层网络权重学不到有效信息（尤其是靠近输入层）；模型性能很差，无法收敛。

23. 如何避免梯度消失？

     答：ReLU 激活函数来避免 sigmoid/tanh 饱和问题；合理初始化，如 Xavier / Kaiming 初始化保持梯度方差平稳；残差连接（ResNet），梯度直接传递，减小消失风险；BatchNorm / LayerNorm，缓解梯度消失，保持输入稳定；更少的非线性层，或设计合理的网络结构（如 Transformer）；使用 LSTM / GRU（代替普通 RNN），其门控机制，长时间记忆通过加法传递（非乘法），以及乘法和加法混合，可解决 RNN 的长依赖梯度消失问题。

24. 神经网络为什么会产生梯度爆炸现象（exploding gradient）？

     答：根据链式法则，如果每一层神经元对上一层的输出的偏导乘上权重结果都大于 1 的话，多次链乘之后会接近正无穷。常见的触发还有：网络太深；权重初始化得太大；梯度累积导致数值溢出（尤其是 ReLU 激活）；RNN 长序列传递。

25. 梯度爆炸会导致什么后果？

     答：Loss 变为 NaN 或 Inf；参数更新后变成无穷大，模型直接崩溃；不可逆的数值错误。

26. 如何避免梯度爆炸？

     答：梯度裁剪（Gradient Clipping），强制限制梯度的最大值；合理初始化（如 Kaiming），保持梯度方差平衡；正则化（如 Dropout、L2），减小过拟合导致的梯度波动；BatchNorm / LayerNorm，平稳训练过程；调整学习率，学习率太大会加剧梯度爆炸；混合精度训练时使用 GradScaler，防止数值溢出。

27. Loss 出现 NaN

     答：学习率过大，应减小学习率；梯度爆炸，应梯度裁剪；非法数值操作，如`log(0)`、`0/0`、`sqrt(负数)`；输入数据有 NaN，应数据检查、去除异常；使用 FP16 没有 GradScaler，应使用 `torch.cuda.amp.GradScaler()`。

28. Loss 出现 Inf

     答：指数操作溢出（如 `exp(大数)`），应使用数值稳定版 softmax / logsumexp；分母为 0（除零），应加小常数，如 `+1e-8`；梯度爆炸导致权重溢出，应梯度裁剪；数据异常，应检查输入数据，防止极端值。

29. 如何调参？

     答：网格搜索；随机搜索；贝叶斯优化（更高效，所需实验次数更少，但需要构建代理模型）。

30. 多任务如何学习？

     答：[链接](https://zhuanlan.zhihu.com/p/34916654)

31. CNN

     答：CNN 结构组成
     - 输入层：输入图像（比如 28×28 像素的灰度图，或 RGB 三通道彩色图像）。
     - 卷积层：用卷积核 (filter) 在图像上滑动，提取局部特征。例如：边缘、角点、纹理。参数共享：同一个卷积核在整张图像上应用，大幅减少参数量。
     - 激活函数：常用 ReLU (f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x))，增加非线性。
     - 池化层：最大池化 (Max Pooling)、平均池化 (Average Pooling)。降低特征图大小，减少计算量，增强模型的平移鲁棒性。
     - 全连接层：把高维特征展平，输入到分类器或回归器。最后一层通常用 softmax（分类） 或 线性（回归）。
     - 输出层：分类：输出概率分布；回归：输出数值预测。

32. CNN 在卷积和池化过程中，输入特征和输出特征的关系是怎样的？

     答：输出尺寸 = (输入尺寸 - filter + 2 * padding）/ stride + 1。计算尺寸不被整除，卷积向下取整，池化向上取整。

33. RNN 是什么？

     答：$$o_t=\sigma(W_o[h_{t-1}, x_t] + b_o)$$
    ```python
    def rnn_forward(X, Wx, Wh, b, h0):
	    """
	    X: (T, N, D)
	    Wx: (D, H)
	    Wh: (H, H)
	    b: (H,)
	    h0: (N, H)
	    """
	    h = h0
	    for t in range(X.shape[0]):
	        h = torch.tanh(X[t] @ Wx + h @ Wh + b)
	    return h  # 返回最后一步隐藏状态
	```

34. LSTM 是什么？

     答：遗忘门：$$f_t=\sigma(W_f[h_{t-1}, x_t] + b_f)$$，输出 [0, 1]，来表示信息保留程度。

     输入门：$$i_t=\sigma(W_i[h_{t-1}, x_t] + b_i)$$，输出 [0, 1]，来表示信息更新程度。

     输入转换为初步输出：$$\tilde{C_t}=\sigma(W_C[h_{t-1}, x_t] + b_C)$$。

     使用遗忘门和输入门：$$C_t=f_t*C_{t-1}+i_t*\tilde{C_t}$$。

     输出门：$$o_t=\sigma(W_o[h_{t-1}, x_t] + b_o)$$，输出 [0, 1]，来表示信息输出程度。

     得到最终输出：$$h_t=o_t*tanh(C_t)$$。
     
    ```python
    def lstm_forward(X, Wx, Wh, b, h0, c0):
	    h, c = h0, c0
	    H = h0.shape[1]
	    for t in range(X.shape[0]):
	        z = X[t] @ Wx + h @ Wh + b
	        i = sigmoid(z[:, :H])
	        f = sigmoid(z[:, H:2*H])
	        o = sigmoid(z[:, 2*H:3*H])
	        g = torch.tanh(z[:, 3*H:])
	        c = f * c + i * g
	        h = o * torch.tanh(c)
	    return h, c
	```

35. GRU 是什么？

     答：LSTM 的变种，将遗忘门和输入门合在一起，输入门 = 1 - 遗忘门。
    ```python
    def gru_forward(X, Wx, Wh, b, h0):
	    h = h0
	    H = h0.shape[1]
	    for t in range(X.shape[0]):
	        z = X[t] @ Wx + h @ Wh + b
	        r = sigmoid(z[:, :H])
	        u = sigmoid(z[:, H:2*H])
	        g = torch.tanh(z[:, 2*H:] + r * (h @ Wh[:, 2*H:]))
	        h = (1 - u) * g + u * h
	    return h
	```

36. LSTM 和 GRU 的联系和区别？

     答：都是通过使梯度的乘法变成加法，来解决 RNN 由于梯度消失而不能对长期依赖建模的问题。前者三个门，后者两个门，所以前者计算更耗时。

37. 门机制为什么能解决梯度消失或爆炸问题？

     答：[链接](https://zhuanlan.zhihu.com/p/27485750)

38. TensorFlow 和 Pytorch 如何在不同层使用不同的学习率？

    答：[链接](https://zhuanlan.zhihu.com/p/61590026)

39. TensorFlow 和 Pytorch 如何固定参数和 fine-tune？

    答：[链接](https://zhuanlan.zhihu.com/p/61590026)

40. TensorFlow 怎么实现 learning rate decay？

    答：[链接](https://blog.csdn.net/u012436149/article/details/62058318)

41. Pytorch 怎么实现 learning rate decay？

    答：[链接](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/)

42. TensorFlow 内部求导机制？

    答：符号求导。先提供每一个 op 求导的数学实现，然后使用链式法则求出整个表达式的导数。

43. TensorFlow 创建变量的方式有哪些，有什么区别？

    答：`tf.Variable()`和`tf.get_variable()`。前者一律创建新的变量，遇到同名变量，会在后面加后缀 1，2；后者如果遇到同名变量，则使用之前创建的变量，但要求这个变量一定在 variable_scope 中，且有 reuse 选项。

44. Pytorch 如何切换训练和测试模式？

    答：`model.train()`和`model.eval()`

45. `torch.no_grad`和`model.eval`区别

    答：前者相比后者，BN，dropout 都在。两者结合起来，可以取消 BN，dropout 并不计算梯度。

46. Pytorch 的 view 和 reshape 有什么区别？

    答：view 只能用于连续内存的张量；只改变视图，不复制数据（如果张量是连续的）；如果张量不是连续的，会报错；更快（不涉及数据复制）；在对张量做完 `.contiguous()` 后更常用。reshape 可以用于非连续张量（会自动创建副本）；自动处理非连续张量，可能复制数据；自动处理，返回新的张量；稍慢（可能需要复制内存）；更通用，适用于任何张量。

47. GPU 利用率低怎么办？

    答：dataset API 可以支持以 streaming 的方式读取数据。

### Natural Language Processing

#### Traditional Methods

1. TF-IDF 是什么？

     答：词频（TF，Term Frequency）为词在当前文档中出现的频率，逆向文件频率（IDF，Inverse Document Frequency）由总文件数目除以包含该词的文件数目，再将得到的商取以 10 为底得到。
     
     $$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$
     
     $$\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
     
     $$\text{idf}(t, D) = \log \frac{|D|}{1 + |\{ d \in D : t \in d \}|}$$

2. 手动加权或自主学习权重哪个好？

     答：手动加权相当于引入特征，更为合理，自主学习权重相当于学习特征，能够学到隐含信息。

3. 自然语言处理数据增强的方法？

     答：a、加噪，如随机扔词或调整语序；b、同义词替换；c、回译；d、文档裁剪；e、生成对抗网络。

4. HMM 的两个假设是什么？

    答：齐次马尔可夫性假设（当前时刻状态只跟上一状态相关）和观测独立性假设（当前观测只跟当前状态相关）。

5. CRF 是什么？

    答：给定一组输入随机变量，另一组输出随机变量的条件概率分布模型，输出随机变量构成马尔可夫随机场。

6. HMM 和 CRF 的联系与区别

    答：a.HMM 是生成模型，CRF 是判别模型

    b.HMM 是概率有向图，CRF 是概率无向图

    c.HMM 求解过程可能是局部最优，CRF 可以全局最优

    d.CRF 概率归一化较合理，HMM 则会导致 label bias 问题

#### Word2Vec + Neural Networks

1. 为什么不能用 one-hot 表示词？

     答：维度灾难和语义鸿沟。

2. 分布式假设是什么

     答：相同上下文语境的词有类似含义。

3. 了解哪些词向量方法？

     答：固定表征（word2vec，Glove，FastText）和动态表征（cove，elmo，GPT，bert）。

4. word2vec 的原理？

     答：word2vec 有两种模型 CBOW 和 Skip-Gram，前者是在已知上下文的情况下，预测当前词，后者是在已知当前词的情况下，预测上下文。

5. word2vec 的两个模型哪个对小数据集和非词典词比较友好？

     答：CBOW 每个词在进入模型后，都相当于进行了均值处理，所以非词典词在进行加权平均后，也容易被忽视。

     Skip-Gram 每个词都会被单独得训练，较少受其他高频的干扰。所以对于 Skip-Gram 在小数据集和非词典词上占优。

6. word2vec 中的 hierarchical softmax 是什么？

     答：将一次分类分解为多次分类，从而把分类的时间复杂度从 O(N) 降低到 O(logN)，从而提高运算效率。word2vec 根据词频构建了一棵 Huffman 树，来实现 hierarchical softmax。

7. word2vec 中的 negative sampling 是什么？

     答：负采样即在词汇表中采样一定数目的词作为负例，与原先的正例一起做多次二分类（而不是全词表多分类），从而提高模型训练速度。负采样受到词频影响，词频越高的词越可能被采样到。

8. LSA 是什么？

     答：潜在语义分析，先构建一个单词-标题（或文章）矩阵成，然后再用 SVD 算法等进行处理。

9. Glove 的原理？

     答：通过构建词向量和共现矩阵之间的近似关系，来进行训练

10. Glove 和 LSA 有什么联系或区别？

     答：LSA  采用计算复杂度高的 SVD 算法，Glove 可看作是对 LSA 的一种优化。

11. Glove 和 word2vec 有什么联系或区别？

     答：word2vec 是局部语料库训练的，其特征提取是基于滑窗的；而 Glove 的滑窗是为了构建共现矩阵，是基于全局语料的，因此，word2vec 可以进行在线学习，glove 不能。word2vec 是无监督学习；Glove 通常被认为是无监督学习，但实际上 Glove还是有 label 的，即共现次数。word2vec 损失函数是带权重的交叉熵，权重固定；glove的损失函数是最小平方损失函数，权重可做映射变换。总体来看，Glove 可看作是换了目标函数和权重函数的全局 word2vec。

12. FastText 的原理？

     答：FastText 建立在 word2vec 基础上，其用 subword 来对非词典词进行处理，用 n-gram 来考虑词序。

13. elmo 的原理？

     答：基于语言模型的动态词向量。采用 1 层静态向量 + 2 层 LSTM 提取特征，然后将两个方向得到的向量进行拼接。

14. cove 和 elmo 的联系和区别

     答：都用了 LSTM 编码，但前者的输入单位是词，后者是字符。

15. CNN 在文本分类上一般怎么应用？

     答：卷积核宽度为词向量的维度（词向量交换维度并不会起影响，所以没必要在词向量维度做卷积），长度为关注窗口的大小，通道数可为所用词向量。

#### Large Language Models

1. Raw data 数据收集和质量控制

     答：
     - 多样性：去重（Exact Match，Minhash，Simhash。可以做 in-dump 去重，也可以做 cross-dump 去重）；多领域（包括Web，书籍，维基百科，论文，数学，代码。可以用词汇多样性来评估）；多语言。
     - 高质量：规则法（词表匹配），分类器，使用小模型打 PPL。
     - 数据配比：平衡；根据需求；根据表现

2. Instruction data 数据收集和质量控制

     答：基于 dataset；人工标注；模型生成（蒸馏）。相比预训练数据，对质量要求更高。

3. Alignment 数据生成和质量控制

     答：正负例样本。

4. Transformer 的原理？

     答：Transformer 的总体架构是 encoder-decoder，它的主要部分是利用 multi-head attention 去计算词与词之间的相似度。此外，为了融入位置信息，它还提出了 position embedding。

5. TransformerBlock 实现

     答：如果是如 LLaMA 等新型 LLM，则改 Norm 和 激活函数类型，并把 PostNorm 改成 PreNorm。
     
     ```python
	class TransformerBlock(nn.Module):
	    def __init__(self, hidden_size=4096, num_heads=32, dropout=0.1):
	        super().__init__()
	        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
	        self.norm1 = nn.LayerNorm(hidden_size)
	        self.ff = nn.Sequential(
	            nn.Linear(hidden_size, hidden_size * 4),
	            nn.ReLU(),
	            nn.Linear(hidden_size * 4, hidden_size)
	        )
	        self.norm2 = nn.LayerNorm(hidden_size)
	        self.dropout = nn.Dropout(dropout)
	
	    def forward(self, x, mask=None):
	        # Self-attention
	        attn_out = self.attention(x, mask)
	        # Residual Connection
	        x = x + self.dropout(attn_out)
	        x = self.norm1(x)
	
	        # Feedforward with residual
	        ff_out = self.ff(x)
	        x = x + self.dropout(ff_out)
	        x = self.norm2(x)
	
	        return x
	```

6. multi-head attention 的公式是怎样的？

     答：$$Attention(Q,K,V) = softmax({QK^T\over {\sqrt {d_k}}})V$$。
 
7. multi-head attention 实现

     答：
     ```python
     import torch
     import torch.nn as nn
     import torch.nn.functional as F
     
     class MultiHeadAttention(nn.Module):
	    def __init__(self, embed_dim, num_heads):
	        super(MultiHeadAttention, self).__init__()
	        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
	
	        self.embed_dim = embed_dim
	        self.num_heads = num_heads
	        self.head_dim = embed_dim // num_heads
	
	        # Linear layers to project input to Q, K, V
	        self.q_proj = nn.Linear(embed_dim, embed_dim)
	        self.k_proj = nn.Linear(embed_dim, embed_dim)
	        self.v_proj = nn.Linear(embed_dim, embed_dim)
	
	        # Final output projection
	        self.out_proj = nn.Linear(embed_dim, embed_dim)
	
	    def forward(self, x, mask=None):
	        B, T, E = x.size()
	
	        # Linear projections
	        Q = self.q_proj(x)  # (B, T, E)
	        K = self.k_proj(x)
	        V = self.v_proj(x)
	
	        # Split into multiple heads: (B, num_heads, T, head_dim)
	        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
	        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
	        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
	
	        # Scaled dot-product attention
	        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)
	        if mask is not None:
	            scores = scores.masked_fill(mask == 0, float("-inf"))
	        weights = F.softmax(scores, dim=-1)
	        attended = torch.matmul(weights, V)  # (B, num_heads, T, head_dim)
	
	        # Concatenate heads: (B, T, E)
	        attended = attended.transpose(1, 2).contiguous().view(B, T, E)
	
	        # Final linear projection
	        output = self.out_proj(attended)
	        return output
     ```

8. multi-head attention 时间复杂度

    答：$$O(d * seq\_len * seq\_len)$$。

9. Transformer 使用的时候，制约显存的最关键因素是什么？

    答：序列长度。

10. casual mask 怎么生成

    答：
    ```python
	def causal_mask(seq_len):
	    # 下三角矩阵 (seq_len, seq_len)
	    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
	    return mask
     ```

11. 用 multi-head attention 做 cross-attention

    答：拼接后输入。

12. grouped-query attention 实现

	答：grouped-query attention 中，query 使用比 key/value 更多的 heads。因为在推理阶段，Q 是即时计算的，而 K/V 是缓存的。
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
     
	class GroupedQueryAttention(nn.Module):
	    def __init__(self, embed_dim, num_query_heads, num_kv_heads):
	        super(GroupedQueryAttention, self).__init__()
	        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
	        assert embed_dim % num_kv_heads == 0, "embed_dim must be divisible by num_kv_heads"
	
	        self.embed_dim = embed_dim
	        self.num_query_heads = num_query_heads
	        self.num_kv_heads = num_kv_heads
	        self.q_head_dim = embed_dim // num_query_heads
	        self.kv_head_dim = embed_dim // num_kv_heads
	
	        # Projection layers
	        self.q_proj = nn.Linear(embed_dim, embed_dim)
	        self.k_proj = nn.Linear(embed_dim, embed_dim)
	        self.v_proj = nn.Linear(embed_dim, embed_dim)
	        self.out_proj = nn.Linear(embed_dim, embed_dim)
	
	    def forward(self, x):
	        B, T, E = x.shape
	
	        # Project inputs
	        Q = self.q_proj(x).view(B, T, self.num_query_heads, self.q_head_dim).transpose(1, 2)  # (B, QH, T, Dq)
	        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)     # (B, KVH, T, Dk)
	        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)     # (B, KVH, T, Dv)
	
	        # Expand K/V to match query heads if needed
	        if self.num_query_heads != self.num_kv_heads:
	            repeat_factor = self.num_query_heads // self.num_kv_heads
	            K = K.repeat_interleave(repeat_factor, dim=1)  # (B, QH, T, Dk)
	            V = V.repeat_interleave(repeat_factor, dim=1)  # (B, QH, T, Dv)
	
	        # Scaled dot-product attention
	        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.q_head_dim ** 0.5)  # (B, QH, T, T)
	        attn_weights = F.softmax(scores, dim=-1)
	        context = torch.matmul(attn_weights, V)  # (B, QH, T, Dv)
	
	        # Combine heads
	        context = context.transpose(1, 2).contiguous().view(B, T, E)  # (B, T, E)
	        output = self.out_proj(context)  # Final linear projection
	
	        return output
     ```

13. multi-head attention + kv cache 实现

     答：query 不参与下一 token 的注意力过程，无需缓存，而 key/value 是过去的记忆，需要缓存。
     ```python
    import torch
	import torch.nn as nn
	import torch.nn.functional as F
	class SelfAttentionWithKVCache(nn.Module):
	    def __init__(self, embed_dim, num_heads, max_seq_len):
	        super().__init__()
	        assert embed_dim % num_heads == 0
	        self.embed_dim = embed_dim
	        self.num_heads = num_heads
	        self.head_dim = embed_dim // num_heads
	
	        self.q_proj = nn.Linear(embed_dim, embed_dim)
	        self.k_proj = nn.Linear(embed_dim, embed_dim)
	        self.v_proj = nn.Linear(embed_dim, embed_dim)
	        self.out_proj = nn.Linear(embed_dim, embed_dim)
	
	        # 初始化 KV Cache（支持最多 max_seq_len 步）
	        self.register_buffer("key_cache", torch.zeros(1, num_heads, max_seq_len, self.head_dim))
	        self.register_buffer("value_cache", torch.zeros(1, num_heads, max_seq_len, self.head_dim))
	        self.max_seq_len = max_seq_len
	
	    def forward(self, x, start_pos):
	        """
	        x: [B, 1, E] - 当前一步的 token 表示
	        start_pos: int - 当前 token 在生成序列中的位置
	        """
	        B, T, E = x.shape  # T == 1 during generation
	
	        # QKV projection
	        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
	        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
	        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
	
	        # 更新 KV cache
	        self.key_cache[:, :, start_pos:start_pos+1, :] = K
	        self.value_cache[:, :, start_pos:start_pos+1, :] = V
	
	        # 从 0 到当前 step，取所有 KV
	        K_cached = self.key_cache[:, :, :start_pos+1, :]   # [B, H, T_cache, D]
	        V_cached = self.value_cache[:, :, :start_pos+1, :] # [B, H, T_cache, D]
	
	        # 注意力计算
	        scores = torch.matmul(Q, K_cached.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, T_cache]
	        attn_weights = F.softmax(scores, dim=-1)
	        out = torch.matmul(attn_weights, V_cached)  # [B, H, 1, D]
	
	        out = out.transpose(1, 2).contiguous().view(B, T, E)  # [B, 1, E]
	        return self.out_proj(out)
     ```

14. FlashAttention

    答：GPU 存储单元主要有 HBM 和 SRAM：HBM 容量大但访问速度慢，SRAM 容量小但访问速度快。由于这一特性，传统 self-attention 受 context length 限制，无法利用 SRAM，只能对 HBM 进行多次读写操作。而 FlashAttention 通过分块和在线 softmax 的方式，从 HBM 中读取块到 SRAM 中，用 SM（Streaming Multiprocessor）计算之后再写到 HBM 中，减少对 HBM 的访问，从而提高了速度。时间复杂度并没有变化，而空间复杂度因为没有存储中间的 attention matrix，从 O(N^2) 降低到 O(N)。
    
    FlashAttention 基于以下几点：
    
	1. 分块（`for i in range(0, L, block_size)`查询块，`for j in range(0, L, block_size)`键/值块）
    将 Q, K, V 按块划分，分块计算 Attention。每次只加载当前块，节省内存。传统 softmax 需要先计算全部 QK^T，然后减 max 值再求 exp，最后归一化。而 FlashAttention 分块之后，softmax 的计算成为了一个难点。FlashAttention 通过引入两个额外的统计量（`m_prev` 和 `d_prev`）来在线计算 softmax。多个 block 的 softmax 计算可以并行，最后再合并。
    
    2. 重计算
    在后向梯度计算时，一般需要 attention matrix，但 FlashAttention 可以利用两个额外的统计量在 SRAM 上快速重新计算 attention。
	
	```python
	import torch
	import torch.nn.functional as F
	
	def flash_attention_blocked(Q, K, V, block_size=64, mask=None):
	    # Q, K, V: [B, L, D]，Batch x Length x Dim
	    # block_size: 每块序列长度
	    B, L, D = Q.shape
	    output = torch.zeros_like(Q)
	    
	    # 遍历查询块
	    for i in range(0, L, block_size):
	        q_block = Q[:, i:i+block_size, :]   # [B, block, D]
	        out_block = torch.zeros_like(q_block)
	
	        # 记录在线 softmax 的统计量
	        m_prev = None  # 当前块的最大值
	        d_prev = None  # 当前块的 exp 和
	
	        # 遍历键/值块
	        for j in range(0, L, block_size):
	            k_block = K[:, j:j+block_size, :]
	            v_block = V[:, j:j+block_size, :]
	
	            # Attention logits
	            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / D**0.5
	            if mask is not None:
	                scores += mask[:, i:i+block_size, j:j+block_size]
	
	            # 在线 softmax
	            m_new = scores.max(dim=-1, keepdim=True).values
	            if m_prev is None:
	                d_new = torch.exp(scores - m_new).sum(dim=-1, keepdim=True)
	                y = torch.exp(scores - m_new) / d_new
	            else:
	                m_comb = torch.max(m_prev, m_new)
	                d_new = torch.exp(m_prev - m_comb) * d_prev + torch.exp(scores - m_comb).sum(dim=-1, keepdim=True)
	                y = torch.exp(scores - m_comb) / d_new
	
	            # 累加输出块
	            out_block += torch.matmul(y, v_block)
	
	            # 更新统计量，用于下一个键块（在线 softmax）
	            m_prev = m_new
	            d_prev = d_new
	
	        output[:, i:i+block_size, :] = out_block
	
	    return output
	```
	
	FlashAttention v2 改进：
	- 优化计算次序，减少非矩阵计算量。
	- 增加 seq_len 维度的并行计算，提升 SM 利用率。
	- 优化 warp 级工作模式，减少内部通信和 shared memory 访问。
	
	FlashAttention v3 改进：
	- 引入生产者-消费者异步机制，提升并行度。
	- 优化 GEMM 和 Softmax 操作的重叠计算。
	- 支持 FP8 低精度硬件加速，提升吞吐量并减少精度损失。

15. Multi-head Latent Attention (MLA)

    答：在 MHA 中，K 和 V 是对 $$h_t$$ 分别用投影矩阵进行变化得到的，而 MLA 把 KV 的变换改成使用一个共用的 down-projection matrix 将 $$h_t$$ 映射为 $$c_t$$，再用两个 up-projection matrices 将 $$c_t$$ 映射为 $$k_t$$ 和 $$v_t$$。在做 Q、K 点积时，由于 $$k_t$$ 对应的 up-projection matrix 可以被 Q 的映射矩阵（此处也是低秩映射矩阵）吸收，所以 Q、K 点积本质上是 Q 和 C 点积。同理 $$v_t$$ 也不需要计算，因此两个 up-projection matrices 不需要用到，减少了 kv cache 的负担。
    
    由于 MLA 没有显式计算 K，且 ROPE 不能加在 latent vector 上，因此 MLA 使用了 decoupled RoPE，即使用额外的 multi-head queries 和一个 shared key 来携带 RoPE 的位置信息，其维度为 $d_h$。新增的 q 和 k 维度使用常规的 RoPE 计算，用于携带位置信息；而原来的维度依然使用低秩分解的方式计算，最后再计算 attention 的时候两个部分拼接起来。
    
    由于 $$d_c$$ 远小于 $$d_h * seq_len$$，时间复杂度从 $$O(d_h * seq\_len * seq\_len)$$ 降至 $$O(d_h * seq\_len * latent\_len)$$。

16. 为什么要 multi-head

    答：多头注意力允许模型在不同的表示子空间中学习信息，这样可以让模型同时关注不同的信息维度。每个头学习到的信息可以独立地编码输入序列的不同方面，然后将这些信息综合起来，得到更丰富的表示。

17. Transformer 的 Q 和 K 为什么使用不同的权重矩阵生成？如果强行让 Q=K 会发生什么？

    答：注意力将退化为自相似匹配，容易捕捉到 trivial 信息（如位置对称性）；表达能力显著下降，模型性能变差；实际论文实验证明，共用 Q/K/V 权重会损害性能

18. Transformer 为什么是 Q * K^T，而不是 Q + K？

    答：点积是最自然的相似度度量，而加法并不能提供一个明确的匹配度分数，它只是两个向量的混合，没有“匹配程度”的含义。

19. Transformer 为什么是点积，而不是 cosine？

    答：cosine 会归一化，损失模长信息，而且计算复杂度更高。

20. 为什么要除以 $$\sqrt {d_k}$$

    答：Q 和 K 点积可以理解成 $$d_k$$ 项的和。如果不缩放，$$d_k$$ 越大，点积值方差越大，同时点积值过大会导致 softmax 函数偏向某个位置，接近 one-hot，梯度变得非常小。缩放了可以使得方差标准化到 Q 和 K 的方差，这有助于数值稳定性，使得学习过程更加稳定。

21. multi-head attention 的 embed 会不会有低秩的问题，怎么解决？

    答：是的，可能因 head 冗余、聚合退化等原因呈现低秩结构，从而降低表达能力。可以通过正则化（在多头 projection 矩阵上加正交约束）、架构设计、训练策略等方法缓解，并可用奇异值分析评估问题严重程度。

22. 为什么要用 FFN？

    答：引入非线性表达能力，因为 self-attention 是线性的。FFN 通常是两层网络，先升维再降维。

23. 为什么大模型要使用 left padding

    答：left padding KV Cache 在右侧连续生长，无需移动缓存，支持高效批量并发生成，动态 KV Cache 底层优化都支持左填充。right padding KV Cache 在不同位置，难以对齐，难以批量对齐，增加显存开销，很少支持右填充。

24. BPE，WordPiece 和 Unigram 的区别是？

    答：BPE 是基于贪心的频率合并。初始时将文本拆成最小单位（单字符），然后反复合并出现频率最高的连续字符对，直到迭代终止/达到预定词表大小/频率最高的连续字符对低于频率阈值/没有显著提升 token 压缩率。WordPiece（BERT 使用）跟 BPE 类似，不过是根据最大似然估计进行合并。Unigram 基于概率模型，先初始化大量子词候选，然后用 EM 算法估计每个子词的概率，迭代优化删除低概率子词，最终得到固定大小词表。

25. 传统中文分词

    答：前向匹配（Forward Maximum Matching）+ 动态规划（如 Viterbi 算法）

26. Position Embedding

    答：绝对位置编码，即每个位置有个固定编码。包括：
    
    正弦-余弦位置编码 Sinusoidal：无需学习参数。根据当前位置和维度 i 确定编码。偶数位置，使用正弦编码，在奇数位置，使用余弦编码。任意位置的 $$PE_{pos+k}$$ 都可以被 $$PE_{pos}$$ 的线性函数表示。
    
    Learnable Embedding。
    
    相对位置编码，可针对长序列，包括：
    
    RoPE：无需学习参数，对 Query 和 Key 的每个向量维度用旋转变换编码位置信息，因此 attention 结果会依赖于两个 token 的相对距离。
    
    ALiBi：无需学习参数，通过为 Attention 权重加上线性位置偏置来编码位置信息。
    
    YARN：需学习参数，通过为 Attention 权重加上相对位置偏置表示来编码位置信息。
    
    无参数 Position Embedding 支持序列长度外推。

27. 为什么 Position Embedding 可以与 Token Embedding 相加？

    答：在经历线性转换后，concat 与相加是等效的；在高维空间，两者几乎正交，因此相加并不干扰；减少计算量。

28. RoPE 实现

    答：
    ```python
    import torch
	import math
	
	def rope(x):
	    """
	    x: (seq_len, batch_size, d_model)
	    返回加上 RoPE 的 embedding
	    """
	    seq_len, batch_size, d_model = x.shape
	    # RoPE 要求维度是偶数，因为每两个维度为一对进行二维旋转
	    assert d_model % 2 == 0, "RoPE embedding dim must be even"
	    
	    # 每两维组成一对
	    half_dim = d_model // 2
	    # theta 是频率向量，计算不同维度的旋转角度，低维旋转快，高维旋转慢。旋转越慢，越适合捕捉长距离关系
	    theta = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
	    # 序列中每个 token 的位置索引
	    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
	    # 每个 token 每一对维度的旋转角度
	    angles = pos / theta  # (seq_len, half_dim)
	    
	    # 计算 sin/cos
	    sin = torch.sin(angles)  # (seq_len, half_dim)
	    cos = torch.cos(angles)
	    
	    # 将 x 分成两半
	    x1 = x[:, :, 0::2]  # even indices
	    x2 = x[:, :, 1::2]  # odd indices
	    
	    # 旋转
	    x_rotated_0 = x1 * cos - x2 * sin
	    x_rotated_1 = x1 * sin + x2 * cos
	    
	    # 交替组合回原始维度
	    x_out = torch.zeros_like(x)
	    x_out[:, :, 0::2] = x_rotated_0
	    x_out[:, :, 1::2] = x_rotated_1
	    return x_out
    ```

29. 外推性

    答：测试时要接收处理比训练时更长的上下文。

30. 如何提升外推能力

    答：位置编码外推：ALiBi；
    
    长度泛化技术：动态调整 RoPE 的旋转角；
    
    推理策略增强：CoT，Self- Consistency。

31. LLM 常用的激活函数有？

    答：ReLU：f(x) = max(0, x)
     
    GeLU：f(x) ≈ x * Φ(x)，Φ是标准正态分布的累积分布函数。
     
    GLU：GLU(a,b)=a×σ(b)，其中，输入被分成两部分 a 和 b，σ 是 sigmoid 函数。
     
    SwiGLU：SwiGLU = 线性 × SwiSH 激活。Swish 函数代替了原始 GLU 中的 Sigmoid 激活，其为 x 乘以 Sigmoid(x)。
     
    ReLU，GeLU 不能门控，GLU，SwiGLU 能门控。

32. Batch Normalization (BN)

    答：BN 就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

    对于深度学习这种包含很多隐层的网络结构，在训练过程中，因为各层参数不停变化，所以每个隐层都面临 covariate shift 的问题，输入分布老是变来变去，这就是 Internal Covariate Shift，Internal 指的是深层网络隐层，发生在网络内部。BatchNorm 的基本思想是让每个隐层节点的激活输入分布固定下来，避免 Internal Covariate Shift 问题。

    经过 BN 后，大部分 Activation 的值落入非线性函数的线性区内，对应导数远离导数饱和区，加速训练收敛过程。

    BN 为了保证非线性的获得，对变换后的 x 又进行了 scale 加上 shift 操作：y = scale * x + shift。

33. Batch Normalization (BN) vs Layer Normalization (LN) vs RMSNorm

    答：这些都是为了防止梯度消失/爆炸，引入参数为了提高表达能力，从而提高泛化能力。
    
    BN 是跨样本跨 token 统计的，会泄漏信息，所以 LN 更适合变长序列和单样本推理，RMSNorm 参数量（d，缩放因子）为 LN （2d，缩放因子和偏移因子） 一半，更高效和稳定，并表现与 LN 相似。
     
    输入是形状为 `(batch_size, seq_len, hidden_dim)` 的张量，BN 通常对 batch 和 seq_len 两个维度联合计算均值和方差，也就是对每个 hidden_dim 维度独立归一化。LN/RMSNorm 对每个样本每个 token 的 hidden_dim 维度做归一化，即对 seq_len 中的每个位置独立归一化，计算均值和方差都在 hidden_dim 上。

34. 实现 LayerNorm

    答：
    ```python
    import torch
	import torch.nn as nn
	
	class LayerNorm(nn.Module):
	    def __init__(self, dim, eps=1e-5):
	        super(LayerNorm, self).__init__()
	        self.eps = eps
	        self.gamma = nn.Parameter(torch.ones(dim))  # 缩放因子
	        self.beta = nn.Parameter(torch.zeros(dim))  # 偏移因子
	
	    def forward(self, x):
	        mean = x.mean(dim=-1, keepdim=True)
	        var = x.var(dim=-1, unbiased=False, keepdim=True)
	        x_norm = (x - mean) / torch.sqrt(var + self.eps)
	        return self.gamma * x_norm + self.beta
    ```

35. 实现 RMSNorm

    答：RMSNorm 不减去均值，只用输入的均方根（RMS）来进行归一化。它更轻量，计算更快，没有 `mean` 操作。
	```python
	import torch
	import torch.nn as nn
	
	class RMSNorm(nn.Module):
	    def __init__(self, dim, eps=1e-8):
	        super(RMSNorm, self).__init__()
	        self.eps = eps
	        self.scale = nn.Parameter(torch.ones(dim))  # 可学习缩放因子
	
	    def forward(self, x):
	        # 计算 RMS
	        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
	        x_norm = x / rms
	        return self.scale * x_norm
	```

36. Pre Norm 和 Post Norm 有什么区别？

    答：Pre Norm 在子层（Self-Attn / FFN）之前，Post Norm 在子层（Self-Attn / FFN）之后。Pre Norm 更常用，因为其更稳定，更容易收敛。

37. temperature/Top-k/Top-p

    答：temperature：控制采样随机性，温度越高越随机。它的做法是将得到的 logits 除以温度，再作 softmax。当温度为 0 时，相当于 argmax/greedy；当温度为 1 时，相当于 softmax；当温度小于 1，分布变得尖锐，熵降低；当温度大于 1，分布变得平坦，熵升高。
    
    Top-k：采样时只考虑概率最高的 k 个 token，然后再归一化。k = 1 为 argmax/greedy，k = 词表数目就是纯 temperature。
    
    Top-p：累积概率阈值，从高到低计算累计概率值，只采样总概率 ≥ p 的前几个 token。当 p 比较小，其接近于 Top-k，当 p 接近 1，其接近纯 temperature。
    
    对于初始 logits 熵大的，叫做高熵 token，意味着 LLM 在此处犹豫不决；反之叫做低熵 token，意味着 LLM 在这非常自信。在推理阶段，较低的 temperature 会导致多样性降低，较高的 temperature 会导致生成质量降低，产生幻觉。

38. speculative decoding

    答：使用一个小型辅助模型（称为“提议模型”或“draft model”）先快速生成多个候选token序列（草稿）。主模型（大型语言模型）随后只对这些候选进行验证和纠正，而不是每一步都全量生成和计算概率。这种方式能显著减少主模型的计算成本，提高生成速度。

39. Beam Search 实现

    答：
    ```python
    import numpy as np

	def beam_search(start_token, get_next_probs, beam_width=3, max_len=10):
	    """
	    start_token: 初始 token
	    get_next_probs: 函数 f(seq) -> dict{token: prob} 返回下一个 token 的概率
	    beam_width: beam 大小
	    max_len: 最大生成长度
	    """
	    # 初始 beam: list of (sequence, score)
	    beams = [( [start_token], 0.0 )]  # score 用 log 概率
	    
	    for _ in range(max_len):
	        new_beams = []
	        for seq, score in beams:
	            next_probs = get_next_probs(seq)
	            for token, prob in next_probs.items():
	                new_seq = seq + [token]
	                new_score = score + np.log(prob + 1e-12)  # 避免 log(0)
	                new_beams.append((new_seq, new_score))
	        
	        # 选择 top-k
	        new_beams.sort(key=lambda x: x[1], reverse=True)
	        beams = new_beams[:beam_width]
	    
	    return beams  # 返回最终 beam 列表
    ```

40. MoE

    答：MoE 模型中，输入先经过门控网络，分流到 TopK 个 MoE 层里。MoE 层代替传统 Transformer 的 FFN，其中每一个对应的专家通常是 FFN。最终 MoE 层的输出综合得到结果。

41. 为什么 LLM 流行 MoE？

    答：MoE 能显著提高模型容量而不成比例地增加计算成本，且支持 expert parallelism。另外 MoE 提高了模型可解释性。

42. MoE 负载均衡

    答：使用 MoE，模型可能会由于专家 token 分配不均，退化成只用少数几个专家，从而导致参数利用率低，训练/推理时部分 GPU 负载过高，OOM 或速度瓶颈。负载均衡常用方法：用辅助损失（Load Balancing Loss）让实际分配和概率分布尽量接近均匀分布；Capacity Factor（容量限制），即如果一个专家超出容量，多余 token 会被丢弃或 reroute 到别的专家，避免某个专家被塞爆。Token Dropping，即丢掉超额 token（只在训练时，用于正则化），或 Token Rerouting，即把超额 token 转发到第二选择的专家（常见于 top-2 gating）；Noisy Gating，在门控 logits 上加噪声（通常是 Gumbel 或 Gaussian），使 gating 更随机化，防止早期训练时过快收敛到少数专家。Sinkhorn / Optimal Transport Gating（更高级），即用最优传输（OT）方法在 token 和专家之间分配，强制更均匀。比如 BASE Layers、Hash Layers 里会用到。

43. 手撕 MoE

    答：
    ```python
    import torch
	import torch.nn as nn
	import torch.nn.functional as F
	
	class SimpleMoE(nn.Module):
		def __init__(self, input_dim, output_dim, num_experts):
			super().__init__()
			self.num_experts = num_experts
			self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
			self.gate = nn.Linear(input_dim, num_experts)

	    def forward(self, x):
	        gate_probs = F.softmax(self.gate(x), dim=1)  # [batch, num_experts]
	        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, output_dim]
	        gate_probs = gate_probs.unsqueeze(-1)  # [batch, num_experts, 1]
	        return torch.sum(gate_probs * expert_outputs, dim=1)  # [batch, output_dim]
    ```

44. Prefix LM 和 Causal LM 区别是什么？

    答：Causal LM 是单向的，只看左边上下文；Prefix LM 是半双向的，可以看整个 prefix 的信息（左侧上下文），预测后缀。

45. 为什么大部分 LLM 是 decoder-only？

    答：生成范式的统一性；任务更难
    
    双向 attention 的注意力矩阵因为是 n * d 与 d * n 的矩阵相乘，理论上最大秩只能为 min(n, d)，而一般 n 远大于 d，所以 n * n 的注意力矩阵容易退化成低秩状态，而 causal attention 的注意力矩阵是下三角矩阵，其秩为对角线上非零的个数，而因为 softmax 输出为正，因此必然是满秩的，建模能力更强。

46. SFT

    答：选择模型和模版，保证当前模版在当前模型上已有较好的表现。

47. 强化学习和监督学习有什么区别？

    答：监督学习中每一个决策（预测标签）是独立的，它对决策的优化取决于标签。强化学习每一个决策是相互影响的，它对决策的优化取决于延时标签（奖励）。过去的 AI 训练方式主要依赖监督学习，也就是让 AI 通过大量人类标注的数据来学习。换句话说，AI 只是一个“超级记忆机”，它能模仿人类的答案，但却不一定真正理解问题的本质。而强化学习的出现，让 AI 不再是单纯的模仿者，而是能够主动探索、试错、优化自己推理方式的智能体。这就像是在训练一个孩子解数学题，监督学习相当于直接告诉他答案，而强化学习则是让他自己尝试解题，并根据最终的正确率进行调整。

48. PPO

    答：PPO 每一次迭代流程如下：
     
	- 准备 prompt；
	
	- 重要性采样：将 prompt 输入到策略模型（Actor/Policy Model，参数需更新），采样生成多个完整输出（以下只用其中一个输出 o 举例说明），并计算输出 o 的概率：`old_log_probs`。
	
	- 输出 o 被输入到冻结的参考模型（Reference Model），得到`ref_log_probs`和 KL 散度。
	 
	- 输出 o 被输入到冻结的奖励模型（Reward Model），生成该完整输出的结果正确性 score（sample-level 的一个标量），注意只有完整输出的 score 不为 0，不完整输出的 score 都为 0。
	
	- 将过程合理性奖励和结果正确性奖励合并起来，得到最终奖励 reward。对于不完整输出，其 reward 为`ref_log_probs - old_log_probs`，对于完整输出，其 reward 为`ref_log_probs - old_log_probs + score`。
	 
	- 输出 o 被输入到 Critic/Value Model（同步更新，可由 Actor Model 部分参数初始化，或由 Reward Model 初始化），其用 value head 输出每个不完整输出的 $$V(s_t)$$，其物理意义为当前状态下所有 action 的平均预期收益。
	
	- 计算优势 advantages，其物理意义采取当前动作会比平均收益多多少，即相对收益，$$Q(s_t, a_t) - V(s_t)$$。评估这一优势主要有两种方法，每种方法都有其利弊，即：1）蒙特卡洛 (Monte-Carlo，MC)：使用完整输出的 reward。由于奖励稀疏，只在生成最后一个 token 时有奖励，这种方法的方差很大，且从 LLM 中获取足够的样本来使用 MC 进行优化成本很高，但它的偏差很低，因为我们可以准确地模拟奖励；2）时间差分 (Temporal difference，TD)：使用一步轨迹奖励（即衡量刚根据提示生成的单词的优劣），即`advantages = reward - values_response.sum(dim=1) / response_mask.sum(dim=1)`。通过这样做，我们可以在 token 级别计算奖励，这显著降低了方差，但同时偏差会增加，因为我们无法从部分生成的响应中准确预测最终奖励。这就是 GAE 的用武之地，它提出通过多步时间差分 (multi-step TD) 来平衡偏差和方差。具体是从 reward 回溯分配每个 token 的 TD 残差 $$\delta_t$$，用 GAE 计算每个 token 的优势 $$A_t$$，其中 gamma 是时间折扣因子，控制未来奖励的重要性，越大代表未来奖励越重要。lambda 是 GAE 平衡因子，控制 bias-variance 权衡，lambda 越大 → 方差大，偏差小；λ 越小 → 方差小，偏差大。
    
    ```python
    def compute_gae(rewards, values, gamma=1.0, lam=0.95):
	    advantages = torch.zeros_like(rewards)
	    last_adv = 0
	    for t in reversed(range(rewards.size(1))):
	        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
	        advantages[:, t] = last_adv = delta + gamma * lam * last_adv
	    return advantages
     ```
	 	
	- 根据采样到的数据进行多次策略迭代更新，每次更新之后得到`log_probs`和新的`values`。
	
	- 用以下 loss 对 Actor/Policy Model 进行优化，剪切函数限制策略更新幅度，确保数值稳定性。当 $$A_t > 0$$，意味着 critic model 对当前 action 做出了正反馈，因此 $$r_t(\theta)$$ 要提高，反之要降低。
	  
	$$L^{\text{clip}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1 - \epsilon,\ 1 + \epsilon) \hat{A}_t \right) \right]$$
     
    其中 $$t$$ 为当前 token，$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$ 为重要性采样比率，$$\hat{A}_t$$是优势函数的估计，$$\epsilon$$ 是控制策略变动幅度的裁剪阈值（如 0.2）。
	 
    ```python
    def actor_loss(log_probs, old_log_probs, advantages, clip_range=0.2):
	    ratio = torch.exp(log_probs - old_log_probs)  # [B]
	    unclipped = ratio * advantages
	    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
	    loss = -torch.min(unclipped, clipped).mean()
	    return loss
	```
	
	- 再根据`rewards`和`values`得到`critic_loss`，优化 Critic/Value Model。
	
	- `actor_loss`和`critic_loss`加权求和后用来最终优化。

49. PPO 有了 reward model 为什么还要 critic/value model？

     答：critic/value model 是内部奖励，仅需当前上下文，会在 RL 过程中更新，reward model 是外部奖励，需要完整回答，是训练好的。

50. 为什么 PPO 用 reward model 而不是 LLM-as-a-Judge？

     答：需要用标注样本训练；分类模型代价低。

51. DPO

    答：
     
    $$L^{\text{DPO}}(\theta) = -\log \left( \frac{\exp\left( \beta \cdot \log \pi_\theta(y^+ \mid x) \right)}{\exp\left( \beta \cdot \log \pi_\theta(y^+ \mid x) \right) + \exp\left( \beta \cdot \log \pi_\theta(y^- \mid x) \right)} \right)$$
     
    其中，$$y^+$$ 是人类偏好的回答，$$y^-$$ 是较差的回答，$$\beta$$ 是温度系数，控制偏好强度
    ```python
    def dpo_loss(logp_chosen, logp_rejected, beta=0.1):
	    diff = logp_chosen - logp_rejected  # [B]
	    loss = -torch.nn.functional.logsigmoid(beta * diff).mean()
	    return loss
    ```
 
52. GRPO
   
    答：
     
    $$L^{\text{GRPO}}(\theta) = - \log \left( \frac{\exp\left(R_\theta(x, y^+)\right)}{\exp\left(R_\theta(x, y^+)\right) + \exp\left(R_\theta(x, y^-)\right)} \right)$$
     
    其中，$$R_\theta$$ 表示奖励形式的打分函数：
     
    $$R_\theta(x, y) = \beta \cdot \left( \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x) \right)$$
     
    $$\pi_{\text{ref}}$$ 是参考策略（例如预训练模型），用于提供稳定的对比基准。
    
	GRPO 流程如下：
     
	- 查询 q 是任务输入，例如一个上下文或状态；
	
	- 输入到策略模型（Policy Model），生成对应的多个输出 o_1, o_2, ..., o_G（动作或结果），即用可更新的 LLM 生成 q 的 o_1, o_2, ..., o_G；
	 
	- 输出 o_i 被输入到冻结的奖励模型（Reward Model），可为训练的，也可为基于规则的，生成奖励 r_i（通常是 sample-level 的一个标量），用于衡量 o_i 的质量；
	 
	- 根据 r_1, r_2, ..., r_G，计算奖励均值和奖励标准差，得到 o_i 的相对奖励，即 advantages；​
	 
	- 根据相对奖励，得到每一个样本的 loss，进行优化；
     
    ```python
    def grpo_loss(group_log_probs, group_old_log_probs, group_advantages, clip_range=0.2):
	    # 计算每个组的 ratio
	    ratio = torch.exp(group_log_probs - group_old_log_probs)  # [G, B]
	    
	    # 计算组内相对优势（相对于组内其他策略优势的平均）
	    mean_advantages = group_advantages.mean(dim=0, keepdim=True)  # [1, B]
	    relative_advantages = group_advantages - mean_advantages     # [G, B]
	    
	    # 计算组内相对 ratio（相对于组内其他策略 ratio 的平均）
	    mean_ratio = ratio.mean(dim=0, keepdim=True)  # [1, B]
	    relative_ratio = ratio / (mean_ratio + 1e-8)  # [G, B]
	    
	    # Unclipped and clipped losses 基于相对比率和相对优势
	    unclipped = relative_ratio * relative_advantages
	    clipped = torch.clamp(relative_ratio, 1 - clip_range, 1 + clip_range) * relative_advantages
	    
		# 对所有组和批次求平均，取最小
	    loss = -torch.min(unclipped, clipped).mean()
	    return loss
    ```
    
    - 输出 o_i 被输入到冻结的参考模型（Reference Model），计算输出 o_i 与参考策略之间的 KL 散度，用于限制策略更新。

53. DAPO

    答：DAPO 主要是根据 GRPO 进行改进，主要改进点为
    - 去掉了 KL 散度，KL 散度可以限制模型同初始模型不会显著偏离，但是在训练 long-CoT reasoning model 时，模型分布会显著偏离初始模型，所以去掉 KL 散度的约束。
    - 提高剪切上限（Clip-Higher）以避免熵过早坍缩，导致某些组生成的结合相同，限制探索
    - 动态采样（Dynamic Sampling）解决一组输出准确率为 1 或 0 时的梯度消失导致 Policy 没有优化，样本利用效率降低的问题
    - GRPO 先在样本内按 Token 数平均 loss，再在样本间聚合 loss，从而导致较长样本和较短样本的损失贡献是一样的，即对于答案正确的，GRPO 偏向于选择答案长度较短的回复，而对于答案错误的，GRPO 偏向于让模型生成更长的回复。DAPO 改进为 Token-Level 策略梯度损失
    - 在 RL 训练中，一般会设置最大长度，对过长回复进行截断，从而其 reward 会为 -1，扰乱训练。DAPO 设置了对长序列的合理惩罚（Overlong Reward Shaping），避免过长后截断导致模型无法得到奖励的情形，以缓解噪声并稳定训练。

54. GSPO

    答：重要性采样修正不再对应 token 级别，而是对应序列级别。

55. on-policy vs off-policy

    答：数据来源于当前策略生成为 on-policy，其数据只能更新一次 policy，之后需要用更新的 policy 重新采样数据，因此利用效率低；数据来源于历史偏好数据为 off-policy。

56. PPO vs DPO vs GRPO

    答：所有算法都需要加 KL 散度来控制模型不要过于远离原先模型。PPO 是 token-level，DPO/GRPO 是 sample-level，但 GRPO 可以回传到 token-level。PPO 依赖于 reward model 和 value model；DPO 没有显式探索机制。

57. GRPO 怎么去掉 critic/value model 的？

     答：采样多次，用 reward model 评价的平均值来充当 critic/value model

58. KL 散度的四种计算方式

59. 熵控制在强化学习里的作用

     答：在大模型训练的强化学习阶段，设置较高的 temperature 可以防止模型过度自信，鼓励模型采取高熵动作，从而扩大探索空间。另一种方式是在 group-level 用 smi/dpp/self-bleu 计算多样性，进行 reward shaping 来控制熵的变化。
     
     熵坍塌：随着训练的进行，entropy 逐渐降低。导致某些 group 采样出的 response 几乎相同，使得模型在早期变得更加确定，限制了模型的探索空间。

60. RLVR

     答：用 Verifier，通过与预设的答案或规则相比较，给出一个二元值，这种方式仅适用于有标准答案的场景，而在开放问题中则不太适用。

61. LoRA

     答：LoRA 的公式为 $$W‘ = W + \alpha * BA$$，$$A \in R^{r \times d}$$，$$B \in R^{d \times r}$$，A 用的是小的高斯随机初始化，B 用的是全 0 初始化，所以初始时 W = W’，$$\alpha$$ 是缩放因子，用于控制 LoRA 注入的权重大小。target_modules 一般为`q_proj`、`v_proj`，有时也会注入到 `k_proj` 或 `o_proj`。modules_to_save 表示指定哪些原模型模块需要一起训练 & 保存，如果扩展了词表可能要加 `embed_tokens`、`lm_head`。

62. 手撕 LoRA

     答：
     ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

	class LoRALinear(nn.Module):
	    def __init__(self, in_features, out_features, r=4, alpha=1):
	        super().__init__()
	        self.r = r
	        self.scale = alpha / r
	        self.weight = nn.Parameter(torch.randn(out_features, in_features))
	        self.weight.requires_grad = False
	        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
	        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
	        self.bias = nn.Parameter(torch.zeros(out_features))

	    def forward(self, x):
	        base = F.linear(x, self.weight, self.bias)
	        lora = F.linear(x, self.B @ self.A) * self.scale
	        return base + lora
    ```

63. Adapter

     答：插入小型网络模块

64. Prefix Tuning

     答：Prefix Tuning 会为每层添加一组虚拟的 Key 和 Value，Query 保持不变。embedding 的输入不会添加。

65. Base model eval

     答：General Tasks: MMLU (5-shot), MMLU-Pro (5-shot, CoT), MMLU-redux (5-shot), BBH (3-shot, CoT), SuperGPQA (5-shot, CoT).
     
     Math & STEM Tasks: GPQA (5-shot, CoT), GSM8K (4-shot, CoT), MATH (4-shot, CoT).
    
    Coding Tasks: EvalPlus (0-shot), MultiPL-E (0-shot), MBPP-3shot, CRUX-O of CRUXEval (1-shot)
    
    Multilingual Tasks: MGSM (8-shot, CoT), MMMLU (5-shot), INCLUDE (5-shot).

66. Chat model eval

     答：General Tasks: MMLU-Redux, GPQADiamond, C-Eval, LiveBench.
     
     Alignment Tasks: IFEval, Arena-Hard, AlignBench, Creative Writing V3, WritingBench.
     
     Math & Text Reasoning: MATH-500, AIME’24, AIME’25, ZebraLogic, AutoLogi.
     
     Agent & Coding: BFCL v3, LiveCodeBench, Codeforces Ratings
     
     Multilingual Tasks: instruction following - Multi-IF, knowledge - INCLUDE & MMMLU, mathematics - MT-AIME2024 & PolyMath, and logical reasoning - MlogiQA.

67. Safety / Halluciation

    答：出现幻觉原因：1. 语料中存在过时，虚构的内容，或因长尾效应缺乏与下游任务相关的领域知识；2. 语言模型的本质机制是预测下一个最可能的词，它只保证语言上看起来连贯合理，并不保证事实正确，所以它倾向即使不知道，也会编一个出来，在不确定时依然输出确定性答案，很少说我不知道；3. 推理时随机采样的生成策略。
    
    解决方案：提高训练数据质量；RAG 提供权威资料；Prompt Engineering：明确告诉模型不要编造、请回答已知事实，或让模型先思考再输出（如 Let’s think step by step）；生成之后进行事实校验，如比对知识图谱或自动校验；RLHF；多模型协作。

68. Long Context

    答：位置编码改进；模型结构优化；记忆缓存机制；检索增强（RAG）；分块/窗口机制；扩展训练数据。

69. LLM设计中的 System 1 和 System 2

    答：默认模式是 System 1：标准的自回归生成，快速但单步预测。
     
     通过 Prompt Engineering 或架构设计激活 System 2：
    
    - Chain-of-Thought（思路链）提示，引导模型一步步“推理”。
        
    - 多阶段推理框架，如 ReAct、Self-Ask、Tool Augmentation。
        
    - 结合检索（RAG）、记忆模块或外部计算器等工具。

70. LLM + 知识

    答：RAG 可以解决 LLM 知识过时，幻觉问题以及无法调用私有数据等问题。
    
    RAG
    - Indexing：小 chunk 语义更纯净，对齐一个单一主体；向量表示更聚焦，易被检索模型命中；数量多，覆盖面大，召回率高，但一个问题需要多个 chunk 才能解决。大 chunk 信息更完整，但主题分散，检索难度大。
    - Pre-Retrieval
    - Retrieval
    - Post-Retrieval (Re-ranking, Prompt Compression)
    - Generation
    
    document 的顺序会对 RAG 的性能造成比较大的影响。
    
    另一种方式是 search engine as a tool。

71. 文本分块

    答：文本分块需考虑平衡信息完整性和检索效率。最常见的方式是根据标点符号和长度切。

72. Reasoning

    答：Prompting：CoT，ToT，Self-Consistency，s1。
    
    改进模型/系统结构：RAG，Memory，Tool Use。
    
    改进学习方式：SFT，RLHF，Critic Models：PRM 和 ORM。

73. Test-time Scaling

    答：实现 test-time scaling，需要先激励 LLM 在 thinking 上耗费更多资源，从而生成更长的回答，或者更多的回答。
    
    更长的回答可以通过如 CoT 的 prompting，如 s1 的改变解码策略。
    
    更多的回答可以通过如 Self-Consistency 的 Parallel Scaling，如 Self-Refine 的 Sequential Scaling，如 MoA 的模型混合。
    
    获得回答之后，需要用 PRM 或 ORM 进行验证。PRM 有助于缩小搜索空间，相比于 ORM 的奖励稀疏，它的奖励更加密集。它的实现包括训练一个独立的模型。ORM 的实现包括训练一个独立的模型，self-consistency，voting 或如 deepseek 的启发式验证。
    
    另外一种方案是搜索，如 ToT，MCTS，Beam Search。
    
    提供最终答案的方式包括 Best-of-N，self-consistency，拒绝采样。

74. Agent

    答：Agent = LLM + Planning + Memory + Tool。
    
    Planning：Task decomposition（CoT，ToT），Self-Reflection（ReAct）。
    
    Memory：short-term（ICL），long-term。
    
    Single-Agent: AutoGPT
    
    Multi-Agent: HuggingGPT

75. MCP 和 function calling 有什么区别？

    答：MCP 可以在一次回复中调用多个函数，function calling 每轮最多调用一个函数。

76. LangChain

    答：LangChain 让你像搭乐高一样搭建一个 LLM 应用，串起来 Prompt、模型、知识库、工具、记忆等组件，快速构建复杂应用。

77. Coding

    答：Agent，RLVR，Long Context

78. bf16，fp16，fp32，int8 区别

    答：指数位决定了数值范围，尾数位决定了精度。bf16 保留了 fp32 的指数位，只截断尾数，精度略低于 fp16，但数值范围与 fp32 一致。int8 可用于量化，因为整数乘法比浮点乘法快，且用缩放映射保留大部分信息。合理设置 scale 和 zero-point，配合 clip 操作，可以安全地把浮点数映射到 int8，不会溢出。

79. 混合精度计算

    答：fp16/bf16 做前向 & 反向传播，fp32 保存主权重。

80. 估算 LLM 的参数量

    答：embedding 层的维度为 Vh，若不与输出层的权重矩阵共享，则需加上输出层的权重矩阵 2Vh。
    
    Transformer 每一层分为 self-attention 和 MLP，self-attention 设计 Q，K，V，O 四个权重矩阵和偏置，因此是 4h^2 + 4h。MLP 一般有两层，先升维再降维，如升到 4h，那么参数量为 8h^2 + 5h。两个模块都有 layer normalization，包含两个可训练参数，形状都为 h，所以参数量总和为 4h。因此，每一层参数量为 12h^2 + 13h。

81. 估算 7B 模型在训练和推理时的显存占用

    答：模型大小（参数量） × 精度 = 参数显存占用，fp16/bf16 精度为 2 字节，fp32 精度为 4 字节。
    
    训练显存 ≈ 模型参数 × 3 或 4（包括权重 + 梯度 + Optimizer 状态 * 1 或 2） + 激活（反向传播时，需要用它来计算梯度），主要瓶颈是激活值和优化器状态。batch_size 越大，激活越大。如果是强化学习，可能还有考虑其他模型的显存占用以及 rollout 的代价。
    
    推理显存 ≈ 参数显存 + batch_size × seq_len × num_layers × hidden_size × 2 × bytes，主要瓶颈是 KV Cache。 

82. 多卡多机训练

    答：Data Parallel：数据被切分成小批量（mini-batch），分别送到不同 GPU，但模型必须能放进单卡显存，无法解决超大模型训练。
    
    Tensor Parallel：单层内的权重矩阵切分到多张卡上，比如一个大矩阵做列切分/行切分。可以训练单卡装不下的层，需要频繁跨卡通信，通信开销大。
    
    Pipeline Parallel：把整个模型按层切分到不同的 GPU/节点上。显存分布在多卡上，可训练更深的模型。
    
    Expert Parallel：在 MoE 模型里，每个样本只激活部分专家网络。专家被分配在不同 GPU/节点上。

83. DataParallel（DP）和 DistributedDataParallel（DDP）区别

    答：DP 单进程，多 GPU（主卡调度），主卡负责 forward/backward；DDP 多进程，每个 GPU 一个进程，每卡独立计算 + 自动同步梯度。

84. PD 分离

    答：Prefill 阶段对初始提示（Prompt）进行处理，生成初始的隐藏状态（Hidden States）。这个阶段通常涉及对整个模型的一次前向传播，是并行计算，因此计算密集度较高，以矩阵乘法为主，GPU 利用率高。对于每个新的输入序列，都需要进行一次 Prefill。
    
    Decode 阶段：在 Prefill 阶段之后，模型基于初始隐藏状态逐步生成后续的文本。这一阶段的特点是计算相对较少，但需要反复进行计算，是串行计算，直到生成足够的文本或达到某个终止条件。在生成过程中，只计算最新的 token 激活值，并进行 attention 计算，计算最终的预测 token。其访存频繁，因此内存带宽受限，且 GPU 利用率低。
    
    Prefill 阶段是计算密集型操作，需要大量并行计算能力；而 Decode 阶段则是内存密集型操作，更依赖高带宽内存访问。
    
    PD 分离一般涉及三要素，调度器、Prefill 实例和 Decode 实例。调度器负责对外发布推理接口，P、D 负责各自推理阶段的计算。P、D 一般在不同的机器资源上运行。具体来说，Prefill 阶段被分配到专门的高算力 GPU上 执行，以充分利用其并行计算能力；而 Decode 阶段则被分配到具有大显存和高内存带宽的 GPU 上执行，以满足其内存访问需求。两个阶段之间通过高速网络（如 NVLink 或 RDMA）传输中间状态（主要是 KV 缓存）。

85. 为什么 MoE 训练使用 Expert Parallelism 而不是 Tensor Parallelism

    答：MoE 用 gating 网络在多个专家中选择最合适的几个来处理输入，因此 Expert Parallelism 不会损失 Data Parallelism 的数量，因为不同 Expert 处理不同的 Data

86. deepspeed 的 Zero-1， Zero 2， Zero 3

    答：Zero-1 优化器状态拆分（例如 Adam 的动量），Zero-2 再加梯度拆分，Zero-3 参数也切分，每卡只保存部分权重。三个模式支持自动 Offload 到 CPU / NVMe，进一步节省显存。参数、梯度、优化器状态始终绑定，分配到同一张 GPU 上。

87. 量化

    答：PTQ（训练后量化）和 QAT（训练时量化）。
    
    动态量化只量化 weight，激活仍是浮点，推理过程中，根据输入数据动态计算激活量化参数。静态量化收集代表性数据集，统计激活最大/最小值，提前固定 scale/zero_point，推理时直接用事先量化好的权重和激活，无需动态计算。
    
    GPTQ (GPT Quantization) 的主要创新是它采用逐层、逐通道的方式优化量化参数，使用二次误差最小化方法来确定最佳量化值，并通过重建误差传播来补偿量化误差。这种方法在保持模型性能的同时实现了高压缩率。
    
    AWQ (Activation-aware Weight Quantization) 改进 GPTQ，减少激活主导的精度偏差。核心思想是根据激活值的重要性选择性地量化权重。

88. vllm

    答：传统的静态分配 KV 缓存不使用虚拟内存，直接对物理内存进行操作，会导致显存碎片和过度预留，因此 vllm 使用了 PagedAttention，即把 KV 缓存当作虚拟内存，每条序列的缓存被划分成块，可动态分配到显存中，允许在不连续的内存空间中存储。
    
    另外 vllm 的 PagedAttention 使用了 memory sharing，即单个 prompt 生成多个序列时，可以共享显存。

89. GPT 的原理？

    答：基于语言模型的动态词向量。采用单向的、多层的、并行能力强的 Transformer 提取特征，利用到的是 Transformer 的 decoder 部分，见到的都是不完整的句子。

90. BERT 的原理？

    答：基于语言模型的动态词向量。采用双向的、多层的、并行能力强的 Transformer 提取特征，利用到的是 Transformer 的 encoder 部分，采用了完整句子。

91. BERT 的训练目标？

    答：BERT 有 masked language modeling 和 next sentence prediction 两个目标

92. RoBERTa 相比 BERT 做了哪些改进？

    答：更大的训练数据；移除 Next Sentence Prediction（NSP）任务，发现没有它模型更稳定、更强；更长时间的训练；更大的 batch size 和学习率调度优化；BERT 的 masking 是静态的（数据预处理阶段决定），RoBERTa 每个 epoch 随机重新 mask。

93. RoBERTa 强于 RNN 的地方？

    答：并行，对大数据比较友好。

94. Qwen

    答：QwenMoE

95. Deepseek-V1 - Deepseek-V3

    答：
    - MLA（Multi-Head Latent Attention）机制，通过引入一个中间稀疏表示（Latent）空间，在推理（inference）阶段有效节约了 KV-Cache 的内存使用和访问开销。
    - Multi-Token Prediction
    - 细粒度专家划分：在保持参数数量不变的情况下，通过分割 FFN 中间隐藏维度来将专家分割成更细的粒度。相应地，在保持计算成本不变的情况下，可激活更多细粒度的专家，以实现激活专家组合的更高灵活性。
    - 共享专家隔离：将某些专家隔离出来，作为始终激活的共享专家，旨在捕获不同上下文中的共同知识。通过将共同知识压缩到这些共享专家中，可以减轻其他路由专家之间的冗余，这可以提高参数效率，确保每个路由专家专注于不同方面而保持专业化。
    - 除了专家级负载均衡，v1 还引入了设备级负载均衡。v2 引入了更多的 loss。v3 直接把这些 loss 都去掉，用一个可动态调节的 bias 来做到负载均衡。当检测到专家是过载的状态时，就减小该专家的 bias，反之则增加。
    - v3 将门控函数的对更小的小数位会敏感的 softmax（multi-class classification）改成了值域更宽的 sigmoid（multi-label classification）
    - fp8 精度计算

96. Deepseek-R1-Zero

    答：证明了在没有任何人类标注数据做 SFT 的情况下，RL 也可以取得不错结果。
    1. 采用 GRPO 算法，去除了 value model，显著降低 RL 训练成本，提高训练稳定性。与此同时，GRPO 让 AI 生成多个答案，并计算每个答案的得分，通过奖励机制来告诉 AI 哪个回答更好。
    2. 基于规则的奖励机制，包括准确性奖励：依据任务的正确性，如数学题的标准答案或代码编译结果进行评估；格式奖励：要求模型在回答中使用 `<think>` 标签包裹推理过程，用 `<answer>` 标签包裹最终答案。不使用神经网络奖励模型，以避免奖励欺骗（Reward Hacking）。
    3. R1-Zero 存在重复内容，可读性差，语言混杂和早期阶段难以收敛的问题。

97. Deepseek-R1

    答：成功经验
    - 在 SFT 阶段采用冷启动，只使用了少量（几千条）高质量的冷启动数据进行 SFT，然后再大规模 RL。冷启动数据主要生成方式：通过 Few-shot Prompting 生成长链式推理数据 (Long CoT)；收集并优化 DeepSeek-R1-Zero 生成的高质量输出；由人工标注者进行后期筛选与润色。
    - 在接近收敛时，通过拒绝采样生成 SFT 数据，即让 AI 生成多个答案，然后只选择最优的答案来继续 SFT 训练，最后再加入非推理类任务数据 (如写作、问答等)，进行全场景强化学习。
    - 大模型的推理能力可以蒸馏到小模型，使其性能优于直接对小模型进行 RL 训练。
    - 为解决 language mixing 的问题，在 RL 阶段增加一致性奖励项，计算目标语言词汇占比
    
    失败经验
    - 过程奖励模型：思维过程正确，行动过程正确。希望用来解决奖励黑洞，但发现只能用于简单的推理任务
    - 蒙特卡洛树搜索（MCTS）：由于推理任务的搜索空间远比围棋复杂，AI 需要在每一步做出决策，而 MCTS 无法有效地指导 AI 进行合理的搜索。

#### Search/Recommendation

1. 搜索/推荐 Pipeline

    答：由于海量数据，所以不是端到端。
    Recall，Rank，Rerank

2. 浏览器的联想词运用了什么理论和原理？

    答：贝叶斯原理。

3. CTR（点击率预估）

    答：样本偏差包括：
    
	1. 选择性偏差（Selection Bias）：用户只点击曝光的部分结果，非点击不代表不感兴趣。
	
	2. 位置偏差（Position Bias）：排名靠前位置点击率更高，影响真实兴趣判断。

4. NDCG

    答：衡量推荐结果的排序质量，考虑点击或相关性得分及其在列表中的位置。结果越靠前且相关性越高，分数越大。

5. map，mrr 是什么？

     答：[链接](https://blog.csdn.net/hackerzer/article/details/79632951)

6. 推荐系统召回策略

    答：1. 热门召回：浏览/点击量统计。通用性强、无冷启动问题
    2. 协同过滤召回：UserCF、ItemCF。利用用户-物品交互的相似性
    3. 向量召回：Item2Vec、DSSM、双塔模型。能捕捉语义和行为的相似性

7. 推荐系统多路召回

    答：通过多个不同的召回策略或通道，并行生成多个初步候选 item 集合，再融合形成最终的召回候选集合，送入排序模型。
    
    1. 合并取 Top-N：从每路召回中取 top-K，合并去重；
    2. 加权融合：每一路打分加权（可用模型或策略决定）；
    3. 训练融合：用模型（如召回 ranker）对召回候选打分融合。

8. 精排模型

    答：DIN

9. 多目标排序

10. 推荐系统冷启动

    答：用户冷启动和物品冷启动。元信息（如用户基本属性、物品描述、标签等），尽可能收集各种信息（问卷调查）。

11. 推荐系统如何解决长尾问题？

    答：提升长尾内容曝光。

### CV and position related

#### Basic

1. Choose one paper

     答：multilingual，coding，RAG，reasoning

2. Choose one LLM

     答：Qwen, LLaMA, GPT, deepseek

3. Choose one project

	答：task description; solution; results; future work; challenges

4. 将已有项目与热门话题链接

#### Sentiment Analysis

1. 情感分析相对于一般文本分类有什么不同？

    答：特征方面（情感词典，融合了情感的词向量）。

2. 怎么构建情感词典？

    答：可以先人工制作一个小型的情感词典，然后利用语料，根据一些启发式规则（如“and”连接的词情感一样，“but”相反）来获得新的情感词，不断迭代扩展词典大小。

3. 情感词典会有哪些问题？

    答：不同的领域的情感词可能不一样；表达情感的词不一定是情感词；讽刺语境。

4. 如何检测不文明词汇？

    答：词典，分类问题。

#### Multilinguality

1. Modeling

    答：
    - 数据
		- 重点针对新语言
	    - 考虑可信赖的数据集而非从头开始收集
	    - 采样
		    - 平衡：参考 XLM-R，用 temperature-based sampling，即 $$q = p ^ {\alpha} / sum$$。当 $$\alpha$$ 为 1 时，表示没有变化；当 $$\alpha$$ 为 0 时，表示均匀分布；当 $$\alpha$$ 大于 1 时，分布尖锐，对高资源语言有优势；当 $$\alpha$$ 小于 1 时，分布变平，对低资源语言有优势
		    - 避免遗忘：对已覆盖语言采取最少采样
    - 模型训练
	    - 词表扩展（新 embedding 学习/推理速度/OOV）
	    - continued pretraining（LoRA）
	    - 训练环境和框架
    - 模型评估
	    - PPL，NLU and NLG（measures）
		- high-resource languages help low-resource languages

2. Adaptation

    答：Cross-lingual RAG vs MT（对于高资源，可能是先翻译再做 RAG；对于低资源，训练一个 MT 系统比检索系统要难；跨文化翻译行不通；取决 English-centric 还是多语言模型）
    
    Contrastive Loss（一对样本）vs Triplet Loss（三元组）vs InfoNCE（一个正样本 + 多个负样本，还有温度参数）

### Others

1. 性格测试: [链接 1](https://www.zhihu.com/question/28728468/answer/41961812)

2. 自我介绍

3. 请用三个词形容自己

4. 每天的生活怎么安排

5. 生活和工作上遇到的问题

6. 优势和劣势

7. 工作中遇到的挑战及应对方案

8. 团队合作

9. 为什么加入我们公司

10. 有其他 offer 时怎么选择

11. 职业规划（工作方向）: [链接](https://www.zhihu.com/question/20054953)

12. 个人工作内容 & 部门工作内容（业务，技术栈）/团队规模/团队资源（GPU）

13. 工作地点：城市，具体位置，远程办公

14. 薪资（期望薪资，最低接收工资，固定几薪/绩效浮动，Base） & 定级 & 绩效考核 & 晋升机制 & 转正: [链接 1](https://www.zhihu.com/question/19841590)，[链接 2](https://www.zhihu.com/question/34557602)

15. 工作时间：日常工作时间，单双休，年假