#### 递推法

##### 简单递推

思路：扩展递推式，将其转换为一个和式，然后计算该和式。

例 1：$$T(n) = 2T(n-1) + 4$$

解答：https://stackoverflow.com/questions/22741590/what-is-the-time-complexity-of-the-recurrence-tn-2tn-1-4

例 2：$$T(n) = 2T(n-1) + n$$

解答：https://math.stackexchange.com/questions/239974/solve-the-recurrence-tn-2tn-1-n

##### 递推树辅助递推

思路：将递推式表示为一棵递归树，然后计算所有结点上表示的数之和。

例 3：$$T(n) = 2T(n-1) + \log_{}n$$

解答：https://cs.stackexchange.com/questions/57424/solving-tn-2tn-2-log-n-with-the-recurrence-tree-method?newreg=13ed0f2ed91f46d8b69559031a3a6cbf

例 4：$$T(n) = T(n/3) + T(2n/3) + cn$$

解答：https://math.stackexchange.com/questions/1112012/recursion-tree-tn-tn-3-t2n-3-cn

n替换法(猜测加归纳证明)

n换元

n生成函数法

n特征方程法

nK阶常系数齐次递归方程

nK阶常系数非齐次递归方程

n通用递归方程法

n积分法