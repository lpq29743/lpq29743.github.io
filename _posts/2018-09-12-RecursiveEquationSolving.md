---
layout: post
title: 递归方程求解方法
categories: Algorithm
description: 递归方程求解方法
keywords: 算法, 递归
---

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

#### 替换法

思路：试扩展几个 n 比较小的递推式求值，发现规律，然后猜测并用数学归纳法证明。

#### 换元法

思路：对函数的定义域进行转换，并在新的定义域里，定义一个新的递归方程；把问题转换为对新的递归方程的求解；然后再把所得的解转换回原方程的解

#### 生成函数法

例 5：$$a_n = a_{n - 1} + a_{n-2}$$

解答：https://math.stackexchange.com/questions/371714/solving-recursive-sequence-using-generating-functions

#### 特征方程法

例 6：$$T(n) = T(n - 1) + T(n - 2)$$

解答：https://math.stackexchange.com/questions/2292707/fibonacci-recurrence-relations