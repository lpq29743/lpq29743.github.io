---
layout: post
title: Fibonacci 数列问题
categories: Algorithm
description: Fibonacci数列问题
keywords: 算法, Fibonacci, 斐波那契数列
---

Fibonacci 数列在算法中十分常见，其数列形式如下：

> 1, 1, 2, 3, 5, 8, 13, 21, 34, ......

其递归公式可以表示为：
$$
F(n)=F(n-1)+F(n-2) \\
F(0)=0 \\
F(1)=1
$$
这篇文章主要围绕它的几种解决方法展开。

**递归法**

```python
def fibonacci(n):
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == '__main__':
    print(fibonacci(10))
```

递归法思路简单，但有一定的时间上和空间上的消耗，存在重复计算，容易栈溢出。

**递推法**

```python
def fibonacci(n):
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == '__main__':
    print(fibonacci(10))
```

递推法与递归法相反，它是一种自底向上的方法，减少了重复的计算，时间复杂度为 $$O(n)$$。

**矩阵递推法**

Fibonacci 数列可表示为以下矩阵：
$$
\left[
\begin{matrix}
Fib(n+1) \\
Fib(n)
\end{matrix}
\right]
=
\left[
\begin{matrix}
1 & 1 \\
1 & 0
\end{matrix}
\right]
\left[
\begin{matrix}
Fib(n) \\
Fib(n-1)
\end{matrix}
\right]
$$
即：
$$
\left[
\begin{matrix}
Fib(n+1) \\
Fib(n)
\end{matrix}
\right]
=
\left[
\begin{matrix}
1 & 1 \\
1 & 0
\end{matrix}
\right]
^n
\left[
\begin{matrix}
Fib(1) \\
Fib(0)
\end{matrix}
\right]
$$
因此，问题可以转换为求矩阵的 $$n$$ 次方，这里可以采用[快速幂方法](https://baike.baidu.com/item/%E5%BF%AB%E9%80%9F%E5%B9%82)。举个例子，如果求 $$2^{20}$$，由于 $$2^{20}=2^{16}*2^4$$，而 $$2^2$$ 可以通过 $$2^1 \times 2^1$$ 来求，$$2^4$$ 可以通过 $$2^2 \times 2^2$$ 来求。以此类推，通过这种方法，我们可以以时间复杂度 $$O(logn)$$ 求解 Fibonacci 问题。

矩阵递推法是一种有趣的思路，可以推广到其他问题上，实际上，对于 Fibonacci 数列，我们用上面的递推法基本就可以满足我们的需求了。

**通项法**

我们可以用各种方法求解 Fibonacci 数列的通项公式，包括构造等比数列法、线性代数法、特征方程法和母函数法，最终可以求得 Fibonacci 数列的通项公式为：
$$
F(n)=\frac{1}{\sqrt{5}} \times ([\frac{1+\sqrt{5}}{2}]^n - [\frac{1-\sqrt{5}}{2}]^n)
$$
通过通项公式，我们可以直接以 $$O(1)$$ 的时间复杂度解决 Fibonacci 问题。