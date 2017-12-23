---
layout: post
title: Python 工具库之 Numpy
categories: Python
description: Python工具库之Numpy
keywords: Python, Python工具库, Numpy
---

Numpy 是 Python 的一个常用工具库，可以用来处理多维数组和矩阵运算。这篇文章就让我们一起来学习一下 Numpy。

先来看一些简单的操作：

```python
a = np.array([1, 2, 3])
# 输出数组：[1 2 3]
print(a)
# 输出数组维数：1
print(a.ndim)
# 输出数组维度：(3,)
print(a.shape)
# 输出数组元素个数：3
print(a.size)
# 输出元素类型：int32（因机器而异）
print(a.dtype)
# 输出元素所占字节大小：4
print(a.itemsize)
```

Numpy 还可以定义一些特殊的数组，具体如下：

```python
# 定义一个 1 * 5 的全 0 数组，默认类型为 float64
np.zeros(5)
# 定义一个 3 * 4 的全 1 数组，类型为 int32
np.ones(shape=(3, 4), dtype=np.int32)
# 定义一个元素从 0 到 11 的数组
np.arange(12)
```

Numpy 也可以执行数组之间的操作，具体如下：

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a * 2
# 输出数组 b；[[2, 4, 6], [8, 10, 12]]
print(b)
c = b - a
# 输出数组 c：[[1, 2, 3], [4, 5, 6]]
print(c)
# 输出数组 c 的第 2 列：[2, 5]
print(c[:, 1])
# 输出数组 c 的元素和：21
print(c.sum())
# 输出数组 c 的元素平均数：3.5
print(c.mean())
# 输出数组 c 的列平均数：[2.5, 3.5, 4.5]
print(c.mean(axis=0))
# 输出数组 c 的行平均数：[2., 5.]
print(c.mean(axis=1))
```

Numpy 可以对数组的形状、元素进行修改，具体如下：

```python
a = np.arange(6)
b = a.reshape(2, 3)
print(a)
# 输出数组 b：[[0, 1, 2], [3, 4, 5]]
print(b)
# 令数组从下标为 3 的元素开始都为 1
a[3:] = 1
# 输出数组 a：[0, 1, 2, 1, 1, 1]
print(a)
# 数组 b 也随之变化，输出为：[[0, 1, 2], [1, 1, 1]]
print(b)
print(b.dtype)
# 改变数组 b 的元素数据类型
b = b.astype('float64')
print(b.dtype)
```

Numpy 还可以直接从文件中读取数据，具体如下：

```python
npd = np.genfromtxt('data.csv', delimiter=',')
```