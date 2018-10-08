---
layout: post
title: 01多背包问题
categories: Algorithm
description: 01多背包问题
keywords: 数据结构与算法, 算法, 背包问题, 动态规划, 01多背包问题
---

之前笔者根据《背包九讲》做了一些笔记，具体如下：

- [《背包九讲》笔记（一）—— 01 背包问题](https://lpq29743.github.io/redant/algorithm/2017/08/21/Pack1/)
- [《背包九讲》笔记（二）——完全背包问题](https://lpq29743.github.io/redant/algorithm/2017/08/22/Pack2/)
- [《背包九讲》笔记（三）——物品冲突问题](https://lpq29743.github.io/redant/algorithm/2017/08/25/Pack3/)

当最近笔者遇到了一道新的题目，它来源于[深信服的秋招面试题第 4 题](https://www.nowcoder.com/discuss/116594?type=0&order=0&pos=22&page=1)。这道题实际上可以算是 01 多背包问题，而这个问题在《背包九讲》中没有被提及，因此笔者参考 [Knapsack Problem](http://www.or.deis.unibo.it/knapsack.html) Chapter 6 对这个问题进行了深入理解。

### 题目

有 N 件物品和 M 个容量为 V 的背包。第 i 件物品的耗费的空间是 Ci ，价值是 Wi 。求解将哪些物品装入背包可使背包价值总和最大。

### 题目分析



### 实现代码

```c
for k = 1 to K
	for v = V to 0
		for item i in group k
			F[v] = max{F[v], F[v − Ci] + Wi}
```