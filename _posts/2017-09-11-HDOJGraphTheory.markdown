---
layout:     post
title:      "杭电OJ刷题记之图论"
subtitle:   "算法中的离散数学"
date:       2017-09-11 01:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 体验图论的魅力
>


## 前言

离散数学在算法中的运用相当广泛，其中就包括极为重要的图论知识，这篇文章，我们就来看看杭电OJ上面的图论问题。

---

## 正文

**男孩和女孩**

***题目来源***

[HDOJ 1068 Girls and Boys](http://acm.hdu.edu.cn/showproblem.php?pid=1068)

***题目分析***

这是我做的第一道二分图最大匹配问题，所以也是查了很多的资料。关于二分图最大匹配问题，我们常用的解决方法是匈牙利算法，具体可以通过[《二分图的最大匹配、完美匹配和匈牙利算法》](https://www.renfei.org/blog/bipartite-matching.html)这篇文章进行了解，这类题的[模板](http://www.cnblogs.com/kuangbin/archive/2011/08/09/2132828.html)，kuangbin大神也已经给出。

对于这道题，实际上要求的是最大独立集，所以我们可以通过`二分图最大独立集 = 顶点数 - 二分图最大匹配`这条公式来进行计算，但由于输入建图是双向的，所以最大匹配要取一半。由于是第一次做这种题，所以我也写了较多的注释。

***实现代码***

```c++
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 500

int n;      // 二分图两个集合点数
int g[N][N];    // 采用邻接矩阵记录边
int linker[N];      // 记录右边匹配顶点及匹配边
int used[N];

int dfs(int u) {    // 寻找增广路
    for(int v = 0; v < n; v++) {    // 遍历右侧顶点
        if(g[u][v] && !used[v]) {   // 如果存在边且右边顶点还没用过
            used[v] = 1;
            if(linker[v] == -1 || dfs(linker[v])) {     // 如果能找到非匹配点
                linker[v] = u;
                return 1;
            }
        }
    }
    return 0;
}

int hungary() {
    int res = 0;    // 最大匹配数
    memset(linker, -1, sizeof(linker));
    for(int u = 0; u < n; u++) {    // 从左边第1个顶点开始，寻找增广路
        memset(used, 0, sizeof(used));
        if(dfs(u)) {  // 如果找到增广路，则匹配数加一
            res++;
        }
    }
    return res;
}

int main() {
    int u, v, num;
    while(scanf("%d", &n) == 1) {
        memset(g, 0, sizeof(g));
        for(int i = 0; i < n; i++) {
            scanf("%d: (%d)", &u, &num);
            for(int j = 0; j < num; j++) {
                scanf("%d", &v);
                g[u][v] = 1;
            }
        }
        int result = hungary();
        printf("%d\n", n - result / 2);
    }
    return 0;
}
```
## 后记

这些题目虽然大部分都是模板题，但是理解其原理对我们的学习有很大帮助，所以不可忽视。
