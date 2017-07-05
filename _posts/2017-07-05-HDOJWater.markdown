---
layout:     post
title:      "杭电OJ刷题记之水题"
subtitle:   "特殊的一类算法题"
date:       2017-07-05 08:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 每道题都不能被放过
>


## 前言

在杭电OJ中有一个分类，叫做水题，主要包括一些难以分类的杂题，在这里面也有很多质量极高的题目，今天就让我们一起来做一做。

---

## 正文

**数字序列**

***题目来源***

[HDOJ 1005 Number Sequence](http://acm.hdu.edu.cn/showproblem.php?pid=1005)

***题目分析***

对于这道题，我一开始的做法是根据进行n次迭代得到最后的结果，可是提交程序的结果是TLE。于是我谷歌一查才发现，原来这是典型的找规律题，从迭代函数`f(n) = (A * f(n - 1) + B * f(n - 2)) mod 7`可以看出，f(n-1)的取值可能有7种，f(n-2)也有7种，故f(n-1)f(n-2)的组合可能有49种，于是可以得到，在49次迭代内f(n)必有规律可寻，这便是此题的核心思路。值得注意的是，规律不一定是从1 1开始，因为序列可能是1 1 2 3 2 3……，因此我们在每次迭代之后必须从第一个数遍历到当前迭代数，来得到规律开始的地方及规律的周期。

***实现代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int t;
    scanf("%d", &t);
    for(int i = 0; i < t; i++) {
        int n;
        int *s, *sum, max;
        int a, b, A, B;
        scanf("%d", &n);
        s = (int *)malloc(sizeof(int) * n);
        sum = (int *)malloc(sizeof(int) * n);
        for(int j = 0; j < n; j++) {
            scanf("%d", &s[j]);
            sum[j] = 0;
        }
        if(i != 0) {
            printf("\n");
        }
        max = sum[0] = s[0];
        A = B = a = b = 0;
        for(int j = 1; j < n; j++) {
            if(sum[j - 1] + s[j] >= s[j]) {
                sum[j] = sum[j - 1] + s[j];
                b++;
            } else {
                sum[j] = s[j];
                a = b = j;
            }
            if(sum[j] > max) {
                max = sum[j];
                A = a;
                B = b;
            }
        }
        printf("Case %d:\n%d %d %d\n", i+1, max, A + 1, B + 1);
    }
    return 0;
}
```
## 后记

水题也有着极高的价值，千万不能忽视。
