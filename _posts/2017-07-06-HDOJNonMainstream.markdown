---
layout:     post
title:      "杭电OJ刷题记之非主流类题目"
subtitle:   "少见却又典型的算法题"
date:       2017-07-06 16:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 算法中的非主流世界
>


## 前言

这篇文章的这些题目虽然被分类在非主流类型中，但实际上它们也很常见，就让我们一起来学习一下吧！

---

## 正文

**圆环套玩具游戏**

***题目来源***

[HDOJ 1007 Quoit Design](http://acm.hdu.edu.cn/showproblem.php?pid=1007)

***题目分析***

这道题本质上是一道最近点对问题。如果采用暴力求解的方法的话，会导致TLE，所以我们必须采用分治的方法解决。具体的做法是将所给平面上n个点的集合S分成两个子集S1和S2，每个子集中约有n/2个点，然后在每个子集中递归地求最接近的点对。递归退出的条件是集合中有2或3个点。

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

这一篇文章中有很多有趣的问题，可以慢慢琢磨。
