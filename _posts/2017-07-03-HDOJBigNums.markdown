---
layout:     post
title:      "杭电OJ刷题记之大数运算"
subtitle:   "做一个爱算法的程序员"
date:       2017-07-03 15:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 从0到无穷大
>


## 前言

为了备战面试，也为了增强自身的能力，决定按杭电OJ的[题目分类集](http://acm.hdu.edu.cn/typeclass.php)进行一段时间的刷题和学习，这篇文章主要整理的是杭电OJ中的大数运算题目。

---

## 正文

**大数相加**

***题目来源***

[HDOJ 1002 A + B Problem II](http://acm.hdu.edu.cn/showproblem.php?pid=1002)

***题目分析***

大数相加问题是比较常见的大数运算题目，也是最基础的大数运算，其具体的解决思路如下：

1. 计算出两个大数的长度，并记录最小长度
2. 初始化存储结果的字符数组，数组中每个元素的初始值均为0
3. 在最小长度内进行计算，分为进位和不进位两种情况
4. 根据长度较大的大数长出的部分进行计算，同样分为进位和不进位两种情况
5. 计算结果字符数组的长度，并倒序输出结果

***实现代码***

```c
#include<stdio.h>

int main() {
    int n;
    char a[1001], b[1001], c[1001];
    scanf("%d", &n);
    for(int i = 0; i < n; i++) {
        scanf("%s %s", &a, &b);
        int alength = 0, blength = 0;
        while(a[alength] != '\0') {
            alength++;
        }
        while(b[blength] != '\0') {
            blength++;
        }
        int length = alength;
        if(length > blength)
            length = blength;
        for(int i = 0; i < 1001; i++) {
            c[i] = 0;
        }
        for(int i = 0; i < length; i++) {
            int t1 = a[alength - 1 - i] - '0';
            int t2 = b[blength - 1 - i] - '0';
            if(t1 + t2 + c[i] > 9) {
                c[i] = (t1 + t2 + c[i]) % 10;
                c[i + 1] = 1;
            } else {
                c[i] = t1 + t2 + c[i];
            }
        }
        if(length < alength) {
            for(int i = length; i < alength; i++) {
                if(a[alength - 1 - i] - '0' + c[i] > 9) {
                    c[i] = (a[alength - 1 - i] - '0' + c[i]) % 10;
                    c[i + 1] = 1;
                } else {
                    c[i] = a[alength - 1 - i] - '0' + c[i];
                }
            }
        }
        if(length < blength) {
            for(int i = length; i < blength; i++) {
                if(b[blength - 1 - i] - '0' + c[i] > 9) {
                    c[i] = (b[blength - 1 - i] - '0' + c[i]) % 10;
                    c[i + 1] = 1;
                } else {
                    c[i] = b[blength - 1 - i] - '0' + c[i];
                }
            }
        }
        if(i != 0) {
            printf("\n");
        }
        printf("Case %d:\n%s + %s = ", i + 1, a, b);
        int clength = 1000;
        while(c[clength] == 0) {
            clength--;
        }
        for(int i = clength; i >= 0; i--) {
            printf("%d", c[i]);
        }
        printf("\n");
    }
    return 0;
}
```
## 后记

大数运算在实际编程中运用十分广泛，特别是对于没有大数处理功能的编程语言来说。学习好大数运算，对我们的编程工作有很大帮助。
