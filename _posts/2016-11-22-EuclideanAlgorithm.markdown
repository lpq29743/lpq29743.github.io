---
layout:     post
title:      "欧几里得算法求最大公约数和最小公倍数"
subtitle:   "感受欧几里得的光芒"
date:       2016-11-22 11:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 经典的东西总会留下来


## 前言

欧几里得算法是很多人入门程序、算法以及数学都会遇到的，本文简单讲解一下这一算法。

---

## 正文

**算法描述**

> 欧几里德算法（辗转求余法）原理： gcd(a,b)=gcd(b,a mod b)(a>b)，当b为0时，两数的最大公约数即为a。
>
> 最小公倍数=两数的乘积/*最大公约数
>

**算法实现**

```c
#include<stdio.h>
unsigned int Gcd(unsigned int M, unsigned int N) {
    unsigned int Rem;
    while(N > 0) {
        Rem = M % N;
        M = N;
        N = Rem;
    }
    return M;
}
int main(void) {
    int a, b, result;
    scanf("%d %d", &a, &b);
    result = Gcd(a, b);
    printf("最大公约数为%d\n", result);
    printf("最小公因数为%d\n", a * b / result);
    return 0;
}
```

## 后记

这个算法十分简单，但简单的东西往往藏着更多的知识点，有兴趣的同学可以去了解一下。