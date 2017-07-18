---
layout:     post
title:      "杭电OJ刷题记之演绎推理篇"
subtitle:   "隐藏在数字之下的规律"
date:       2017-07-12 22:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 事情总按一定的规律发展着
>


## 前言

这篇文章涉及到都是一些多多少少有点规律的题目。如果我们没有发现题目的规律，而是一味暴力求解，那无疑会碰壁。

---

## 正文

**出栈可能数**

***题目来源***

[HDOJ 1023 Train Problem II](http://acm.hdu.edu.cn/showproblem.php?pid=1023)

***题目分析***

这道问题一上来，可能有些算法初学者会打算通过栈和队列来模拟所有出栈可能，然后计算出所有的可能数。这种方法是可行的，但难度较大，而且计算量也会非常大，很容易做错。这个时候我们就要用到组合数学中的卡特兰数了。卡特兰数在算法题中运用十分广泛，如果还不熟悉的话，可以参考网上的资料或者是《编程之美》中的买票找零问题。对于卡特兰数，我们主要是记住两条公式，第一条是最后的解`h(n) = C(2n, n) / (n + 1)`，这在选择填空题中用的比较多，而另一条公式则在编程题中用的多，它也是用来解决这道题的公式，该公式是`h(n) = h(n - 1) * (4 * n - 2) / (n + 1)`。

对于这道题，由于计算到后面的数据会很大，所以我们在计算卡特兰数的时候还要用到大数运算的思维，题目主要用到的是大数乘法和大数除法。总体来说，这道题的质量还是相当的高的，能够大大锻炼我们的算法和编程能力。

***实现代码***

```c++
#include<stdio.h>
#define N 101

// 第n个Catalan数存在a[n]中，a[n][0]表示长度
// 数是倒着存的，输出时需倒序输出
int s[N][N];

void ktl() {
    int len;        // 上一个数的长度
    int t;        // 进位值
    s[1][0] = 1;
    s[1][1] = 1;
    s[2][0] = 1;
    s[2][1] = 2;
    len = 1;
    for(int i = 3; i < 101; i++) {
        t = 0;

        // 大数乘法

        // 在被乘数的长度范围内进行计算
        for(int j = 1; j <= len; j++) {
            int tmp = s[i - 1][j] * (4 * i - 2) + t;
            t = tmp / 10;
            s[i][j] = tmp % 10;
        }
        // 根据进位值添加长度并赋值
        while(t) {
            s[i][++len] = t % 10;
            t /= 10;
        }

        // 大数除法
        for(int j = len; j > 0; j--) {
            int tmp = s[i][j] + t * 10;
            s[i][j] = tmp / (i + 1);
            t = tmp % (i + 1);
        }
        while(!s[i][len]) {
            len--;
        }
        s[i][0] = len;
    }
}

int main() {
    ktl();
    int n;
    while(scanf("%d", &n) == 1) {
        for(int i = s[n][0]; i > 0; i--) {
            printf("%d", s[n][i]);
        }
        printf("\n");
    }
    return 0;
}
```
**三角波**

***题目来源***

[HDOJ 1030 Delta-wave](http://acm.hdu.edu.cn/showproblem.php?pid=1030)

***题目分析***

这是一道典型的规律题，找到的规律可以有很多种形式，这里提供两种：[简单易懂的](http://www.wutianqi.com/?p=2362)和[简洁晦涩的](http://blog.csdn.net/u014174811/article/details/41443177)。这里我们直接采用第二种的代码。

***实现代码***

```c++
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main() {
    int a, b;
    int aX, aY, bX, bY, aLayer, bLayer, step;
    while(scanf("%d%d", &a, &b) != EOF) {
        aLayer = ceil(sqrt((double)a)); //求出数a所在层
        bLayer = ceil(sqrt((double)b)); //求出数b所在层
        if(aLayer == bLayer) {
            printf("%d\n", abs(a - b));
        } else {
            aX = (aLayer * aLayer - a) / 2; //计算a的X坐标
            bX = (bLayer * bLayer - b) / 2; //计算b的X坐标
            aY = (a - (aLayer * aLayer - 2 * aLayer + 2)) / 2; //计算a的Y坐标
            bY = (b - (bLayer * bLayer - 2 * bLayer + 2)) / 2; //计算b的Y坐标
            step = abs(aX - bX) + abs(aY - bY) + abs(aLayer - bLayer);
            printf("%d\n", step); //求出最终步骤
        }
    }
}
```

## 后记

无论是简单的刷题，还是生活中的处事，一定要切记不能一味求快，要先分析好问题，然后再行动。这也让我想起了今天看到的美国海豹突击队的标语：慢则稳，稳则快。
