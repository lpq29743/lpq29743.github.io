---
layout:     post
title:      "北航复试上机题"
subtitle:   "让梦像飞鸟般翱翔"
date:       2017-06-12 22:45:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 不忘初心，方得始终
>


## 前言

最近被保研、考研以及夏令营等一系列事情弄得心烦意燥，这几天才慢慢平静下来。只要为梦想不断努力，对自己就问心无愧。不多聊这些废话了，下面进入正文！

---

## 正文

**迭代求立方根**

***题目描述***

立方根的逼近迭代方程是`y(n+1) = y(n)*2/3 + x/(3*y(n)*y(n))`，其中`y0=x`，求给定的x经过n次迭代后立方根的值

***输入描述***

输入有多组数据。每组一行，输入x n

***输出描述***

迭代n次后的立方根，double精度，保留小数点后面六位

***输入例子***

3000000 28

***输出例子***

144.224957

***程序代码***

```c
#include <stdio.h>

int main() {
    double x;
    int n;
    while(scanf("%lf %d", &x, &n) == 2) {
        double y = x;
        int count = 0;
        while(count < n) {
            y = y * 2 / 3 + x / (3 * y * y);
            count++;
        }
        printf("%.6lf\n", y);
    }
    return 0;
}
```
**素数**

***题目描述***

输入整数n(2<=n<=10000)，要求输出所有从1到这个整数之间(不包括1和这个整数)个位为1的素数，如果没有则输出-1

***输入描述***
输入有多组数据。每组一行，输入n

***输出描述***

输出所有从1到这个整数之间(不包括1和这个整数)个位为1的素数(素数之间用空格隔开，最后一个素数后面没有空格)，如果没有则输出-1

***输入例子***

100

***输出例子***

11 31 41 61 71

***程序代码***

```c
#include <stdio.h>

int isPrime(int i) {
    for(int j = 2; j < i; j++) {
        if(i % j == 0) {
            return 0;
        }
    }
    return 1;
}

int main() {
    int n;
    int hasFirst;
    while(scanf("%d", &n) == 1) {
        hasFirst = 0;
        for(int i = 2; i < n; i++) {
            if(isPrime(i) && i % 10 == 1) {
                if(hasFirst) {
                    printf(" ");
                }
                printf("%d", i);
                if(!hasFirst) {
                    hasFirst = 1;
                }
            }
        }
        if(!hasFirst) {
            printf("-1\n");
        } else {
            printf("\n");
        }
    }
    return 0;
}
```

## 后记

加油！梦会实现的！
