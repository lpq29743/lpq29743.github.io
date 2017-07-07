---
layout:     post
title:      "杭电OJ刷题记之搜索"
subtitle:   "最快的方式找到最好的结果"
date:       2017-07-07 14:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> The way to the answer
>


## 前言

搜索是程序常见的操作之一，经常会使用到递归等知识，这篇文章就让我们一起来学一下。

---

## 正文

**诱惑者的骨**

***题目来源***

[HDOJ 1010 Tempter of the Bone](http://acm.hdu.edu.cn/showproblem.php?pid=1010)

***题目分析***

这道题是常见的递归题，相对来说比较简单，但有两点必须注意：

1. 递归实际上是效率很低的做法，操作不当甚至可能导致程序崩溃，所以在某些情况下，我们要尽可能避免使用递归。本题由于相对简单，所以为了思路清晰，我们依旧使用递归。但为了加强递归的效率，我们必须做一些适当的剪枝，这也是算法题中经常出现的。下面的程序并没有做太多的剪枝，网上提供的其他版本程序提供的剪枝很多，比如奇偶剪枝，即当出发地与目的地之间的距离与所给的时间奇偶性不同的话，那么肯定无法走出迷宫
2. 这道题的另一个关键点是回溯法。由于题意要求，每走完一步，方格就不可走，所以我们在进行试探前需将方格设为不可走，试探后必须回溯，将方格设为可走

***实现代码***

```c++
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<algorithm>
using namespace std;

typedef struct {
    double x;
    double y;
} point;

int cmpxy (const point a, const point b) {
    if(a.x != b.x) {
        return a.x < b.x;
    } else {
        return a.y < b.y;
    }
}

double dist(point *p, int i, int j) {
    return sqrt((p[i].x - p[j].x) * (p[i].x - p[j].x) +
                (p[i].y - p[j].y) * (p[i].y - p[j].y));
}

double getMin(point *p, int low, int high) {
    if(low == high - 1) {
        return dist(p, low, high);
    } else if(low == high - 2) {
        double dist1, dist2, dist3, temp;
        dist1 = dist(p, low, low + 1);
        dist2 = dist(p, low, high);
        dist3 = dist(p, low + 1, high);
        temp = dist1 > dist2 ? dist2 : dist1;
        return temp > dist3 ? dist3 : temp;
    } else {
        double dist1, dist2;
        int mid = low + (high - low) / 2;
        dist1 = getMin(p, low, mid);
        dist2 = getMin(p, mid + 1, high);
        double mindist = dist1 > dist2 ? dist2 : dist1;
        for(int i = mid + 1; i <= high; i++) {
            if(p[i].x > (p[mid].x - mindist) && p[i].x < (p[mid].x + mindist)) {
                if(dist(p, i, mid) < mindist) {
                    mindist = dist(p, i, mid);
                }
            }
        }
        return mindist;
    }
}

int main() {
    int n;
    while(scanf("%d", &n) != 0 && n) {
        point *p = (point *)malloc(sizeof(point) * n);
        for(int i = 0; i < n; i++) {
            scanf("%lf %lf", &p[i].x, &p[i].y);
        }
        sort(p, p + n, cmpxy);
        int tag = 0;
        double eps = 1e-8;
        for(int i = 0; i < n - 1; i++) {
            if(fabs(p[i].x - p[i + 1].x) < eps && fabs(p[i].y - p[i + 1].y) < eps)
                tag = 1;
        }
        if(tag) {
            printf("0.00\n");
            continue;
        } else {
            printf("%.2lf\n", getMin(p, 0, n - 1) / 2);
        }
    }
    return 0;
}
```
## 后记

搜索经常涉及到递归、剪枝、回溯等等这些知识点，只要多加练习，我们才可以掌握好这类题目。
