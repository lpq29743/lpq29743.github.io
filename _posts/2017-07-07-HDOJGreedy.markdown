---
layout:     post
title:      "杭电OJ刷题记之贪心算法"
subtitle:   "从局部最优到整体最优"
date:       2017-07-07 13:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 走好每一小步，方能走好一大步
>


## 前言

动态规划和贪心算法可谓是最优解题目中的双子星，之前我们已经有一篇文章围绕动态规划进行展开，这篇文章就让我们来学习贪心算法。

---

## 正文

**肥鼠交易**

***题目来源***

[HDOJ 1009 FatMouse' Trade](http://acm.hdu.edu.cn/showproblem.php?pid=1009)

***题目分析***

这道题目属于很简单的贪心算法题目，能够帮助我们很好地理解贪心算法。得到换算数组之后，我们可以求出每一个房间换算的比率，然后按照比率从大到小的方式进行换算。

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

贪心算法与动态规划一样重要而且有趣，一定要很好的掌握。
