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

这道题本质上是一道最近点对问题。如果采用暴力求解的方法的话，会导致TLE，所以我们必须采用分治的方法解决。具体的做法是将所给平面上n个点的集合S分成两个子集S1和S2，每个子集中约有n/2个点，然后在每个子集中递归地求最接近的点对。在这里，一个关键的问题是如何实现分治法中的合并步骤，即由S1和S2的最接近点对，如何求得原集合S中的最接近点对。如果这两个点分别在S1和S2中，问题就变得复杂了。

对于这个关键点，我们处理的方式是先将集合进行排序，然后再分别求得两个集合中的最小距离d1和d2，设d1和d2的最小值为mindist。如果两个点分别存在两个不同的集合中，则两个点与中心点的横坐标之差和纵坐标之差均小于mindist（不可能是等于，如果是等于，其中一个集合可求出），且其中一个点必定是分界点，而另一个点则必定在没有包含分界点的集合里。利用这一点，我们可以在得到mindist后，在可能范围内进行遍历，这样便可以大大减少计算量。

如果对于上面的解释不太清楚的话，可以查看[这里](http://www.cnblogs.com/hxsyl/p/3230164.html)。对于这个问题，还有几个关键的地方，具体如下：

1. 程序排序用到的是C++的库文件algorithm里面的qsort算法，原本尝试过ANSI C的stdlib.h头文件中的sort算法，但效果不佳，会出现TLE。所以这个程序需要在C++环境下运行
2. 浮点数（float，double）是不存在完全相等的。我们可以用eps（一般为1e-6或1e-8），利用fabs（abs是整数取绝对值）判断范围是否小于eps，从而判断浮点数是否相等

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

这一篇文章中有很多有趣的问题，可以慢慢琢磨。
