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

typedef struct {
    int j;
    int f;
    double ratio;
} room;

int main() {
    int m, n;
    room *rooms;
    while(scanf("%d %d", &m, &n) == 2 && m != -1 && n != -1) {
        rooms = (room *)malloc(sizeof(room) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d %d", &rooms[i].j, &rooms[i].f);
            rooms[i].ratio = (double)rooms[i].j / rooms[i].f;
        }
        for(int i1 = 0; i1 < n - 1; i1++) {
            for(int i2 = 0; i2 < n - i1 - 1; i2++) {
                if(rooms[i2].ratio < rooms[i2 + 1].ratio) {
                    room temp = rooms[i2];
                    rooms[i2] = rooms[i2 + 1];
                    rooms[i2 + 1] = temp;
                }
            }
        }
        double sum = 0;
        for(int i = 0; i < n; i++) {
            if(m <= 0) {
                break;
            } else if(m > rooms[i].f) {
                m -= rooms[i].f;
                sum += rooms[i].j;
            } else {
                sum += m * rooms[i].ratio;
                break;
            }
        }
        printf("%.3lf\n", sum);
    }
    return 0;
}
```
**移动桌子**

***题目来源***

[HDOJ 1050 Moving Tables](http://acm.hdu.edu.cn/showproblem.php?pid=1050)

***题目分析***

这是很经典的贪心算法问题，我们可以设一个数组存储桌子从每个门口经过的次数，最后遍历数组，记录最大经过次数，即可得到最少时间。

***实现代码***

```c
#include<stdio.h>
#include<string.h>
#define N 200

int main() {
    int t;
    int s[N];
    scanf("%d", &t);
    while(t--) {
        memset(s, 0, sizeof(s));
        int n;
        scanf("%d", &n);
        while(n--) {
            int a, b;
            scanf("%d%d", &a, &b);
            if(a > b) {
                int t = a;
                a = b;
                b = t;
            }
            for(int i = (a - 1) / 2; i <= (b - 1) / 2; i++) {
                s[i]++;
            }
        }
        int max = 0;
        for(int i = 0; i < N; i++) {
            if(s[i] > max) {
                max = s[i];
            }
        }
        printf("%d\n", max * 10);
    }
    return 0;
}
```

**木棍**

***题目来源***

[HDOJ 1051 Wooden Sticks](http://acm.hdu.edu.cn/showproblem.php?pid=1051)

***题目分析***

简单的贪心算法题，对结构体进行排序后求出LIS的个数。

***实现代码***

```c++
#include<stdio.h>
#include<algorithm>
#define N 5000
using namespace std;

struct stick {
    int l;
    int w;
    int flag;
};

int t, n;
stick s[N];

bool cmp(stick x, stick y) {
    if(x.l == y.l)
        return x.w < y.w;
    else
        return x.l < y.l;
}

int main() {
    scanf("%d", &t);
    while(t--) {
        scanf("%d", &n);
        for(int i = 0; i < n; i++) {
            scanf("%d%d", &s[i].l, &s[i].w);
            s[i].flag = 0;
        }
        sort(s, s + n, cmp);
        int result = 0;
        int isFound, pos;
        while(1) {
            isFound = 0;
            for(int i = 0; i < n; i++) {
                if(s[i].flag == 0) {
                    isFound = 1;
                    pos = i;
                    result++;
                    s[i].flag = 1;
                    break;
                }
            }
            if(isFound) {
                for(int i = pos + 1; i < n; i++) {
                    if(s[i].w >= s[pos].w && s[i].flag == 0) {
                        pos = i;
                        s[i].flag = 1;
                    }
                }
            } else {
                break;
            }
        }
        printf("%d\n", result);
    }
    return 0;
}
```

## 后记

贪心算法与动态规划一样重要而且有趣，一定要很好的掌握。
