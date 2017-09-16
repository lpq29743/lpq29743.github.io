---
layout:     post
title:      "哈工大复试上机题"
subtitle:   "迈出成功的第一步"
date:       2017-09-13 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 相信自己
>


## 前言

之前已经刷过北航和北邮的复试题，这次来刷刷哈工大的。

---

## 正文

**字符串内排序**

***题目描述***

输入一个字符串，长度小于等于200，然后将输出按字符顺序升序排序后的字符串

***输入描述***

测试数据有多组，输入字符串

***输出描述***

对于每组输入，输出处理后的结果

***输入例子***

```
bacd
```

***输出例子***

```
abcd
```

***程序代码***

```c++
#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;

int main() {
    char s[205];
    while(scanf("%s", s) == 1) {
        sort(s, s + strlen(s));
        printf("%s\n", s);
    }
    return 0;
}
```
**数组逆置**

***题目描述***

输入一个字符串，长度小于等于200，然后将数组逆置输出

***输入描述***

测试数据有多组，每组输入一个字符串

***输出描述***

对于每组输入，请输出逆置后的结果

***输入例子***

```
hdssg
```

***输出例子***

```
gssdh
```

***程序代码***

```c
#include<stdio.h>
#include<string.h>

int main() {
    char s[205];
    while(scanf("%s", s) == 1) {
        for(int i = 0; i <= (strlen(s) - 1) / 2; i++) {
            char c = s[i];
            s[i] = s[strlen(s) - i - 1];
            s[strlen(s) - i - 1] = c;
        }
        printf("%s\n", s);
    }
    return 0;
}
```

**找x**

***题目描述***

输入一个数n，然后输入n个数值各不相同，再输入一个值x，输出这个值在这个数组中的下标（从0开始，若不在数组中则输出-1）

***输入描述***

测试数据有多组，输入n(1<=n<=200)，接着输入n个数，然后输入x

***输出描述***

对于每组输入，请输出结果

***输入例子***

```
2
1 3
0
```

***输出例子***

```
-1
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int n, x;
    int s[201];
    while(scanf("%d", &n) == 1) {
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        scanf("%d", &x);
        int ans = -1;
        for(int i = 0; i < n; i++) {
            if(s[i] == x) {
                ans = i;
                break;
            }
        }
        printf("%d\n", ans);
    }
    return 0;
}
```

**最大公约数**

***题目描述***

输入两个正整数，求其最大公约数

***输入描述***

测试数据有多组，每组输入两个正整数

***输出描述***

对于每组输入，请输出其最大公约数

***输入例子***

```
49 14
```

***输出例子***

```
7
```

***程序代码***

```c
#include<stdio.h>

int gcd(int a, int b) {
    int t;
    while(b) {
        t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int main() {
    int a, b;
    while(scanf("%d %d", &a, &b) == 2) {
        printf("%d\n", gcd(a, b));
    }
    return 0;
}
```

**百万富翁问题**

***题目描述***

一个百万富翁遇到一个陌生人，陌生人找他谈了一个换钱的计划。该计划如下：我每天给你10 万元，你第一天给我1 分钱，第二天2 分钱，第三天4 分钱……
这样交换 30 天后，百万富翁交出了多少钱？陌生人交出了多少钱？（注意一个是万元，一个是分）

***输入描述***

该题没有输入

***输出描述***

输出两个整数，分别代表百万富翁交出的钱和陌生人交出的钱，富翁交出的钱以万元作单位，陌生人交出的钱以分作单位

***程序代码***

```c
#include<stdio.h>
#include<math.h>

int main() {
    printf("%d %d\n", 10 * 30, (int)pow(2, 30) - 1);
    return 0;
}
```

**完数**

***题目描述***

求1-n内的完数，所谓的完数是这样的数，它的所有因子相加等于它自身，比如6有3个因子1,2,3,1+2+3=6，那么6是完数。即完数是等于其所有因子相加和的数

***输入描述***

测试数据有多组，输入n，n数据范围不大

***输出描述***

对于每组输入,请输出1-n内所有的完数。如有案例输出有多个数字，用空格隔开，输出最后不要有多余的空格

***输入例子***

```
6
```

***输出例子***

```
6
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int isPerfectNum(int n) {
    int sum = 0;
    for(int i = 1; i <= sqrt(n); i++) {
        if(n % i == 0) {
            if(sqrt(n) == i) {
                sum += i;
            } else {
                sum += i + n / i;
            }
        }
    }
    return sum == 2 * n;
}

int main() {
    int n, *s, count;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        count = 0;
        for(int i = 1; i <= n; i++) {
            if(isPerfectNum(i)) {
                s[count++] = i;
            }
        }
        for(int i = 0; i < count; i++) {
            if(i != 0) {
                printf(" ");
            }
            printf("%d", s[i]);
        }
        printf("\n");
    }
    return 0;
}
```

## 后记

这几天来，收到的消息喜忧参半，心情也比较复杂，希望能够收拾好行囊，继续朝实现梦想的道路前进。
