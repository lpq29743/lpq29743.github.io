---
layout:     post
title:      "北邮复试上机题"
subtitle:   "寄出梦想的明信片"
date:       2017-07-03 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 付出就有回报
>


## 前言

之前做了一下北航的复试题目，现在来试一下北邮的题目。

---

## 正文

**二进制数**

***题目描述***

大家都知道，数据在计算机里中存储是以二进制的形式存储的。小明学了C语言后，想知道一个类型为unsigned int类型的数字，存储在计算机中的二进制串是什么样子的。你能帮帮小明吗？并且，小明不想要二进制串中前面的没有意义的0串，即要去掉前导0

***输入描述***

第一行是数字T（T<=1000），表示要求数字个数。接下来有T行，每行一个数字n（0<=n<=10^8），表示要求的二进制串

***输出描述***

输出共T行。每行输出求得的二进制串

***输入例子***

```
5
23
535
2624
56275
989835
```

***输出例子***

```
10111
1000010111
101001000000
1101101111010011
11110001101010001011
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int n;
    scanf("%d", &n);
    for(int i = 0; i < n; i++) {
        int source;
        char result[30];
        scanf("%d", &source);
        int loc = 0;
        while(source != 0) {
            result[loc++] = source % 2 + '0';
            source /= 2;
        }
        for(int j = loc - 1; j >= 0; j--) {
            printf("%d", result[j] - '0');
        }
        printf("\n");
    }
    return 0;
}
```
**哈夫曼树**

***题目描述***

哈夫曼树，第一行输入一个数n，表示叶结点的个数。需要用这些叶结点生成哈夫曼树，根据哈夫曼树的概念，这些结点有权值，即weight，题目需要输出所有结点的值与权值的乘积之和。

***输入描述***

输入有多组数据。每组第一行输入一个数n，接着输入n个叶节点（叶节点权值不超过100，2<=n<=1000）

***输出描述***

输出权值

***输入例子***

```
5  
1 2 2 5 9
```

***输出例子***

```
37
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        for(int i = 0; i < n - 1; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j] < s[j + 1]) {
                    int temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int wpl = 0;
        for(int i = 0; i < n - 1; i++) {
            int sum = s[n - i - 1] + s[n - i - 2];
            wpl = wpl + sum;
            s[n - i - 2] = sum;
            for(int j = n - i - 2; j > 0; j--) {
                if(s[j] > s[j - 1]) {
                    int temp = s[j];
                    s[j] = s[j - 1];
                    s[j - 1] = temp;
                } else {
                    break;
                }
            }
        }
        printf("%d\n", wpl);
    }
    return 0;
}
```

**比较奇偶数个数**

***题目描述***

第一行输入一个数，为n，第二行输入n个数，这n个数中，如果偶数比奇数多，输出NO，否则输出YES

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数（1<=n<=1000）

***输出描述***

如果偶数比奇数多，输出NO，否则输出YES。

***输入例子***

```
5
1 5 2 4 3
```

***输出例子***

```
YES
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        int count = 0;
        for(int i = 0; i < n; i++) {
            if(s[i] % 2 == 0) {
                count++;
            } else {
                count--;
            }
        }
        printf("%s\n", count > 0 ? "NO" : "YES");
    }
    return 0;
}
```

**查找第K小数**

***题目描述***

查找一个数组的第K小的数，注意同样大小算一样大。 如 2 1 3 4 5 2 第三小数为3

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数(1<=n<=1000)，再输入k

***输出描述***

输出第k小的整数

***输入例子***

```
6
2 1 3 5 2 2
3
```

***输出例子***

```
3
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n, k;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        scanf("%d", &k);
        for(int i = 0; i < n - 1; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j] > s[j + 1]) {
                    int temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int count = 0;
        for(int i = 0; i < n; i++) {
            while(i + 1 < n && s[i] == s[i + 1]) {
                i++;
            }
            count++;
            if(count == k) {
                printf("%d\n", s[i]);
            }
        }
    }
    return 0;
}
```

## 后记

继续前进，没有一滴汗水会白流。
