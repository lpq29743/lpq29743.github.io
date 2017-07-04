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

**矩阵幂**

***题目描述***

给定一个n*n的矩阵，求该矩阵的k次幂，即P^k

***输入描述***

输入包含多组测试数据。数据的第一行为一个整数T(0<T<=10)，表示要求矩阵的个数。接下来有T组测试数据，每组数据格式如下： 第一行：两个整数n（2<=n<=10）、k（1<=k<=5），两个数字之间用一个空格隔开，含义如上所示。接下来有n行，每行n个正整数，其中，第i行第j个整数表示矩阵中第i行第j列的矩阵元素Pij且（0<=Pij<=10）。另外，数据保证最后结果不会超过10^8

***输出描述***

对于每组测试数据，输出其结果。格式为：n行n列个整数，每行数之间用空格隔开，注意，每行最后一个数后面不应该有多余的空格

***输入例子***

```
3
2 2
9 8
9 3
3 3
4 8 4
9 3 0
3 5 7
5 2
4 0 3 0 1
0 0 5 8 5
8 9 8 5 3
9 6 1 7 8
7 2 5 7 3
```

***输出例子***

```
153 96
108 81
1216 1248 708
1089 927 504
1161 1151 739
47 29 41 22 16
147 103 73 116 94
162 108 153 168 126
163 67 112 158 122
152 93 93 111 97
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int t;
    scanf("%d", &t);
    for(int i = 0; i < t; i++) {
        int n, k;
        int **s1, **s2, **s3;
        scanf("%d %d", &n, &k);
        s1 = (int **)malloc(sizeof(int *) * n);
        s2 = (int **)malloc(sizeof(int *) * n);
        s3 = (int **)malloc(sizeof(int *) * n);
        for(int j = 0; j < n; j++) {
            s1[j] = (int *)malloc(sizeof(int) * n);
            s2[j] = (int *)malloc(sizeof(int) * n);
            s3[j] = (int *)malloc(sizeof(int) * n);
        }
        for(int j1 = 0; j1 < n; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                scanf("%d", &s1[j1][j2]);
                s2[j1][j2] = s1[j1][j2];
                s3[j1][j2] = 0;
            }
        }
        for(int j1 = 1; j1 < k; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                for(int j3 = 0; j3 < n; j3++) {
                    int sum = 0;
                    for(int j4 = 0; j4 < n; j4++) {
                        sum += s1[j2][j4] * s2[j4][j3];
                    }
                    s3[j2][j3] = sum;
                }
            }
            for(int j2 = 0; j2 < n; j2++) {
                for(int j3 = 0; j3 < n; j3++) {
                    s2[j2][j3] = s3[j2][j3];
                }
            }
        }
        for(int j1 = 0; j1 < n; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                if(j2 != 0) {
                    printf(" ");
                }
                printf("%d", s2[j1][j2]);
            }
            printf("\n");
        }
    }
    return 0;
}
```

**C翻转**

***题目描述***

首先输入一个5 * 5的数组，然后输入一行，这一行有四个数，前两个代表操作类型，后两个数x y代表需操作数据为以x y为左上角的那几个数据。 操作类型有四种：  1 2 表示：90度，顺时针，翻转4个数  1 3 表示：90度，顺时针，翻转9个数  2 2 表示：90度，逆时针，翻转4个数  2 3 表示：90度，逆时针，翻转9个数 

***输入描述***

输入有多组数据。每组输入一个5 * 5的数组，然后输入一行，这一行有四个数，前两个代表操作类型，后两个数x y代表需操作数据为以x y为左上角的那几个数据

***输出描述***

输出翻转后的数组

***输入例子***

```
1 2 3 4 5
6 7 8 9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
1 3 1 1
```

***输出例子***

```
11 6 1 4 5
12 7 2 9 10
13 8 3 14 15
16 17 18 19 20
21 22 23 24 25
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int matrix[5][5];
    int a, b, x, y;
    while(1) {
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                if(scanf("%d", &matrix[i][j]) != 1) {
                    break;
                }
            }
        }
        if(scanf("%d %d %d %d", &a, &b, &x, &y) != 4) {
            break;
        }
        // 1 2 表示：90度，顺时针，翻转4个数
        if(a == 1 && b == 2) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x][y - 1];
            matrix[x][y - 1] = matrix[x][y];
            matrix[x][y] = matrix[x - 1][y];
            matrix[x - 1][y] = temp;
        }
        // 1 3 表示：90度，顺时针，翻转9个数
        if(a == 1 && b == 3) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x + 1][y - 1];
            matrix[x + 1][y - 1] = matrix[x + 1][y + 1];
            matrix[x + 1][y + 1] = matrix[x - 1][y + 1];
            matrix[x - 1][y + 1] = temp;
            temp = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y - 1];
            matrix[x][y - 1] = matrix[x + 1][y];
            matrix[x + 1][y] = matrix[x][y + 1];
            matrix[x][y + 1] = temp;
        }
        // 2 2 表示：90度，逆时针，翻转4个数
        if(a == 2 && b == 2) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y];
            matrix[x][y] = matrix[x][y - 1];
            matrix[x][y - 1] = temp;
        }
        // 2 3 表示：90度，逆时针，翻转9个数
        if(a == 2 && b == 3) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x - 1][y + 1];
            matrix[x - 1][y + 1] = matrix[x + 1][y + 1];
            matrix[x + 1][y + 1] = matrix[x + 1][y - 1];
            matrix[x + 1][y - 1] = temp;
            temp = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y + 1];
            matrix[x][y + 1] = matrix[x + 1][y];
            matrix[x + 1][y] = matrix[x][y - 1];
            matrix[x][y - 1] = temp;
        }
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                if(j != 0) {
                    printf(" ");
                }
                printf("%d", matrix[i][j]);
            }
            printf("\n");
        }
    }
    return 0;
}
```

## 后记

继续前进，没有一滴汗水会白流。
