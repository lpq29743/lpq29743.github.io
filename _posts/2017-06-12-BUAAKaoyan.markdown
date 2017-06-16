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

```
3000000 28
```

***输出例子***

```
144.224957
```

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

```
100
```

***输出例子***

```
11 31 41 61 71
```

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

**旋转矩阵**

***题目描述***

任意输入两个9阶以下矩阵，要求判断第二个是否是第一个的旋转矩阵，如果是，输出旋转角度（0、90、180、270），如果不是，输出-1。 要求先输入矩阵阶数，然后输入两个矩阵，每行两个数之间可以用任意个空格分隔。行之间用回车分隔，两个矩阵间用任意的回车分隔

***输入描述***

输入有多组数据。每组数据第一行输入n(1<=n<=9)，从第二行开始输入两个n阶矩阵

***输出描述***

判断第二个是否是第一个的旋转矩阵，如果是，输出旋转角度（0、90、180、270），如果不是，输出-1。如果旋转角度的结果有多个，则输出最小的那个

***输入例子***

```
31 2 34 5 67 8 97 4 18 5 29 6 3
```

***输出例子***

```
90
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int isReverseBy0(int **s1, int **s2, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(s1[i][j] != s2[i][j]) {
                return 0;
                break;
            }
        }
    }
    return 1;
}

int isReverseBy90(int **s1, int **s2, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(s1[i][j] != s2[j][n - i - 1]) {
                return 0;
                break;
            }
        }
    }
    return 1;
}

int isReverseBy180(int **s1, int **s2, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(s1[i][j] != s2[n - i - 1][n - j - 1]) {
                return 0;
                break;
            }
        }
    }
    return 1;
}

int isReverseBy270(int **s1, int **s2, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(s1[i][j] != s2[n - j - 1][i]) {
                return 0;
                break;
            }
        }
    }
    return 1;
}

int main() {
    int n;
    while(scanf("%d", &n) == 1) {
        int **s1, **s2;
        s1 = (int **)malloc(sizeof(int*) * n);
        s2 = (int **)malloc(sizeof(int*) * n);
        for(int i = 0; i < n; i++) {
            s1[i] = (int *)malloc(sizeof(int) * n);
            s2[i] = (int *)malloc(sizeof(int) * n);
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                scanf("%d", &s1[i][j]);
            }
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                scanf("%d", &s2[i][j]);
            }
        }
        if(isReverseBy0(s1, s2, n)) {
            printf("0\n");
        } else if(isReverseBy90(s1, s2, n)) {
            printf("90\n");
        } else if(isReverseBy180(s1, s2, n)) {
            printf("180\n");
        } else if(isReverseBy270(s1, s2, n)) {
            printf("270\n");
        } else {
            printf("-1\n");
        }
    }
}
```

**数组排序**

***题目描述***

输入一个数组的值,求出各个值从小到大排序后的次序

***输入描述***

输入有多组数据。每组输入的第一个数为数组的长度n(1<=n<=10000)，后面的数为数组中的值，以空格分割

***输出描述***

各输入的值按从小到大排列的次序(最后一个数字后面没有空格)。

***输入例子***

```
4 -3 75 12 -3
```

***输出例子***

```
1 3 2 1
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

struct num {
    int value;
    int loc;
    int rank;
};

int main() {
    int n;
    while(scanf("%d", &n) == 1) {
        struct num *s;
        s = (struct num *)malloc(sizeof(struct num) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i].value);
            s[i].loc = i;
            s[i].rank = 0;
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j].value > s[j + 1].value) {
                    struct num temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int rank = 0;
        for(int i = 0; i < n; i++) {
            if(i - 1 < 0 || s[i - 1].value != s[i].value) {
                rank++;
            }
            s[i].rank = rank;
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j].loc > s[j + 1].loc) {
                    struct num temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int flag = 0;
        for(int i = 0; i < n; i++) {
            if(flag == 0) {
                printf("%d", s[i].rank);
            } else {
                printf(" %d", s[i].rank);
            }
            if(i == 0) {
                flag = 1;
            }
        }
        printf("\n");
    }
}
```

**字符串的查找删除**

***题目描述***

给定一个短字符串（不含空格），再给定若干字符串，在这些字符串中删除所含有的短字符串

***输入描述***

输入只有1组数据。输入一个短字符串（不含空格），再输入若干字符串直到文件结束为止

***输出描述***

删除输入的短字符串(不区分大小写)并去掉空格，输出

***输入例子***

```
in
include
int main()
{
printf(" Hi ");
}
```

***输出例子***

```
#clude
tma()
{
prtf("Hi");
}
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 100

char *deleteSubString(char *source, char *input, char *result) {
    int isSubString;
    int loc = 0;
    if(strlen(source) > strlen(input)) {
        return NULL;
    } else {
        for(int i = 0; i < strlen(input) - strlen(source) + 1; i++) {
            isSubString = 1;
            for(int j = 0; j < strlen(source); j++) {
                if(source[j] != input[i + j] && source[j] - 32 != input[i + j] && source[j] + 32 != input[i + j] ) {
                    isSubString = 0;
                    break;
                }
            }
            if(isSubString == 0 && input[i] != ' ') {
                result[loc++] = input[i];
            } else if(isSubString == 1) {
                i += strlen(source) - 1;
            }
        }
    }
    if(isSubString == 0) {
        for(int i = strlen(input) - strlen(source) + 1; i < strlen(input); i++) {
            result[loc++] = input[i];
        }
    }
    result[loc] = '\0';
    return result;
}

int main() {
    char source[N], input[N], result[N];
    gets(source);
    while(gets(input) != NULL) {
        printf("%s\n", deleteSubString(source, input, result));
    }
}
```

**字符串匹配**

***题目描述***

读入数据string[ ]，然后读入一个短字符串。要求查找string[ ]中和短字符串的所有匹配，输出行号、匹配字符串。匹配时不区分大小写，并且可以有一个用中括号表示的模式匹配。如“aa[123]bb”，就是说aa1bb、aa2bb、aa3bb都算匹配

***输入描述***

输入多组数据。每组数据第一行输入n(1<=n<=1000)，第二行开始输入n个字符串（不含空格），接下来输入匹配字符串

***输出描述***

输出匹配到的字符串的行号和该字符串（匹配时不区分大小写）

***输入例子***

```
4
Aab
a2B
ab
ABB
a[a2b]b
```

***输出例子***

```
1 Aab
2 a2B
4 ABB
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 100

int isSubString(char *s, char *pattern) {
    int loc = 0, isMatch;
    for(int i = 0; i < strlen(pattern); i++) {
        if(pattern[i] == '[') {
            isMatch = 0;
            while(pattern[++i] != ']') {
                if(s[loc] - 32 == pattern[i] || s[loc] + 32 == pattern[i] || s[loc] == pattern[i]) {
                    isMatch = 1;
                }
            }
            if(isMatch == 0) {
                return 0;
                break;
            } else {
                loc++;
            }
        } else {
            if(s[loc] - 32 != pattern[i] && s[loc] + 32 != pattern[i] && s[loc] != pattern[i]) {
                return 0;
                break;
            } else {
                loc++;
            }
        }
    }
    return 1;
}

int main() {
    int n;
    char **s, pattern[N];
    while(scanf("%d", &n) == 1) {
        getchar();
        s = (char **)malloc(sizeof(char *) * n);
        for(int i = 0; i < n; i++) {
            s[i] = (char *)malloc(sizeof(char) * N);
            gets(s[i]);
        }
        gets(pattern);
        for(int i = 0; i < n; i++) {
            if(isSubString(s[i], pattern)) {
                printf("%d %s\n", i + 1, s[i]);
            }
        }
    }
}
```

## 后记

加油！梦会实现的！
