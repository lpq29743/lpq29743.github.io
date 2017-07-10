---
layout:     post
title:      "杭电OJ刷题记之水题"
subtitle:   "特殊的一类算法题"
date:       2017-07-05 08:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 每道题都不能被放过
>


## 前言

在杭电OJ中有一个分类，叫做水题，主要包括一些难以分类的杂题，在这里面也有很多质量极高的题目，今天就让我们一起来做一做。

---

## 正文

**数字序列**

***题目来源***

[HDOJ 1005 Number Sequence](http://acm.hdu.edu.cn/showproblem.php?pid=1005)

***题目分析***

对于这道题，我一开始的做法是根据进行n次迭代得到最后的结果，可是提交程序的结果是TLE。于是我谷歌一查才发现，原来这是典型的找规律题，从迭代函数`f(n) = (A * f(n - 1) + B * f(n - 2)) mod 7`可以看出，f(n-1)的取值可能有7种，f(n-2)也有7种，故f(n-1)f(n-2)的组合可能有49种，于是可以得到，在49次迭代内f(n)必有规律可寻，这便是此题的核心思路。值得注意的是，规律不一定是从1 1开始，因为序列可能是1 1 2 3 2 3……，因此我们在每次迭代之后必须从第一个数遍历到当前迭代数，来得到规律开始的地方及规律的周期。

***实现代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int t;
    scanf("%d", &t);
    for(int i = 0; i < t; i++) {
        int n;
        int *s, *sum, max;
        int a, b, A, B;
        scanf("%d", &n);
        s = (int *)malloc(sizeof(int) * n);
        sum = (int *)malloc(sizeof(int) * n);
        for(int j = 0; j < n; j++) {
            scanf("%d", &s[j]);
            sum[j] = 0;
        }
        if(i != 0) {
            printf("\n");
        }
        max = sum[0] = s[0];
        A = B = a = b = 0;
        for(int j = 1; j < n; j++) {
            if(sum[j - 1] + s[j] >= s[j]) {
                sum[j] = sum[j - 1] + s[j];
                b++;
            } else {
                sum[j] = s[j];
                a = b = j;
            }
            if(sum[j] > max) {
                max = sum[j];
                A = a;
                B = b;
            }
        }
        printf("Case %d:\n%d %d %d\n", i+1, max, A + 1, B + 1);
    }
    return 0;
}
```
**数根**

***题目来源***

[HDOJ 1013 Digital Roots](http://acm.hdu.edu.cn/showproblem.php?pid=1013)

***题目分析***

这道题有两种做法：常规算法和九余数法。其中我在第一种做法上消耗了较长时间，主要是我分配的字符数组太小了。尽管早就知道不能直接用int变量接收输入的n，而要用字符串来表示，但我给出的长度为1000的数组还是满足不了题目的要求（在C++版本的程序中是可以的），最后数组的长度是10000。另一种方法则更为简洁和常用，在数根的[维基百科英文字条](https://en.wikipedia.org/wiki/Digital_root)中就有提及到，另外知乎上也有关于其证明的[讨论](https://www.zhihu.com/question/30972581)。九余数算法的核心公式就是`dr(n) = 1 + ((n - 1) mod 9)`，只要利用这个公式，我们就可以大大简化我们的程序。

***实现代码***

常规做法：

```c
#include<stdio.h>
#include<string.h>

int main() {
    char str[10000];
    int sum, i;
    while(scanf("%s", str)) {
        sum = 0;
        if(str[0] == '0') {
            break;
        }
        for(int i = 0; i < strlen(str); i++) {
            sum += str[i] - '0';
        }
        while(sum >= 10) {
            i = 0;
            while(sum > 0) {
                i += sum % 10;
                sum /= 10;
            }
            sum = i;
        }
        printf("%d\n", sum);
    }
    return 0;
}
```

九余数法：

```c
#include<stdio.h>
#include<string.h>

int main() {
    char str[10000];
    int sum;
    while(scanf("%s", str)) {
        sum = 0;
        if(str[0] == '0') {
            break;
        }
        for(int i = 0; i < strlen(str); i++) {
            sum += str[i] - '0';
        }
        printf("%d\n", 1 + (sum - 1) % 9);
    }
    return 0;
}
```

**随机数生成器**

***题目来源***

[HDOJ 1014 Uniform Generator](http://acm.hdu.edu.cn/showproblem.php?pid=1014)

***题目分析***

关于这道题，我一开始是直接根据题意进行解决的，具体的思路便是获取 x 为 0 - mod-1 时 seed 的数值，每次获取的时候与之前与获取到的数值进行比较，如果出现重复的数，则说明是一个 Bad Choice ，并退出获取操作。这个思路表面是可行的，可惜耗时太长，被OJ判了TLE。那么我们应该怎么解决这个问题呢？方法就是验证 step 和 mod 的最大公约数是不是1，如果是，则是Good Choice，否则是Bad Choice。这个方法可行的原因如下：seed(0)为0，第一次计算后结果为step，第二次为 2 \* step ，第三次是 3 \* step ，一直到 (k \* step) % mod ，如果此时 k < mod ，则不合题意，所以要满足Good Choice，则 step 和 mod 的最大公约数只能为1。从这道题我们也可以看出，编码前进行一定的分析能够提高我们解决问题的效率。

***实现代码***

```c
#include<stdio.h>
#include<stdlib.h>

int gcd(int m, int n) {
    int temp;
    while(n != 0) {
        temp = m % n;
        m = n;
        n = temp;
    }
    return m;
}

int main() {
    int step, mod;
    while(scanf("%d%d", &step, &mod) == 2) {
        int result = gcd(step, mod);
        printf("%10d%10d    ", step, mod);
        if(result == 1) {
            printf("Good Choice");
        } else {
            printf("Bad Choice");
        }
        printf("\n\n");
    }
    return 0;
}
```

**窃贼**

***题目来源***

[HDOJ 1015 Safecracker](http://acm.hdu.edu.cn/showproblem.php?pid=1015)

***题目分析***

这道题实质上考察的是深度优先搜索，为了满足题意要求，我们先对输入的字符串中的字符降序排列，这样子只要找到一个满足题意的字符串，我们就可以退出操作了。

***实现代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<algorithm>
using namespace std;

int target;
char s[12];
int temp[5];
int visited[12];
int flag;

int cmp(char a, char b) {
    return a > b;
}

int isMatch() {
    int k = temp[0] - (int)(pow(temp[1], 2) + 0.5) + (int)(pow(temp[2], 3) + 0.5)
    - (int)(pow(temp[3], 4) + 0.5) + (int)(pow(temp[4], 5) + 0.5);
    return k == target;
}

void dfs(int step) {
    if(step == 5) {
        if(isMatch()) {
            flag = 1;
        }
        return;
    } else {
        for(int i = 0; i < strlen(s); i++) {
            if(flag) {
                return;
            }
            if(visited[i]) {
                continue;
            }
            temp[step] = s[i] - 'A' + 1;
            visited[i] = 1;
            dfs(step + 1);
            visited[i] = 0;
        }
    }
}

int main() {
    while(scanf("%d%s", &target, s) == 2) {
        if(target == 0 && strcmp(s, "END") == 0) {
            break;
        }
        memset(temp, 0, sizeof(temp));
        memset(visited, 0, sizeof(visited));
        sort(s, s + strlen(s), cmp);
        flag = 0;
        dfs(0);
        if(flag) {
            for(int i = 0; i < 5; i++) {
                printf("%c", temp[i] - 1 + 'A');
            }
            printf("\n");
        } else {
            printf("no solution\n");
        }
    }
    return 0;
}
```

## 后记

水题也有着极高的价值，千万不能忽视。
