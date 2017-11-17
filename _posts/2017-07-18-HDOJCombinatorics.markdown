---
layout:     post
title:      "杭电OJ刷题记之组合数学篇"
subtitle:   "数学和计算机又一次完美的结合"
date:       2017-07-18 10:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 陪伴我们成长的排列和组合
>


## 前言

组合数学可以说是数学学科里面比较重要的一个分支，在计算机领域也是运用相当广泛，这篇文章就让我们通过HDOJ的题目一起来看一下两者的亲密结合。

---

## 正文

**伊格和公主（二）**

***题目来源***

[HDOJ 1027 Ignatius and the Princess II](http://acm.hdu.edu.cn/showproblem.php?pid=1027)

***题目分析***

这道题相对来说比较简单。如果用全排列的思维的话，那很可能超时，所以我们使用C++算法库下的next_permutation函数就可以解决问题了。值得注意的是，这个函数调用的次数应该为m-1次，因为初始的排列算是一次。

***实现代码***

```c++
#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;

int n, m;
int arr[1005];

int main() {
    while(scanf("%d%d", &n, &m) == 2) {
        memset(arr, 0, sizeof(arr));
        for(int i = 0; i < n; i++)
            arr[i] = i + 1;
        for(int i = 1; i < m; i++)
            next_permutation(arr, arr + n);
        for(int i = 0; i < n; i++) {
            if(i!=0) {
                printf(" ");
            }
            printf("%d", arr[i]);
        }
        printf("\n");
    }

}
```
## 后记

组合数学在数学中还算是蛮有趣的分支，与算法结合起来，让人更能感觉到数学的魅力。
