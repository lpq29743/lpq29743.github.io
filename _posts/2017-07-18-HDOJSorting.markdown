---
layout:     post
title:      "杭电OJ刷题记之排序篇"
subtitle:   "算法中的老熟人"
date:       2017-07-18 16:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> How old are you
>


## 前言

排序在算法中算是比较基础的内容，也比较常见，这篇文章就让我们来盘点一下HDOJ里面的排序题。

---

## 正文

**伊格和公主（四）**

***题目来源***

[HDOJ 1029 Ignatius and the Princess IV](http://acm.hdu.edu.cn/showproblem.php?pid=1029)

***题目分析***

实际上，这道题分在排序题有点勉强，因为它可以不用排序解决。排序的方法指的是先对数组排序，出现(N + 1) / 2次的元素总会在排序后该序列的中间位置。但排序的做法相对开销比较大，这里我们采用绝对众数的方法，这在我不定时更新的博文[《算法的细枝末节》](https://lpq29743.github.io/redant/2017/04/04/AlgorithmDetail/)中有提到，这种方法复杂度为O(n)，而且容易掌握，推荐大家使用。

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

排序算是我们在算法中的老熟人了，所以一定要牢固掌握，别见到了写不出来哦！
