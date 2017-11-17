---
layout:     post
title:      "快速排序法及其优化"
subtitle:   "认识并掌握最常用的排序算法"
date:       2016-12-04 18:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 快排之旅


## 前言

之前讲过排序算法的比较，并且还提及冒泡排序法的优化。今天我们要讲的是冒泡法的进阶算法，也就是我们经常提及的快速排序法。

---

## 正文

首先给出快速排序法的基本实现：

```c
// 快排算法
#include <stdio.h>
#include <stdlib.h>

int n;
int *s;
int data_input = 1;
int myarray[10] = {3, 0, 4, 2, 7, 1, 5, 6, 8, 7};

void quicksort(int *a, int left, int right) {
    if(left >= right) {
        return ;
    }
    int i = left;
    int j = right;
    int key = a[left];

    while(i < j) {
        while(i < j && key <= a[j]) {
            j--;
        }
        a[i] = a[j];
        while(i < j && key >= a[i]) {
            i++;
        }
        a[j] = a[i];
    }

    a[i] = key;
    quicksort(a, left, i - 1);
    quicksort(a, i + 1, right);
}

void show(int *arr) {
    int i;
    for(i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int i;
    if(data_input == 0) {
        printf("Please input the number of data:\n");
        scanf("%d", &n);
        s = (int *)malloc(sizeof(int) * n);
        printf("Please input the data:\n");
        for(i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        quicksort(s, 0, n - 1);
        show(s);
    } else {
        n = sizeof(myarray) / sizeof(int);
        quicksort(myarray, 0, n - 1);
        show(myarray);
    }
}
```

快速排序法主要是以下三个步骤：

1. 选取基准
2. 分割操作
3. 递归排序

而至于快排的优化可以从选取基准出发。默认的选取基准的方法是以第一个元素为基准，则对于有序数组是一个很不好的选择，因为每次分割之后只能将原数组长度减一，针对这一问题的可用方法之一是随机选取基准，但更常用的是三数取中法，也就是选取第一个数、中间的数、最后一个数中第二大的作为基准进行数组分割。代码如下：

```c
// 快排算法
#include <stdio.h>
#include <stdlib.h>

int n;
int *s;
int data_input = 1;
int myarray[10] = {3, 0, 4, 2, 7, 1, 5, 6, 8, 7};

void quicksort(int *a, int left, int right) {
    if(left >= right) {
        return ;
    }
    int i = left;
    int j = right;
    int k = left + right / 2;

    // 先将最大的值赋给最后一个数，再将第二大的值赋给第一个值
    if(a[i] > a[j]) {
        a[i] = a[i] ^ a[j];
        a[j] = a[j] ^ a[i];
        a[i] = a[i] ^ a[j];
    }
    if(a[k] > a[j]) {
        a[k] = a[k] ^ a[j];
        a[j] = a[j] ^ a[k];
        a[k] = a[k] ^ a[j];
    }
    if(a[k] > a[i]) {
        a[k] = a[k] ^ a[i];
        a[i] = a[i] ^ a[k];
        a[k] = a[k] ^ a[i];
    }

    int key = a[left];

    while(i < j) {
        while(i < j && key <= a[j]) {
            j--;
        }
        a[i] = a[j];
        while(i < j && key >= a[i]) {
            i++;
        }
        a[j] = a[i];
    }

    a[i] = key;
    quicksort(a, left, i - 1);
    quicksort(a, i + 1, right);
}

void show(int *arr) {
    int i;
    for(i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int i;
    if(data_input == 0) {
        printf("Please input the number of data:\n");
        scanf("%d", &n);
        s = (int *)malloc(sizeof(int) * n);
        printf("Please input the data:\n");
        for(i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        quicksort(s, 0, n - 1);
        show(s);
    } else {
        n = sizeof(myarray) / sizeof(int);
        quicksort(myarray, 0, n - 1);
        show(myarray);
    }
}
```

## 后记

快速排序法是排序算法中十分重要也很常用的算法之一，关于它的优化就有很多的资料，有兴趣的朋友可以看看[这里](https://www.zhihu.com/question/39214230/answer/80380554)。
