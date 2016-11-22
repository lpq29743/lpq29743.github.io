---
layout:     post
title:      "图的m着色问题"
subtitle:   "贪心法解决图的着色问题"
date:       2016-11-22 09:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
    - 图
    - 贪心算法
---

> 图论重要问题——着色问题


## 前言

最近开始准备考研，这篇文章是备考路上的第一篇文章，来源是廖明宏等编的《数据结构与算法》中第一章的交叉路口问题，从而让我接触到了着色问题。此文就针对着色问题进行讲述。

---

## 正文

#### 问题一

**提出问题**

图的m-着色判定问题——给定无向连通图G和m种不同的颜色。用这些颜色为图G的各顶点着色，每个顶点着一种颜色，是否有一种着色法使G中任意相邻的2个顶点着不同颜色?

**算法描述**

> 1.color[n]存储n个顶点的着色方案，可以选择的颜色为0到m-1，t=0。
>
> 2.对当前第t个顶点开始着色：
>
> 2.1.若t>=n，则已求得一个解，输出着色方案即可；
>
> 2.2.否则，依次对顶点t着色0-m-1，若t与所有其它相邻顶点无颜色冲突，则继续为下一顶点着色；否则，回溯，测试下一颜色。

**算法实现**

```c
//图着色问题回溯法
/*
无向图邻接矩阵示例
0 1 1 1
1 0 1 0
1 1 0 1
1 0 1 0
*/

#include <stdio.h>
#include <stdlib.h>

int n, m;
int *color;
int **graph;

int ok(int j) {
    int i;
    for(i = 0; i < j; i++)
        if(graph[j][i] == 1 && color[i] == color[j])
            return 0;
    return 1;
}

void graphcolor() {
    int i, j;

    //初始化
    for(i = 0; i < n; i++)
        color[i] = -1;
    j = 0;

    while(j >= 0) {
        color[j] = color[j] + 1;
        //找到一个满足的颜色
        while(color[j] < m) {
            if (ok(j))
                break;
            else
                color[j] = color[j] + 1; //搜索下一个颜色
        }
        if(color[j] < m && j == n - 1) { //求解完毕，输出解
            for(i = 0; i < n; i++)
                printf("%d ", color[i]);
            printf("\n");
        } else if(color[j] < m && j < n - 1) {
            j = j + 1; //处理下一个顶点
        } else {
            color[j] = -1;
            j = j - 1; //回溯
        }
    }

}

int main() {
    int i, j;

    printf("输入顶点数n和着色数m:\n");
    scanf("%d %d", &n, &m);

    color = (int *)malloc(sizeof(int) * n);
    graph = (int **)malloc(sizeof(int *) * n);
    for(i = 0; i < n; i++) {
        graph[i] = (int *)malloc(sizeof(int) * n);
    }

    printf("输入无向图的邻接矩阵:\n");
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("着色所有可能的解:\n");
    graphcolor();

    free(color);
    for(i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    return 0;
}
```

#### 问题二

**提出问题**

图的m着色优化问题——若一个图最少需要m种颜色才能使图中任意相邻的2个顶点着不同颜色，则称这个数m为该图的色数。求一个图的最小色数m的问题称为m-着色优化问题。

**算法描述**

> 1.1.color[n]存储n个顶点的着色方案，可以选择的颜色为0到m-1，t=0。
>
> 2.对当前第t个顶点开始着色：
>
> 2.1.若t>=n，则已求得一个解，退出循环，输出方案；
>
> 2.2.否则，依次对顶点t着色0-m-1，若t与所有其它相邻顶点无颜色冲突，则继续为下一顶点着色。

**算法实现**

```c
//图着色问题回溯法
/*
无向图邻接矩阵示例
0 1 1 1
1 0 1 0
1 1 0 1
1 0 1 0
*/

#include <stdio.h>
#include <stdlib.h>

int n, m;
int *color;
int **graph;

int ok(int j) {
    int i;
    for(i = 0; i < j; i++)
        if(graph[j][i] == 1 && color[i] == color[j])
            return 0;
    return 1;
}

int getM() {
    int i, max = 0;
    for(i = 0; i < n; i++) {
        if(color[i] > max)
            max = color[i];
    }
    return max + 1;
}

void graphcolor() {
    int i, j;

    //初始化
    j = 0;
    for(i = 0; i < n; i++) {
        color[i] = -1;
    }

    while(1) {
        color[j]++;
        //找到一个满足的颜色
        while(1) {
            if (ok(j))
                break;
            else
                color[j]++;
        }
        if(j == n - 1) {
            break;
        } else {
            j++; //处理下一个顶点
        }
    }

    printf("m的值：%d\n", getM());
    for(i = 0; i < n; i++) {
        printf("%d ", color[i]);
    }
    printf("\n");

}

int main() {
    int i, j;

    printf("输入顶点数n:\n");
    scanf("%d", &n);

    color = (int *)malloc(sizeof(int) * n);
    graph = (int **)malloc(sizeof(int *) * n);
    for(i = 0; i < n; i++) {
        graph[i] = (int *)malloc(sizeof(int) * n);
    }

    printf("输入无向图的邻接矩阵:\n");
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    printf("着色方案:\n");
    graphcolor();

    free(color);
    for(i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}
```

#### 问题三

**提出问题**

有5个球队A、B、C、D、E进行比赛，已经比赛完的场次有：A同B、C，B同D，E同C。每个球队每周比赛一次。试给出一种调度方法，使得所有的队能在最短的时间内相互之间完成比赛。（PS：此题笔者一开始题目没看全，险些偏离方向，所以这里也提醒大家要看完整题目在动手解题）

**问题分析**

5个球队，总共要打10场比赛，已经进行了4场，剩余AD、AE、BC、BE、CD、DE这6场，以这6场比赛为节点，无法同周举行的连接一条边，则可转换为m着色问题。即用问题二的解法即可解决。

## 后记

很幸运自己选择了考研这条路，很短的时间内已经让我觉得能够学到很多东西。既然选择了远方,便只顾风雨兼程。