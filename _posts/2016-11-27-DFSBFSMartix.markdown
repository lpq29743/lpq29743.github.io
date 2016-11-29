---
layout:     post
title:      "图的搜索（一）"
subtitle:   "基于邻接矩阵的深度遍历算法和广度遍历算法"
date:       2016-11-27 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
    - 图
---

> 图的搜索上篇~~~


## 前言

最近在复习数据结构，对图的算法重新熟悉了一遍，这一篇就讲讲基于邻接矩阵的图的深搜和广搜。

---

## 正文

#### 算法描述

**深度遍历算法描述**

> 1.访问初始结点v，并标记结点v为已访问。
>
> 2.查找结点v的第一个邻接结点w。
>
> 3.若w存在，则继续执行4，否则算法结束。
>
> 4.若w未被访问，对w进行深度优先遍历递归（即把w当做另一个v，然后进行步骤123）。
>
> 5.查找结点v的w邻接结点的下一个邻接结点，转到步骤3。

**广度遍历算法描述**

> 1.访问初始结点v，并标记结点v为已访问。
>
> 2.结点v入队列。
>
> 3.当队列非空时，继续执行，否则算法结束。
>
> 4.出队列，取得队头结点u。
>
> 5.查找结点u的第一个邻接结点w。
>
> 6.若结点u的邻接结点w不存在，则转到步骤3；否则循环执行以下三个步骤：
>
> 1). 若结点w尚未被访问，则访问结点w并标记为已访问。
>
> 2). 结点w入队列。
>
> 3). 查找结点u的继w邻接结点后的下一个邻接结点w，转到步骤6。

#### 算法实现

```c
// 基于邻接矩阵（无向图）的深度遍历算法和广度遍历算法

/*
0 1 1 1 1 0 0
1 0 0 1 1 0 0
1 0 0 0 0 1 1
1 1 0 0 1 0 0
1 1 0 1 0 0 0
0 0 1 0 0 0 1
0 0 1 0 0 1 0
dfs:a b d e c f g
bfs:a b c d e f g
*/

#include <stdio.h>
#include <stdlib.h>

int n, num;
int **graph;
int *visited, *res, *queue;
int data_input = 1;
// 由于在Jekyll下会被解析错误，所以大家手动输入
/*int myarray[7][7] = {{0, 1, 1, 1, 1, 0, 0}, {1, 0, 0, 1, 1, 0, 0}, {1, 0, 0, 0, 0, 1, 1}, {1, 1, 0, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 1, 0}};*/

void create() {
    int i, j;
    printf("Please input the number of vertices:\n");
    scanf("%d", &n);
    graph = (int **)malloc(sizeof(int *) * n);
    for(i = 0; i < n; i++) {
        graph[i] = (int *)malloc(sizeof(int) * n);
    }
    visited = (int *)malloc(sizeof(int *) * n);
    res = (int *)malloc(sizeof(int *) * n);
    queue = (int *)malloc(sizeof(int *) * n);
    if(data_input == 1) {
        printf("\nPlease input the adjacency matrix:\n");
        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                scanf("%d", &graph[i][j]);
            }
        }
    } else {
        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                graph[i][j] = myarray[i][j];
            }
        }
    }
    printf("\nHere is the adjacency matrix according to your input\n");
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }
}

void init() {
    int i;
    for(i = 0; i < n; i++) {
        visited[i] = 0;
    }
    num = 0;
}

void dfs(int v) {
    visited[v] = 1;
    res[num++] = v;
    int i;
    for(i = 0; i < n; i++) {
        if(graph[v][i] == 1 && visited[i] == 0) {
            dfs(i);
        }
    }
}

void bfs(int v) {
    int i, k, head = 0, tail = 0;
    visited[v] = 1;
    res[num++] = v;
    queue[tail++] = v;
    while(tail - head != 0) {
        k = queue[head++];
        for(i = 0; i < n; i++) {
            if(graph[k][i] == 1 && visited[i] == 0) {
                visited[i] = 1;
                res[num++] = i;
                queue[tail++] = i;
            }
        }
    }
}

void show() {
    int i;
    for(i = 0; i < n; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");
}

int main() {
    create();
    init();
    dfs(0);
    printf("\nthe result of dfs as follows:\n");
    show();
    init();
    bfs(0);
    printf("\nthe result of bfs as follows:\n");
    show();
    return 0;
}
```

#### 总结

深度优先遍历用的数据结构是栈（递归实现的时候使用的是系统栈），主要是递归实现；而广度优先遍历用的数据结构是队列，主要是迭代实现。

## 后记

图是数据结构中蛮重要也比较有趣的一块，深搜和广搜是图的基础，所以必须牢固掌握。
