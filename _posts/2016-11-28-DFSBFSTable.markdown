---
layout:     post
title:      "图的搜索（二）"
subtitle:   "基于邻接表的深度遍历算法和广度遍历算法"
date:       2016-11-29 11:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
    - 图
---

> 图的搜索上篇~~~


## 前言

本文紧接着上一篇继续讲一讲基于邻接表的深搜算法和广搜算法。

---

## 正文

关于深搜算法和广搜算法的算法描述和比较，[上一篇文章](https://lpq29743.github.io/redant/2016/11/27/DFSBFSMartix/)已经提及，所以这篇文章就直接给出算法实现。

```c
// 基于邻接表（无向图）的深度遍历算法和广度遍历算法

/*
myarray[10][2]
dfs:a b d e c f g
bfs:a b c d e f g
*/

#include <stdio.h>
#include <stdlib.h>

struct node {
    int adjvex;
    struct node *next;
};

int n, num;
int *visited, *res, *queue;
int data_input = 1;
int myarray[10][2]={ {0, 1},{0, 2}, {0, 3}, {0, 4}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {3, 4}, {5, 6} };
struct node **graph;

void create() {
    int i, v1, v2;
    struct node *p, *q;
    printf("Please input the number of vertices:\n");
    scanf("%d", &n);
    graph = (struct node **)malloc(sizeof(struct node *) * n);
    for(i = 0; i < n; i++) {
        graph[i] = (struct node *)malloc(sizeof(struct node));
        graph[i]->adjvex = i;
        graph[i]->next = NULL;
    }
    visited = (int *)malloc(sizeof(int *) * n);
    res = (int *)malloc(sizeof(int *) * n);
    queue = (int *)malloc(sizeof(int *) * n);
    for(i = 0; i < 10; i++) {
        v1 = myarray[i][0];
        v2 = myarray[i][1];
        p = graph[v1];
        while(p->next != NULL) {
            p = p->next;
        }
        q = (struct node *)malloc(sizeof(struct node));
        q->adjvex = v2;
        q->next = NULL;
        p->next = q;
        p = graph[v2];
        while(p->next != NULL) {
            p = p->next;
        }
        q = (struct node *)malloc(sizeof(struct node));
        q->adjvex = v1;
        q->next = NULL;
        p->next = q;
    }
    printf("\nthe result of table:\n");
    for(i = 0; i < n; i++) {
        p = graph[i];
        printf("%d:", i);
        while(p->next != NULL) {
            p = p->next;
            printf("%d ", p->adjvex);
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
    struct node *p;
    visited[v] = 1;
    res[num++] = v;
    p = graph[v];
    while(p->next != NULL) {
        p = p->next;
        if(visited[p->adjvex] == 0) {
            dfs(p->adjvex);
        }
    }
}

void bfs(int v) {
    int k, head = 0, tail = 0;
    struct node *p;
    visited[v] = 1;
    res[num++] = v;
    queue[tail++] = v;
    while(tail - head != 0) {
        k = queue[head++];
        p = graph[k];
        while(p->next != NULL) {
            p = p->next;
            if(visited[p->adjvex] == 0) {
                visited[p->adjvex] = 1;
                res[num++] = p->adjvex;
                queue[tail++] = p->adjvex;
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

## 后记

之所以重新提起图的深搜算法和广搜算法，主要是因为在一个求关节点的题目中遇到过，后面笔者也会就关节点的求法写一篇文章。
