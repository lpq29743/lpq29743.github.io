---
layout:     post
title:      "杭电OJ刷题记之搜索"
subtitle:   "最快的方式找到最好的结果"
date:       2017-07-07 14:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> The way to the answer
>


## 前言

搜索是程序常见的操作之一，经常会使用到递归等知识，这篇文章就让我们一起来学一下。

---

## 正文

**诱惑者的骨**

***题目来源***

[HDOJ 1010 Tempter of the Bone](http://acm.hdu.edu.cn/showproblem.php?pid=1010)

***题目分析***

这道题是常见的递归题，相对来说比较简单，但有两点必须注意：

1. 递归实际上是效率很低的做法，操作不当甚至可能导致程序崩溃，所以在某些情况下，我们要尽可能避免使用递归。本题由于相对简单，所以为了思路清晰，我们依旧使用递归。但为了加强递归的效率，我们必须做一些适当的剪枝，这也是算法题中经常出现的。下面的程序并没有做太多的剪枝，网上提供的其他版本程序提供的剪枝很多，比如奇偶剪枝，即当出发地与目的地之间的距离与所给的时间奇偶性不同的话，那么肯定无法走出迷宫
2. 这道题的另一个关键点是回溯法。由于题意要求，每走完一步，方格就不可走，所以我们在进行试探前需将方格设为不可走，试探后必须回溯，将方格设为可走

***实现代码***

```c++
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<algorithm>
using namespace std;

typedef struct {
    double x;
    double y;
} point;

int cmpxy (const point a, const point b) {
    if(a.x != b.x) {
        return a.x < b.x;
    } else {
        return a.y < b.y;
    }
}

double dist(point *p, int i, int j) {
    return sqrt((p[i].x - p[j].x) * (p[i].x - p[j].x) +
                (p[i].y - p[j].y) * (p[i].y - p[j].y));
}

double getMin(point *p, int low, int high) {
    if(low == high - 1) {
        return dist(p, low, high);
    } else if(low == high - 2) {
        double dist1, dist2, dist3, temp;
        dist1 = dist(p, low, low + 1);
        dist2 = dist(p, low, high);
        dist3 = dist(p, low + 1, high);
        temp = dist1 > dist2 ? dist2 : dist1;
        return temp > dist3 ? dist3 : temp;
    } else {
        double dist1, dist2;
        int mid = low + (high - low) / 2;
        dist1 = getMin(p, low, mid);
        dist2 = getMin(p, mid + 1, high);
        double mindist = dist1 > dist2 ? dist2 : dist1;
        for(int i = mid + 1; i <= high; i++) {
            if(p[i].x > (p[mid].x - mindist) && p[i].x < (p[mid].x + mindist)) {
                if(dist(p, i, mid) < mindist) {
                    mindist = dist(p, i, mid);
                }
            }
        }
        return mindist;
    }
}

int main() {
    int n;
    while(scanf("%d", &n) != 0 && n) {
        point *p = (point *)malloc(sizeof(point) * n);
        for(int i = 0; i < n; i++) {
            scanf("%lf %lf", &p[i].x, &p[i].y);
        }
        sort(p, p + n, cmpxy);
        int tag = 0;
        double eps = 1e-8;
        for(int i = 0; i < n - 1; i++) {
            if(fabs(p[i].x - p[i + 1].x) < eps && fabs(p[i].y - p[i + 1].y) < eps)
                tag = 1;
        }
        if(tag) {
            printf("0.00\n");
            continue;
        } else {
            printf("%.2lf\n", getMin(p, 0, n - 1) / 2);
        }
    }
    return 0;
}
```
**素数环**

***题目来源***

[HDOJ 1016 Prime Ring Problem](http://acm.hdu.edu.cn/showproblem.php?pid=1016)

***题目分析***

这道题也是一道典型的深度优先搜索问题，使用到了回溯法。我一开始是参考了全排的算法来进行程序的编写的，可不知为什么，程序提交一直WA（如果后面后进展，我再提及）。所以我最终还是参考了图深度优先遍历的算法来进行实现。

对于这道题，用以下的代码就可以AC了，可如果要再进行优化，可以通过以下几点进行考虑：

1. 由于涉及到素数判断很少，最大的判断数也就是18+19=37而已，所以可以通过建立素数表，用查表的方式来加快素数判断
2. 如果输入的n是奇数的话，那么1-n之间一共有n / 2个偶数，n / 2 + 1个奇数，也就是奇数比偶数多一个。那么把这n个数排成一个环，根据鸽巢原理，必然两个奇数相邻，而奇数之和是偶数，偶数不是素数，所以得出结论：如果n是奇数，则没有满足条件的排列。通过这一点，当n是奇数时，我们直接返回即可，这样可大大减少计算量


***实现代码***

```c++
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int n;
int *s;
int *visited;

int isPrime(int num) {
    int result = 1;
    for(int i = 2; i < num; i++) {
        if(num % i == 0) {
            result = 0;
            break;
        }
    }
    return result;
}

void dfs(int step) {
    if(step == n) {
        if(isPrime(s[n - 1] + s[0])) {
            for(int i = 0; i < n; i++) {
                if(i != 0) {
                    printf(" ");
                }
                printf("%d", s[i]);
            }
            printf("\n");
        }
    }
    for(int i = 2; i <= n; i++) {
        if(isPrime(s[step - 1] + i) && visited[i - 1] == 0) {
            s[step] = i;
            visited[i - 1] = 1;
            dfs(step + 1);
            visited[i - 1] = 0;
        }
    }
}

int main() {
    int count = 0;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        visited = (int *)malloc(sizeof(int) * n);
        memset(visited, 0, sizeof(int) * n);
        s[0] = 1;
        printf("Case %d:\n", ++count);
        dfs(1);
        printf("\n");
    }
    return 0;
}
```

**伊格和公主（一）**

***题目来源***

[HDOJ 1026 Ignatius and the Princess I](http://acm.hdu.edu.cn/showproblem.php?pid=1026)

***题目分析***

这道题我先是用深搜的算法做的，可是却无奈超时了。查了一下网上的资料才发现，对于这道题，用广搜来做更加适合，因为广搜相比深搜，能够更快地得到最短的路径。不过有点奇怪的是，我这里原先是用数组来实现队列的操作的，可是会出现超时的情况，而我把它改成C++下自带的队列后，却成功的AC了。

***实现代码***

```c++
#include<stdio.h>
#include<queue>
#define N 101
using namespace std;

typedef struct {
    int x;
    int y;
    int prex;
    int prey;
    int cost;
} Node;

int n, m;
char s[N][N];
Node node[N][N];
int dir[4][2] = { { -1, 0}, {1, 0}, {0, -1}, {0, 1} };

int isOK(int x, int y) {
    if(x >= 0 && x < n && y >= 0 && y < m && s[x][y] != 'X') {
        return 1;
    }
    return 0;
}

void output() {
    if(node[n - 1][m - 1].cost != -1) {
        Node stack[N * N];
        Node a, b;
        int count = 1, tmp, top = -1;
        printf("It takes %d seconds to reach the target position, let me show you the way.\n", node[n - 1][m - 1].cost);
        a = node[n - 1][m - 1];
        while(1) {
            if(a.x == 0 && a.y == 0)
                break;
            stack[++top] = a;
            a = node[a.prex][a.prey];
        }
        a = node[0][0];
        while(top != -1) {
            b = stack[top--];
            printf("%ds:(%d,%d)->(%d,%d)\n", count++, a.x, a.y, b.x, b.y);
            if(s[b.x][b.y] != '.') {
                tmp = s[b.x][b.y] - '0';
                while(tmp--) {
                    printf("%ds:FIGHT AT (%d,%d)\n", count++, b.x, b.y);
                }
            }
            a = b;
        }
    } else {
        printf("God please help our poor hero.\n");
    }
    printf("FINISH\n");
}

void bfs() {
    queue<Node> q;
    Node a, b;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            node[i][j].cost = -1;
        }
    }
    a.x = a.y = a.prex = a.prey = a.cost = 0;
    if(s[0][0] != '.') {
        a.cost += s[0][0] - '0';
    }
    node[0][0] = a;
    q.push(a);
    while(!q.empty()) {
        a = q.front();
        q.pop();
        for(int i = 0; i < 4; i++) {
            b.x = a.x + dir[i][0];
            b.y = a.y + dir[i][1];
            if(!isOK(b.x, b.y)) {
                continue;
            }
            if(s[b.x][b.y] == '.') {
                b.cost = a.cost + 1;
            } else {
                b.cost = a.cost + s[b.x][b.y] - '0' + 1;
            }
            if(b.cost < node[b.x][b.y].cost || node[b.x][b.y].cost == -1) {
                b.prex = a.x;
                b.prey = a.y;
                node[b.x][b.y] = b;
                q.push(b);
            }
        }
    }
    output();
}

int main() {
    while(scanf("%d%d", &n, &m) == 2) {
        for(int i = 0; i < n; i++) {
            scanf("%s", s[i]);
        }
        bfs();
    }
    return 0;
}
```

**收集珠宝**

***题目来源***

[HDOJ 1044 Collect More Jewels](http://acm.hdu.edu.cn/showproblem.php?pid=1044)

***题目分析***

本题我一开始是用DFS的思路做的，可是TLE了。分析了一下发现，深搜的时间复杂度达到2的L次方，耗时过长。查了一下资料，最终使用BFS+DFS的思路解决。具体做法是：用BFS得到起点、珠宝以及终点之间的最短路径（求无权图的最短路径一般使用BFS），然后再用DFS搜索最大获得价值。另外，我们可以用sum保存所有珍宝的价值和来方便深搜剪枝。值得注意的是，根据题目数据，至少要定义51 * 51的二维数组才足以存储地图，因为接收每一行的字符串时不仅要接收数据部分，还要接收结束符。关于此题，网上还有一种BFS+状态压缩的方法，感兴趣的朋友可以去看下。

***实现代码***

```c++
#include<stdio.h>
#include<string.h>
#include<queue>
#define INF 1e8
using namespace std;

struct node {
    int x, y;
    int t;
}s, u, v;

int T;
int W, H, L, M;
int value[12];
char maze[55][55];
int sum, ans;
int visited[55][55], visited2[12];
int path[12][12];
int dir[4][2] = { { -1, 0}, {1, 0}, {0, 1}, {0, -1} };

void bfs(int x, int y, int from) {
    memset(visited, 0, sizeof(visited));
    s.x = x;
    s.y = y;
    s.t = 0;
    visited[s.x][s.y] = 1;
    queue<node> q;
    q.push(s);
    while(!q.empty()) {
        u = q.front();
        q.pop();
        for(int i = 0; i < 4; i++) {
            v.x = u.x + dir[i][0];
            v.y = u.y + dir[i][1];
            if(v.x < 0 || v.x >= H || v.y < 0 || v.y >= W || maze[v.x][v.y] == '*' || visited[v.x][v.y]) {
                continue;
            }
            visited[v.x][v.y] = 1;
            v.t = u.t + 1;
            if(maze[v.x][v.y] != '.') {
                if(maze[v.x][v.y] == '@') {
                    path[from][0] = v.t;
                } else if(maze[v.x][v.y] == '<') {
                    path[from][M + 1] = v.t;
                } else {
                    path[from][maze[v.x][v.y] - 'A' + 1] = v.t;
                }
            }
            q.push(v);
        }
    }
}

void dfs(int cur, int s, int time) {
    if(time > L || ans == sum) {
        return;
    }
    if(cur == M + 1) {
        if(ans < s) {
            ans = s;
        }
        return;
    }
    for(int i = 1; i <= M + 1; i++) {
        if(visited2[i]) {
            continue;
        }
        visited2[i] = 1;
        dfs(i, s + value[i - 1], time + path[cur][i]);
        visited2[i] = 0;
    }
}

int main() {
    int c = 0;
    scanf("%d", &T);
    while(T--) {
        sum = 0;
        scanf("%d%d%d%d", &W, &H, &L, &M);
        for(int i = 0; i < M; i++) {
            scanf("%d", &value[i]);
            sum += value[i];
        }
        value[M] = 0;
        for(int i = 0; i < H; i++) {
            scanf("%s", maze[i]);
        }
        for(int i = 0; i <= M + 1; i++) {
            for(int j = 0; j <= M + 1; j++) {
                path[i][j] = INF;
            }
        }
        for(int i = 0; i < H; i++) {
            for(int j = 0; j < W; j++) {
                if(maze[i][j] == '.' || maze[i][j] == '*') {
                    continue;
                } else if(maze[i][j] == '@') {
                    bfs(i, j, 0);
                } else if(maze[i][j] == '<') {
                    bfs(i, j, M + 1);
                } else if(maze[i][j] <= 'J' && maze[i][j] >= 'A') {
                    bfs(i, j, maze[i][j] - 'A' + 1);
                }
            }
        }
        ans = -1;
        memset(visited2, 0, sizeof(visited2));
        dfs(0, 0, 0);
        printf("Case %d:\n", ++c);
        if(ans == -1) {
            printf("Impossible\n");
        } else {
            printf("The best score is %d.\n", ans);
        }
        if(T) {
            printf("\n");
        }
    }
    return 0;
}
```

**最大碉堡数**

***题目来源***

[HDOJ 1045 Fire Net](http://acm.hdu.edu.cn/showproblem.php?pid=1045)

***题目分析***

本题在HDOJ里面被分在图论和贪心算法两个板块中，但实际上，这道题用DFS解决更加方便，除了以下本人的DFS算法之外，网上还有另一种普遍的DFS算法思路，感兴趣的朋友可以上网查查。如果想要练练二分图或者贪心算法，则可以试一试，不过难度也会随之上升。

***实现代码***

```c
#include<stdio.h>

int n, max;
char map[5][5];

void change(int i, int j, char c, char pre) {
    int curx = i, cury = j;
    map[i][j] = c;
    while(--curx && curx >= 0 && map[curx][j] == pre) {
        map[curx][j] = c;
    }
    while(--cury && cury >= 0 && map[i][cury] == pre) {
        map[i][cury] = c;
    }
    curx = i;
    cury = j;
    while(++curx && curx < n && map[curx][j] == pre) {
        map[curx][j] = c;
    }
    while(++cury && cury < n && map[i][cury] == pre) {
        map[i][cury] = c;
    }
}

void dfs(int x, int y, int count) {
    int isFound = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i < x || (i == x && j < y)) {
                continue;
            }
            if(map[i][j] == '.') {
                isFound = 1;
                change(i, j, count + '0', '.');
                dfs(i, j, count + 1);
                change(i, j, '.', count + '0');
            }
        }
    }
    if(!isFound) {
        if(count > max) {
            max = count;
        }
    }
}

int main() {
    while(scanf("%d", &n) == 1 && n != 0) {
        for(int i = 0; i < n; i++) {
            scanf("%s", map[i]);
        }
        max = 0;
        dfs(0, 0, 0);
        printf("%d\n", max);
    }
    return 0;
}
```

## 后记

搜索经常涉及到递归、剪枝、回溯等等这些知识点，只要多加练习，我们才可以掌握好这类题目。
