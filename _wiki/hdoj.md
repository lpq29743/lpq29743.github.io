---
layout: wiki
title: HDOJ
categories: Algorithm
description: 杭电OJ
keywords: 杭电OJ
---

为了备战面试，也为了增强自身的能力，决定按杭电 OJ 的[题目分类集](http://acm.hdu.edu.cn/typeclass.php)进行一段时间的刷题和学习。

### 第 1 章 大数运算

#### 1.1 大数相加

**题目来源**

[HDOJ 1002 A + B Problem II](http://acm.hdu.edu.cn/showproblem.php?pid=1002)

**题目分析**

大数相加问题是比较常见的大数运算题目，也是最基础的大数运算，其具体的解决思路如下：

1. 计算出两个大数的长度，并记录最小长度
2. 初始化存储结果的字符数组，数组中每个元素的初始值均为0
3. 在最小长度内进行计算，分为进位和不进位两种情况
4. 根据长度较大的大数长出的部分进行计算，同样分为进位和不进位两种情况
5. 计算结果字符数组的长度，并倒序输出结果

**实现代码**

```c
#include<stdio.h>

int main() {
    int n;
    char a[1001], b[1001], c[1001];
    scanf("%d", &n);
    for(int i = 0; i < n; i++) {
        scanf("%s %s", &a, &b);
        int alength = 0, blength = 0;
        while(a[alength] != '\0') {
            alength++;
        }
        while(b[blength] != '\0') {
            blength++;
        }
        int length = alength;
        if(length > blength)
            length = blength;
        for(int i = 0; i < 1001; i++) {
            c[i] = 0;
        }
        for(int i = 0; i < length; i++) {
            int t1 = a[alength - 1 - i] - '0';
            int t2 = b[blength - 1 - i] - '0';
            if(t1 + t2 + c[i] > 9) {
                c[i] = (t1 + t2 + c[i]) % 10;
                c[i + 1] = 1;
            } else {
                c[i] = t1 + t2 + c[i];
            }
        }
        if(length < alength) {
            for(int i = length; i < alength; i++) {
                if(a[alength - 1 - i] - '0' + c[i] > 9) {
                    c[i] = (a[alength - 1 - i] - '0' + c[i]) % 10;
                    c[i + 1] = 1;
                } else {
                    c[i] = a[alength - 1 - i] - '0' + c[i];
                }
            }
        }
        if(length < blength) {
            for(int i = length; i < blength; i++) {
                if(b[blength - 1 - i] - '0' + c[i] > 9) {
                    c[i] = (b[blength - 1 - i] - '0' + c[i]) % 10;
                    c[i + 1] = 1;
                } else {
                    c[i] = b[blength - 1 - i] - '0' + c[i];
                }
            }
        }
        if(i != 0) {
            printf("\n");
        }
        printf("Case %d:\n%s + %s = ", i + 1, a, b);
        int clength = 1000;
        while(c[clength] == 0) {
            clength--;
        }
        for(int i = clength; i >= 0; i--) {
            printf("%d", c[i]);
        }
        printf("\n");
    }
    return 0;
}
```

#### 1.2 N 的阶乘

**题目来源**

[HDOJ 1042 N!](http://acm.hdu.edu.cn/showproblem.php?pid=1042)

**题目分析**

这道题是典型的大数相乘问题，但与以往的题目不同。由于本题数据量太大，所以我们不再像以往一样，选择字符数组来存储大数，而是使用整型数组来存储，其中进制变成逢 100000 进 1。由于进制改变了，所以在最后的输出上要特别注意格式的控制。

**实现代码**

```c
#include<stdio.h>
#include<string.h>
#define N 10000

int n;
int s[N + 1];

int main() {
    while(scanf("%d", &n) == 1) {
        memset(s, 0, sizeof(s));
        s[0] = 1;
        for(int i = 2; i <= n; i++) {
            for(int j = N; j >= 0; j--) {
                s[j] = s[j] * i;
            }
            for(int j = 0; j <= N; j++) {
                s[j + 1] += s[j] / 100000;
                s[j] %= 100000;
            }
        }
        int k = N;
        while(!s[k]) {
            k--;
        }
        printf("%d", s[k--]);
        while(k >= 0) {
            printf("%05d", s[k--]);
        }
        printf("\n");
    }
    return 0;
}
```

#### 1.3 整数探究

**题目来源**

[HDOJ 1047 Integer Inquiry](http://acm.hdu.edu.cn/showproblem.php?pid=1047)

**题目分析**

典型的大数加法题目，比较简单，注意格式还有只有 0 的特殊情况就可以。

**实现代码**

```c
#include<stdio.h>
#include<string.h>

char s[105][105];

void plus(char *a, char *b) {
    int l1, l2, l;
    char c[105];
    l1 = strlen(a);
    l2 = strlen(b);
    l = l1 < l2 ? l1 : l2;
    memset(c, '0', sizeof(c));
    for(int i = 0; i < l; i++) {
        c[i] = a[l1 - i - 1] - '0' + b[l2 - i - 1] - '0' + c[i];
        c[i + 1] = (c[i] - '0') / 10 + '0';
        c[i] = (c[i] - '0') % 10 + '0';
    }
    for(int i = l; i < l1; i++) {
        c[i] = c[i] + a[l1 - i - 1] - '0';
        c[i + 1] = (c[i] - '0') / 10 + '0';
        c[i] = (c[i] - '0') % 10 + '0';
    }
    for(int i = l; i < l2; i++) {
        c[i] = c[i] + b[l2 - i - 1] - '0';
        c[i + 1] = (c[i] - '0') / 10 + '0';
        c[i] = (c[i] - '0') % 10 + '0';
    }
    int l3 = 104;
    while(c[l3] == '0') {
        l3--;
    }
    for(int i = 0; i <= l3/2; i++) {
        char temp = c[i];
        c[i] = c[l3 - i];
        c[l3 - i] = temp;
    }
    c[l3 + 1] = '\0';
    strcpy(b, c);
}

int main() {
    int t;
    scanf("%d", &t);
    while(t--) {
        int num = 0;
        while(1) {
            scanf("%s", s[num]);
            if(strcmp(s[num], "0") == 0) {
                break;
            } else {
                num++;
            }
        }
        for(int i = 0; i < num - 1; i++) {
            plus(s[i], s[i + 1]);
        }
        if(num == 0) {
            printf("0\n");
        } else {
            printf("%s\n", s[num - 1]);
        }
        if(t) {
            printf("\n");
        }
    }
    return 0;
}
```

### 第 2 章 演绎推理

#### 2.1 出栈可能数

**题目来源**

[HDOJ 1023 Train Problem II](http://acm.hdu.edu.cn/showproblem.php?pid=1023)

**题目分析**

这道问题一上来，可能有些算法初学者会打算通过栈和队列来模拟所有出栈可能，然后计算出所有的可能数。这种方法是可行的，但难度较大，而且计算量也会非常大，很容易做错。这个时候我们就要用到组合数学中的卡特兰数了。卡特兰数在算法题中运用十分广泛，如果还不熟悉的话，可以参考网上的资料或者是《编程之美》中的买票找零问题。对于卡特兰数，我们主要是记住两条公式，第一条是最后的解`h(n) = C(2n, n) / (n + 1)`，这在选择填空题中用的比较多，而另一条公式则在编程题中用的多，它也是用来解决这道题的公式，该公式是`h(n) = h(n - 1) * (4 * n - 2) / (n + 1)`。

对于这道题，由于计算到后面的数据会很大，所以我们在计算卡特兰数的时候还要用到大数运算的思维，题目主要用到的是大数乘法和大数除法。总体来说，这道题的质量还是相当的高的，能够大大锻炼我们的算法和编程能力。

**实现代码**

```c++
#include<stdio.h>
#define N 101

// 第n个Catalan数存在a[n]中，a[n][0]表示长度
// 数是倒着存的，输出时需倒序输出
int s[N][N];

void ktl() {
    int len;        // 上一个数的长度
    int t;        // 进位值
    s[1][0] = 1;
    s[1][1] = 1;
    s[2][0] = 1;
    s[2][1] = 2;
    len = 1;
    for(int i = 3; i < 101; i++) {
        t = 0;

        // 大数乘法

        // 在被乘数的长度范围内进行计算
        for(int j = 1; j <= len; j++) {
            int tmp = s[i - 1][j] * (4 * i - 2) + t;
            t = tmp / 10;
            s[i][j] = tmp % 10;
        }
        // 根据进位值添加长度并赋值
        while(t) {
            s[i][++len] = t % 10;
            t /= 10;
        }

        // 大数除法
        for(int j = len; j > 0; j--) {
            int tmp = s[i][j] + t * 10;
            s[i][j] = tmp / (i + 1);
            t = tmp % (i + 1);
        }
        while(!s[i][len]) {
            len--;
        }
        s[i][0] = len;
    }
}

int main() {
    ktl();
    int n;
    while(scanf("%d", &n) == 1) {
        for(int i = s[n][0]; i > 0; i--) {
            printf("%d", s[n][i]);
        }
        printf("\n");
    }
    return 0;
}
```

#### 2.2 三角波

**题目来源**

[HDOJ 1030 Delta-wave](http://acm.hdu.edu.cn/showproblem.php?pid=1030)

**题目分析**

这是一道典型的规律题，找到的规律可以有很多种形式，这里提供两种：[简单易懂的](http://www.wutianqi.com/?p=2362)和[简洁晦涩的](http://blog.csdn.net/u014174811/article/details/41443177)。这里我们直接采用第二种的代码。

**实现代码**

```c++
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main() {
    int a, b;
    int aX, aY, bX, bY, aLayer, bLayer, step;
    while(scanf("%d%d", &a, &b) != EOF) {
        aLayer = ceil(sqrt((double)a)); //求出数a所在层
        bLayer = ceil(sqrt((double)b)); //求出数b所在层
        if(aLayer == bLayer) {
            printf("%d\n", abs(a - b));
        } else {
            aX = (aLayer * aLayer - a) / 2; //计算a的X坐标
            bX = (bLayer * bLayer - b) / 2; //计算b的X坐标
            aY = (a - (aLayer * aLayer - 2 * aLayer + 2)) / 2; //计算a的Y坐标
            bY = (b - (bLayer * bLayer - 2 * bLayer + 2)) / 2; //计算b的Y坐标
            step = abs(aX - bX) + abs(aY - bY) + abs(aLayer - bLayer);
            printf("%d\n", step); //求出最终步骤
        }
    }
}
```

#### 2.3 计算机转换

**题目来源**

[HDOJ 1041 Computer Transformation](http://acm.hdu.edu.cn/showproblem.php?pid=1041)

**题目分析**

这也是一道找规律的题目。一番分析之后，我们可以得到递推公式：`f(n) = f(n - 1) + f(n - 2) * 2`。比以上找规律的题目稍微复杂点的是，它到后面的数据太大，所以我们要用大数加法来解决这个问题。

**实现代码**

```c
#include<stdio.h>
#include<string.h>
#define N 1001

int n;
char s[N][N];

void init() {
    memset(s, '0', sizeof(s));
    s[1][0] = '0';
    s[2][0] = '1';
    for(int i = 3; i < N; i++) {
        for(int j = 0; j < N; j++) {
            s[i][j] += (s[i - 1][j] - '0') + (s[i - 2][j] - '0') * 2;
        }
        for(int j = 0; j < N; j++) {
            if(s[i][j] > '9') {
                s[i][j + 1] += (s[i][j] - '0') / 10;
                s[i][j] = (s[i][j] - '0') % 10 + '0';
            }
        }
    }
}

int main() {
    init();
    while(scanf("%d", &n) == 1) {
        if(n == 1) {
            printf("0\n");
        } else {
            for(int i = N - 1; i >= 0; i--) {
                if(s[n][i] != '0') {
                    while(i >= 0) {
                        printf("%c", s[n][i--]);
                    }
                    printf("\n");
                    break;
                }
            }
        }
    }
    return 0;
}
```

### 第 3 章 图论

#### 3.1 策略游戏

**题目来源**

[HDOJ 1054 Strategic Game](http://acm.hdu.edu.cn/showproblem.php?pid=1054)

**题目分析**

这题考察的是二分图的最小覆盖点，即求最大匹配数，而且由于图是双向的，所以求得的结果还要除以2。对于这道题，如果直接套模板，会出现TLE的情况，这是因为模板存储边采用的是邻接矩阵，时间复杂度为 O(n^3^)。因此我们要对模板进行改进，采用邻接表来存储边，这种情况下复杂度为 O(m*n)，并不会超时。

**实现代码**

```c++
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<vector>
#define N 1550
using namespace std;

int n;
vector<int> edges[N];
int linker[N];
int used[N];

int dfs(int u) {
    for(unsigned int i = 0; i < edges[u].size(); i++) {
        if(!used[edges[u][i]]) {
            used[edges[u][i]] = 1;
            if(linker[edges[u][i]] == -1 || dfs(linker[edges[u][i]])) {
                linker[edges[u][i]] = u;
                return 1;
            }
        }
    }
    return 0;
}

int hungary() {
    int res = 0;
    memset(linker, -1, sizeof(linker));
    for(int u = 0; u < n; u++) {
        memset(used, 0, sizeof(used));
        if(dfs(u)) {
            res++;
        }
    }
    return res;
}

int main() {
    int u, v, num;
    while(scanf("%d", &n) == 1) {
        for(int i = 0; i < n; i++) {
            edges[i].clear();
        }
        for(int i = 0; i < n; i++) {
            scanf("%d:(%d)", &u, &num);
            for(int j = 0; j < num; j++) {
                scanf("%d", &v);
                edges[u].push_back(v);
                edges[v].push_back(u);
            }
        }
        int result = hungary();
        printf("%d\n", result / 2);
    }
    return 0;
}
```

#### 3.2 男孩和女孩

**题目来源**

[HDOJ 1068 Girls and Boys](http://acm.hdu.edu.cn/showproblem.php?pid=1068)

**题目分析**

这是我做的第一道二分图最大匹配问题，所以也是查了很多的资料。关于二分图最大匹配问题，我们常用的解决方法是匈牙利算法，具体可以通过[《二分图的最大匹配、完美匹配和匈牙利算法》](https://www.renfei.org/blog/bipartite-matching.html)这篇文章进行了解，这类题的[模板](http://www.cnblogs.com/kuangbin/archive/2011/08/09/2132828.html)，kuangbin大神也已经给出。

对于这道题，实际上要求的是最大独立集，所以我们可以通过`二分图最大独立集 = 顶点数 - 二分图最大匹配`这条公式来进行计算，但由于输入建图是双向的，所以最大匹配要取一半。由于是第一次做这种题，所以我也写了较多的注释。

**实现代码**

```c++
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 500

int n;      // 二分图两个集合点数
int g[N][N];    // 采用邻接矩阵记录边
int linker[N];      // 记录右边匹配顶点及匹配边
int used[N];

int dfs(int u) {    // 寻找增广路
    for(int v = 0; v < n; v++) {    // 遍历右侧顶点
        if(g[u][v] && !used[v]) {   // 如果存在边且右边顶点还没用过
            used[v] = 1;
            if(linker[v] == -1 || dfs(linker[v])) {     // 如果能找到非匹配点
                linker[v] = u;
                return 1;
            }
        }
    }
    return 0;
}

int hungary() {
    int res = 0;    // 最大匹配数
    memset(linker, -1, sizeof(linker));
    for(int u = 0; u < n; u++) {    // 从左边第1个顶点开始，寻找增广路
        memset(used, 0, sizeof(used));
        if(dfs(u)) {  // 如果找到增广路，则匹配数加一
            res++;
        }
    }
    return res;
}

int main() {
    int u, v, num;
    while(scanf("%d", &n) == 1) {
        memset(g, 0, sizeof(g));
        for(int i = 0; i < n; i++) {
            scanf("%d: (%d)", &u, &num);
            for(int j = 0; j < num; j++) {
                scanf("%d", &v);
                g[u][v] = 1;
            }
        }
        int result = hungary();
        printf("%d\n", n - result / 2);
    }
    return 0;
}
```

#### 3.3 课程和学生

**题目来源**

[HDOJ 1083 Girls and Boys](http://acm.hdu.edu.cn/showproblem.php?pid=1083)

**题目分析**

二分图最大匹配问题，直接套模板。值得注意的几点是：

1. 图从 0 开始为下标
2. 图必须对称的邻接矩阵存储
3. 模板返回的是匹配的点的数目，该数目需是课程数的两倍才满足题意

**实现代码**

```c++
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 500

int n;      // 二分图两个集合点数
int g[N][N];    // 采用邻接矩阵记录边
int linker[N];      // 记录右边匹配顶点及匹配边
int used[N];

int dfs(int u) {    // 寻找增广路
    for(int v = 0; v < n; v++) {    // 遍历右侧顶点
        if(g[u][v] && !used[v]) {   // 如果存在边且右边顶点还没用过
            used[v] = 1;
            if(linker[v] == -1 || dfs(linker[v])) {     // 如果能找到非匹配点
                linker[v] = u;
                return 1;
            }
        }
    }
    return 0;
}

int hungary() {
    int res = 0;    // 最大匹配数
    memset(linker, -1, sizeof(linker));
    for(int u = 0; u < n; u++) {    // 从左边第1个顶点开始，寻找增广路
        memset(used, 0, sizeof(used));
        if(dfs(u)) {  // 如果找到增广路，则匹配数加一
            res++;
        }
    }
    return res;
}

int main() {
    int T;
    scanf("%d", &T);
    while(T--) {
        int p, q;
        scanf("%d%d", &p, &q);
        memset(g, 0, sizeof(g));
        for(int i = 0; i < p; i++) {
            int num;
            scanf("%d", &num);
            for(int j = 0; j < num; j++) {
                int k;
                scanf("%d", &k);
                g[i][p + k - 1] = g[p + k - 1][i] = 1;
            }
        }
        n = p + q;
        int result = hungary();
        if(result == p * 2)
            printf("YES\n");
        else
            printf("NO\n");
    }
    return 0;
}
```

#### 3.4 单词游戏

**题目来源**

[HDOJ 1116 Play on Words](http://acm.hdu.edu.cn/showproblem.php?pid=1116)

**题目分析**

这道题的题意可转化为：假设有一张图，图中的点为从 a 到 z 的 26 个字母。如果输入一个单词，我们就以单词的首字母为起点，最后一个字母为终点，画一条有向边。如果最后由所有被访问到的点组成的子图为欧拉图或半欧拉图的话，我们可以解开题目。

那么构建子图之后，我们怎么判断该子图是否含有欧拉回路或欧拉通路呢？这里我们要把握两个充要条件，一个是连通图，一个是出度和入度的要求。前者我们可以通过并查集的思想实现，而后者我们只要记录每个点的出入度即可。

**实现代码**

```c++
#include <iostream>
#include <string>
#define MAX 30
using namespace std;

int T;
int N;
string s;
int father[MAX];
int visited[MAX], in[MAX], out[MAX];

void MakeSet(int x) {
    father[x] = x;
}

int Find(int x) {
    if(father[x] == x) {
        return x;
    } else {
        return Find(father[x]);
    }
}

void Union(int x, int y) {
    int xRoot = Find(x);
    int yRoot = Find(y);
    father[xRoot] = yRoot;
}

int main() {
    cin >> T;
    while(T--) {
        for(int i = 0; i < 26; i++) {
            visited[i] = 0;
            in[i] = 0;
            out[i] = 0;
            MakeSet(i);
        }
        cin >> N;
        for(int i = 1; i <= N; i++) {
            cin >> s;
            int u = s[0] - 'a';
            int v = s[s.length() - 1] - 'a';
            Union(u, v);
            out[u]++;
            in[v]++;
            visited[u] = visited[v] = 1;
        }
        int ans = 0;
        for(int i = 0; i < 26; i++) {
            if(visited[i] && father[i] == i) {
                ans++;
            }
        }
        if(ans > 1) {
            cout << "The door cannot be opened." << endl;
            continue;
        }
        int x = 0, y = 0, z = 0;
        for(int i = 0; i < 26; i++) {
            if(visited[i] && in[i] != out[i]) {
                if(in[i] - out[i] == 1) {
                    x++;
                } else if (in[i] - out[i] == -1) {
                    y++;
                } else {
                    z++;
                }
            }
        }
        if(!z && ((x == 0 && y == 0) || (x == 1 && y == 1))) {
            cout << "Ordering is possible." << endl;
        } else {
            cout << "The door cannot be opened." << endl;
        }
    }
}
```
#### 3.5 多少张桌子

**题目来源**

[HDOJ 1213 How Many Tables](http://acm.hdu.edu.cn/showproblem.php?pid=1213)

**题目分析**

这是我做的第一道并查集题目。这道题比较简单，也比较典型，因此解决的方法是非常传统的方法，并不需要什么特别技巧。关于并查集的内容，可以查看并查集的[维基百科](https://zh.wikipedia.org/wiki/%E5%B9%B6%E6%9F%A5%E9%9B%86)。

**实现代码**

```c++
#include <iostream>
#define MAX 1005
using namespace std;

int T;
int N, M;
int father[MAX];

void MakeSet(int x) {
    father[x] = x;
}

int Find(int x) {
    if(father[x] == x) {
        return x;
    } else {
        return Find(father[x]);
    }
}

void Union(int x, int y) {
    int xRoot = Find(x);
    int yRoot = Find(y);
    father[xRoot] = yRoot;
}

int main() {
    cin >> T;
    while(T--) {
        cin >> N >> M;
        int ans = N, a, b;
        for(int i = 1; i <= N; i++) {
            MakeSet(i);
        }
        while(M--) {
            cin >> a >> b;
            if(Find(a) != Find(b)) {
                Union(a, b);
                ans--;
            }
        }
        cout << ans << endl;
    }
}
```

### 第 4 章 动态规划

#### 4.1 最大连续子序列和

**题目来源**

[HDOJ 1003 Max Sum](http://acm.hdu.edu.cn/showproblem.php?pid=1003)

**题目分析**

这道题目算是动态规划题目里面比较简单的了。动态规划很重要的一步是找出动态转移方程，这里的动态转移方程是：`sum[i] = max{sum[i-1]+a[i],a[i]}`，根据这道方程，我们可以在 O(n) 的复杂度下求得各个节点的 sum 值，从而得到最大连续子序列和。

**实现代码**

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

#### 4.2 星河战队

**题目来源**

[HDOJ 1011 Starship Troopers](http://acm.hdu.edu.cn/showproblem.php?pid=1011)

**题目分析**

这道题比上一道题复杂了很多，它不仅是动态规划，而且是树形的动态规划，可以说是01背包问题的升级版。但只要我们围绕着动态规划的核心思想出发，就能够解决问题。首先，问题可以定义为在第u个点使用j个士兵能得到大脑的最大可能性，接着根据题意得到状态转移方程：`value[u][j] = max{value[v][j], value[v][j-k]+value[v][k]}`，`value[u][j]`表示的是第 u 个点使用j个士兵能得到大脑的最大可能性，v 表示的是 u 的所有子节点，最后得到的`value[1][m]`就是所求的最大可能性。由于需要从子节点推出父节点的值，所以整体采用后序遍历的方式。另外，`m=0`是一种特殊的情况，需要特殊考虑。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#define N 101

typedef struct {
    int bugs;
    int brains;
} room;

int n, m;
room rooms[N];
int matrix[N][N];
int visited[N];
int value[N][N];

void dpTree(int u) {
    int r;
    visited[u] = 1;
    r = (rooms[u].bugs + 19) / 20;
    for(int i = m; i >= r; i--) {
        value[u][i] = rooms[u].brains;
    }
    for(int v = 1; v <= n; v++) {
        if(matrix[u][v] && !visited[v]) {
            dpTree(v);
            for(int j = m; j >= r; j--) {
                for(int k = 1; k <= j - r; k++) {
                    if(value[u][j - k] + value[v][k] > value[u][j]) {
                        value[u][j] = value[u][j - k] + value[v][k];
                    }
                }
            }
        }
    }
}

int main() {
    while(scanf("%d %d", &n, &m) == 2 && n >= 0 && m >= 0) {
        memset(rooms, 0, sizeof(rooms));
        memset(matrix, 0, sizeof(matrix));
        memset(visited, 0, sizeof(visited));
        memset(value, 0, sizeof(value));
        for(int i = 1; i <= n; i++) {
            scanf("%d %d", &rooms[i].bugs, &rooms[i].brains);
        }
        for(int i = 1; i <= n - 1; i++) {
            int x, y;
            scanf("%d %d", &x, &y);
            matrix[x][y] = matrix[y][x] = 1;
        }
        if(m == 0) {
            printf("0\n");
        } else {
            dpTree(1);
            printf("%d\n", value[1][m]);
        }
    }
    return 0;
}
```

#### 4.3 最大连续子序列和升级版

**题目来源**

[HDOJ 1024 Max Sum Plus Plus](http://acm.hdu.edu.cn/showproblem.php?pid=1024)

**题目分析**

这道题比上道题难了很多，它不仅要考虑状态转移方程，而且还要考虑到优化问题。接下来让我们一起来解决这个问题：

1. 首先我们找到状态转移方程，即`s[i][j] = max{s[i][j-1] + num[j], s[i-1][k] + num[j]}`，其中`s[i][j]`表示 i 个不相交子段以及 j 个数下的最大值，max函数中的第一个表达式表示第 i 个子段包含`num[j]`，而后一个表达式表示不包含，k的取值范围为 i-1 到 j-1。最后所求的最大值为`max{s[m][j]}`
2. 如果直接按这个状态转移方程进行计算的话，时间复杂度和空间复杂度都会太大。我们通过观察状态转移方程可以看到，整个状态转移方程只涉及到 i 和 i-1 两种情况，所以我们可以把状态转移方程简化为：`s[1][j] = max{s[1][j-1] + num[j], s[0][k] + num[j]}`
3. 这样子，空间复杂度直接降了一个维度，但时间复杂度没降，依旧为O(m\*n\*n)。通过观察，我们发现只能处理最后一个维度，即对 k 的遍历。为了消灭这个维度，我们可以创建一个数组max[j]，保存上一个子段值计算时到 j 的最大值。通过这个操作，我们也可以直接把 s 这个二维数组降成一维

**实现代码**

```c
#include<stdio.h>
#include<string.h>
#define N 1000001

int m, n;
int num[N];
int s[N];
int max[N];

int main() {
    while(scanf("%d%d", &m, &n) == 2) {
        for(int i = 1; i <= n; i++) {
            scanf("%d", &num[i]);
        }
        memset(s, 0, sizeof(s));
        memset(max, 0, sizeof(max));
        for(int i = 1; i <= m; i++) {
            for(int j = i; j <= n; j++) {
                s[j] = s[j - 1] + num[j];
                if(max[j - 1] + num[j] > s[j]) {
                    s[j] = max[j - 1] + num[j];
                }
            }
            max[i] = s[i];
            for(int j = i + 1; j <= n; j++) {
                max[j] = max[j - 1];
                if(s[j] > max[j]) {
                    max[j] = s[j];
                }
            }
        }
        printf("%d\n", max[n]);
    }
    return 0;
}
```

#### 4.4 道路问题

**题目来源**

[HDOJ 1025 Constructing Roads In JGShining's Kingdom](http://acm.hdu.edu.cn/showproblem.php?pid=1025)

**题目分析**

这道题稍微分析一下，就可以判断出它是属于LIS问题，即最长递增子串问题。对于 LIS 问题，主要的解决方法有时间复杂度为 O(n\*n) 的动态规划法和时间复杂度为为 O(n\*logn) 的 LIS 算法。

由于本题数据量较大，所以第一种方法会出现 TLE 的情况，但我们也稍微提一下。所谓动态规划，最重要的是要找到状态转移方程。这道题的状态转移方程是：`s[i] = max{s[k]} + 1, 0 <= k < i, num[k] < num[i]`。这个状态转移方程比较好理解，代码也比较好些，但由于此题不可行，所以并不赘述。

另一种方法是 LIS 算法，它的具体思路是增加一个数组 s，s[i] 表示长度为 i 的最长子序列的最后一个数最小可以是多少。然后对输入数组进行遍历，如果该数大于最后一个元素，则进行添加，否则替换数组中第一个大于该元素的元素。对于这个替换操作，我们可以用二分查找来降低时间复杂度。实际上，整个思路很接近于对栈的操作。

**实现代码**

```c
#include<stdio.h>
#include<algorithm>
#define N 500001
using namespace std;

typedef struct {
    int x;
    int y;
} road;

int n, m;
road roads[N];
int s[N];

int cmp(road r1, road r2) {
    return r1.x < r2.x;
}

int main() {
    int nCases = 1;
    while(scanf("%d", &n) == 1) {
        for(int i = 0; i < n; i++) {
            scanf("%d%d", &roads[i].x, &roads[i].y);
        }
        sort(roads, roads + n, cmp);
        s[0] = 0;
        s[1] = roads[0].y;
        m = 1;
        for(int i = 1; i < n; i++) {
            if(roads[i].y > s[m]) {
                s[++m] = roads[i].y;
            } else {
                int low = 0, high = m;
                int mid = low + (high - low) / 2;
                while(low < high - 1) {
                    if(roads[i].y > s[mid]) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                    mid = low + (high - low) / 2;
                }
                s[high] = roads[i].y;
            }
        }
        printf("Case %d:\nMy king, at most %d road", nCases++, m);
        if(m != 1) {
            printf("s");
        }
        printf(" can be built.\n\n");
    }
    return 0;
}
```

#### 4.5 丑数

**题目来源**

[HDOJ 1058 Humble Numbers](http://acm.hdu.edu.cn/showproblem.php?pid=1058)

**题目分析**

这道题的状态转移方程是：`ans[k] = min(ans[m] * 2, ans[n] * 3, ans[p] * 5, ans[q] * 7)`，其中 m, n, p, q 的初始值为 1，并且只有在被选中后才移动。

**实现代码**

```c
#include<stdio.h>

int ans[5900];

int min(int a, int b) {
    if(a < b) {
        return a;
    }
    return b;
}

void init() {
    int m, n, p, q;
    m = n = p = q = 1;
    ans[1] = 1;
    for(int i = 2; i <= 5842; i++) {
        ans[i] = min(ans[m] * 2, min(ans[n] * 3, min(ans[p] * 5, ans[q] * 7)));
        if(ans[i] == ans[m] * 2) {
            m++;
        }
        if(ans[i] == ans[n] * 3) {
            n++;
        }
        if(ans[i] == ans[p] * 5) {
            p++;
        }
        if(ans[i] == ans[q] * 7) {
            q++;
        }
    }
}

void output(int n) {
    printf("The %d", n);
    int last = n % 100;
    if(last == 13 || last == 12 || last == 11) {
        printf("th humble number is %d.\n", ans[n]);
        return ;
    }
    last = n % 10;
    if(last == 1)
        printf("st");
    else if(last == 2)
        printf("nd");
    else if(last == 3)
        printf("rd");
    else
        printf("th");
    printf(" humble number is %d.\n", ans[n]);
}

int main() {
    int n;
    init();
    while(scanf("%d", &n) && n) {
        output(n);
    }
    return 0;
}
```

#### 4.6 划分物品

**题目来源**

[HDOJ 1059 Dividing](http://acm.hdu.edu.cn/showproblem.php?pid=1059)

**题目分析**

这是一道多重背包问题，即判断能否装满容量为总价值一半的背包。但如果直接采用多重背包转 01 背包的做法，则会出现 TLE，所以我们必须采用二进制优化。所谓二进制优化，即将物品容量按 2 的 k 次方进行分割。之所以可以这样做，是因为一个正整数 n 可以被分解成 $$1, 2, 4, …, 2^{k-1}, n-2^k+1$$（$$k$$ 是满足 $$n - 2^k+1>0$$的最大整数）的形式，且 1～n 之内的所有整数均可以唯一表示成 $$1, 2, 4, …, 2^{k-1}, n-2^k+1$$ 中某几个数的和的形式。假设转换后的新数组为 value0，则状态转移方程为`f[k] = f[k] || f[k - value0[i]]`。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>

int value[6], value0[100000];
int sum, isAll;
int *f;

int main() {
    int count = 0;
    while(1) {
        sum = 0;
        isAll = 1;
        count++;
        for(int i = 0; i < 6; i++) {
            scanf("%d", &value[i]);
            if(value[i] != 0) {
                isAll = 0;
            }
            sum += value[i] * (i + 1);
        }
        if(isAll) {
            break;
        }
        if(sum % 2 == 1) {
            printf("Collection #%d:\n", count);
            printf("Can't be divided.\n\n");
        } else {
            f = (int *)malloc(sizeof(int) * (sum / 2 + 1));
            f[0] = 1;
            for(int i = 1; i <= sum / 2; i++) {
                f[i] = 0;
            }
            int id = 0;
            for(int i = 0; i < 6; i++) {
                for(int j = 1; j <= value[i]; j *= 2) {
                    value0[id++] = j * (i + 1);
                    value[i] -= j;
                }
                if(value[i] > 0) {
                    value0[id++] = value[i] * (i + 1);
                }
            }
            for(int i = 0; i < id; i++) {
                for(int k = sum / 2; k >= value0[i]; k--) {
                    if(!f[k] && f[k - value0[i]]) {
                        f[k] = 1;
                    }
                    if(f[sum / 2] == 1) {
                        break;
                    }
                }
            }
            printf("Collection #%d:\n", count);
            if(f[sum / 2]) {
                printf("Can be divided.\n\n");
            } else {
                printf("Can't be divided.\n\n");
            }
        }
    }
    return 0;
}
```

#### 4.7 猴子和香蕉

**题目来源**

[HDOJ 1069 Monkey and Banana](http://acm.hdu.edu.cn/showproblem.php?pid=1069)

**题目分析**

每一种砖块有三种摆法，所以实际上可选择的砖块总共有 3 \* n 块，按长度进行排序后，用动态规划的思维求出严格下降序列即可。

**实现代码**

```c++
#include <iostream>
#include <algorithm>
using namespace std;

int n;
int **block = new int* [100];
int *result = new int[100];

bool cmp(int *b1, int *b2) {
    return b1[1] > b2[1];
}

int main() {
    int cnt = 0;
    while(cin >> n) {
        if(n == 0) {
            break;
        }
        for(int i = 0; i < n; i++) {
            int temp[3];
            cin >> temp[0] >> temp[1] >> temp[2];
            for(int j = 0; j < 3; j++) {
                block[3 * i + j] = new int[3];
                block[3 * i + j][0] = temp[j];
                block[3 * i + j][1] = max(temp[(j + 2) % 3], temp[(j + 1) % 3]);
                block[3 * i + j][2] = min(temp[(j + 2) % 3], temp[(j + 1) % 3]);
            }
        }
        sort(block, block + 3 * n, cmp);
        for(int i = 0; i < n * 3; i++) {
            result[i] = block[i][0];
        }
        for(int i = 1; i < n * 3; i++) {
            for(int j = 0; j < i; j++) {
                if(block[i][2] < block[j][2] && block[i][1] < block[j][1]) {
                    if(result[i] < result[j] + block[i][0]) {
                        result[i] = result[j] + block[i][0];
                    }
                }
            }
        }
        int ans = 0;
        for(int i = 0; i < n * 3; i++) {
            if(result[i] > ans) {
                ans = result[i];
            }
        }
        cout << "Case " << ++cnt << ": maximum height = " << ans << endl;
    }
    return 0;
}
```

#### 4.8 做作业

**题目来源**

[HDOJ 1074 Doing Homework](http://acm.hdu.edu.cn/showproblem.php?pid=1074)

**题目分析**

这是集合上的 DP 问题，所以也就是状态压缩动态规划问题。由于最多 15 门功课的全排列的时间复杂度过大，所以我们用二进制数 i 表示做作业的情况。状态 i 总共有 15 个二进制位，1 表示已做，0 表示未做。对于状态 i，枚举当前的作业 j，如果`i & (1 << j)`为真，则表示当前状态含有作业 j。我们通过`t ^= (1 << j)`可以还原出还没做 j 作业之前的状态，这样就有两个状态了。在进行动态规划转移的时候，我们要记录当前状态的日期以及对应做的作业 j。由于题目要求按字典序输出，所以我们应该倒序循环当前作业 j。

**实现代码**

```c++
#include <iostream>
#include <string>
#include <cstring>
using namespace std;

const int maxn = 1 << 15;
const int inf = (1 << 31) - 1;

int T, N;
int C[20], D[20];
int dp[maxn], day[maxn], pre[maxn];
string S[20];

void output(int x) {
    if(!x) {
        return;
    }
    output(x ^ (1 << pre[x]));
    cout << S[pre[x]] << endl;
}

int main() {
    cin >> T;
    while(T--) {
        cin >> N;
        int bit = 1 << N;
        for(int i = 0; i < N; i++) {
            cin >> S[i] >> D[i] >> C[i];
        }
        for(int i = 1; i < bit; i++) {
            dp[i] = inf;
            for(int j = N - 1; j >= 0; j--) {
                int t = 1 << j;
                int reduce;
                if(!(i & t)) {
                    continue;
                }
                t = i ^ t;
                reduce = day[t] + C[j] - D[j];
                reduce = reduce < 0 ? 0 : reduce;
                if(dp[t] + reduce < dp[i]) {
                    dp[i] = dp[t] + reduce;
                    day[i] = day[t] + C[j];
                    pre[i] = j;
                }
            }
        }
        cout << dp[bit - 1] << endl;
        output(bit - 1);
        memset(pre, 0, sizeof(pre));
    }
    return 0;
}
```

#### 4.9 肥鼠和奶酪

**题目来源**

[HDOJ 1078 FatMouse and Cheese](http://acm.hdu.edu.cn/showproblem.php?pid=1078)

**题目分析**

这题属于动态规划里面的记忆化搜索。记忆化搜索的主要特点是自顶向下，所以这里因为终点未知，故以 (0, 0) 点为终点逆向深度优先搜索。

**实现代码**

```c++
#include <iostream>
#include <algorithm>
using namespace std;

int n, k;
int s[105][105], dp[105][105];
int dx[4] = {1, -1, 0, 0};
int dy[4] = {0, 0, 1, -1};

int isOK(int x, int y) {
    if(x >= 0 && x < n && y >= 0 && y < n) {
        return 1;
    }
    return 0;
}

int dfs(int x, int y) {
    if(dp[x][y]) {
        return dp[x][y];
    }
    int ans = 0;
    for(int i = 1; i <= k; i++) {
        for(int j = 0; j < 4; j++) {
            int newx = x + dx[j] * i;
            int newy = y + dy[j] * i;
            if(isOK(newx, newy) && s[newx][newy] > s[x][y]) {
                ans = max(ans, dfs(newx, newy));
            }
        }
    }
    dp[x][y] = s[x][y] + ans;
    return dp[x][y];
}

int main() {
    while(1) {
        cin >> n >> k;
        if(n == -1 && k == -1) {
            break;
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                cin >> s[i][j];
                dp[i][j] = 0;
            }
        }
        cout << dfs(0, 0) << endl;
    }
    return 0;
}
```

#### 4.10 人类基因函数

**题目来源**

[HDOJ 1080 Human Gene Functions](http://acm.hdu.edu.cn/showproblem.php?pid=1080)

**题目分析**

这题是最长公共子序列（LCS）的变形题。我们可以得到状态转移方程：`dp[i][j] = max(dp[i - 1][j - 1] + val(a[i], b[j]), max(dp[i - 1][j] + val(a[i], '-'), dp[i][j - 1] + val(b[j], '-')))`。初始状态为`dp[i][0] = dp[i - 1][0] + val(a[i], '-')`。

**实现代码**

```c++
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#define CLR(a, b) memset(a, (b), sizeof(a))
using namespace std;
typedef long long LL;
const int MAXN = 1e6 +10;
const int INF = 0x3f3f3f3f;
char a[110], b[110];
int dp[110][110];
int val(char x, char y) {
    if(y != '-' && x > y) swap(x, y);
    if(x == 'A') {
        if(y == 'A') return 5;
        if(y == 'C') return -1;
        if(y == 'G') return -2;
        if(y == 'T') return -1;
        return -3;
    }
    if(x == 'C') {
        if(y == 'C') return 5;
        if(y == 'G') return -3;
        if(y == 'T') return -2;
        return -4;
    }
    if(x == 'G') {
        if(y == 'G') return 5;
        if(y == 'T') return -2;
        return -2;
    }
    if(x == 'T') {
        if(y == 'T') return 5;
        return -1;
    }
}
int main()
{
    int t; scanf("%d", &t);
    while(t--) {
        int n, m;
        scanf("%d%s", &n, a+1);
        scanf("%d%s", &m, b+1);
        CLR(dp, -INF); dp[0][0] = 0;
        for(int i = 1; i <= n; i++) {
            dp[i][0] = dp[i-1][0] + val(a[i], '-');
        }
        for(int i = 1; i <= m; i++) {
            dp[0][i] = dp[0][i-1] + val(b[i], '-');
        }
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++) {
                dp[i][j] = max(dp[i-1][j-1] + val(a[i], b[j]), max(dp[i-1][j] + val(a[i], '-'), dp[i][j-1] + val(b[j], '-')));
            }
        }
        printf("%d\n", dp[n][m]);
    }
    return 0;
}
```

#### 4.11 最大子矩阵

**题目来源**

[HDOJ 1081 To The Max](http://acm.hdu.edu.cn/showproblem.php?pid=1081)

**题目分析**

此题实际上的将一维的最长子序列和问题扩展成二维的最大子序列问题，因此可以枚举起始行 i 和起始行 j，然后把从第 i 行的值到第 j 行的值压缩成一行，就可以将二维问题转换为一维问题。这种思路的时间复杂度为 O(n^3)，为了降低复杂度，可以用前缀和进行压缩，即提前计算并存储每一列的前缀和。

**实现代码**

```c++
#include<iostream>
#include<cstdio>
#include<stdio.h>
#include<cstdlib>
#include<stdlib.h>
#include<algorithm>
#include<string.h>
#include<cstring>
using namespace std;
 
const int maxn=150;
const int INF=0x3f3f3f3f;
int sum[maxn][maxn];
int a[maxn][maxn],n;
 
void debug(){
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            printf("%d%c",sum[i][j],j==n?'\n':' ');
}
 
int main(){
    //freopen("input.txt","r",stdin);
    while(scanf("%d",&n)!=EOF){
        memset(sum,0,sizeof(sum));
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                scanf("%d",&a[i][j]);
        for(int j=1;j<=n;j++){
            sum[1][j]=a[1][j];
            for(int i=2;i<=n;i++)
                sum[i][j]=sum[i-1][j]+a[i][j];
        }
        //debug();
        int ans=-INF;
        for(int i=1;i<=n;i++)
            for(int j=i;j<=n;j++){
                int tmp=0,tmpans=-INF;
                for(int k=1;k<=n;k++){
                    int num=sum[j][k]-sum[i-1][k];
                    if (tmp+num>=0) tmp+=num;
                    else tmp=num;
                    tmpans=max(tmpans,tmp);
                }
                ans=max(ans,tmpans);
            }
        printf("%d\n",ans);
    }
    return 0;
}
```

#### 4.12 小猪银行

**题目来源**

[HDOJ 1114 Piggy Bank](http://acm.hdu.edu.cn/showproblem.php?pid=1114)

**题目分析**

每种硬币可用多次，所以这是完全背包问题。另外这也是一个恰好装满的问题，所以在初始化的时候除了第 0 个元素，其他都必须初始化为 INT_MAX，因为这里是求最小值。值得注意的是，在循环过程中，必须判断值是否为 INT_MAX，避免溢出。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>

int T;
int E, F, N;
int *p, *w, *f;

int main() {
    scanf("%d", &T);
    while(T--) {
        scanf("%d%d", &E, &F);
        f = (int *)malloc(sizeof(int) * (F - E + 1));
        f[0] = 0;
        for(int i = 1; i <= F - E; i++) {
            f[i] = INT_MAX;
        }
        scanf("%d", &N);
        p = (int *)malloc(sizeof(int) * (N + 1));
        w = (int *)malloc(sizeof(int) * (N + 1));
        for(int i = 1; i <= N; i++) {
            scanf("%d%d", &p[i], &w[i]);
        }
        for(int i = 1; i <= N; i++) {
            for(int j = w[i]; j <= F - E; j++) {
                if(f[j - w[i]] != INT_MAX) {
                    f[j] = f[j] < f[j - w[i]] + p[i] ? f[j] : f[j - w[i]] + p[i];
                }
            }
        }
        if(f[F - E] == INT_MAX) {
            printf("This is impossible.\n");
        } else {
            printf("The minimum amount of money in the piggy-bank is %d.\n", f[F - E]);
        }
    }
    return 0;
}
```

#### 4.13 我需要一个 Offer

**题目来源**

[HDOJ 1203 I NEED A OFFER!](http://acm.hdu.edu.cn/showproblem.php?pid=1203)

**题目分析**

这题本质上是 01 背包问题，只不过我们需要修改其中一些细节。在这道题中，状态转移方程是`f[v] = min(f[v], f[v- a[i]] * (1 - b[i]))`，其中`f[v]`表示的是当资金为v的时候没有Offer的概率，因为最后要求拿到Offer的最大概率，所以这里求的是最小概率。另外，`f[v]`的所有元素需初始化为 1，因为刚开始的时候没有学校可选择，所以无论资金多少，拿到Offer的概率都为 0。此题还有两个小细节：一个是无法使用`memset`函数，因为其只适用于 int 类型的数组，另一个细节是要输出`%`，格式应为`%%`。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int n, m;
int *a;
double *b;
double *f;

int main() {
    while(scanf("%d%d", &n, &m) == 2) {
        if(m == 0 && n == 0) {
            break;
        }
        a = (int *)malloc(sizeof(int) * (m + 1));
        b = (double *)malloc(sizeof(double) * (m + 1));
        f = (double *)malloc(sizeof(double) * (n + 1));
        for(int i = 0; i <= n; i++) {
            f[i] = 1;
        }
        for(int i = 1; i <= m; i++) {
            scanf("%d%lf", &a[i], &b[i]);
        }
        for(int i = 1; i <= m; i++) {
            for(int j = n; j >= a[i]; j--) {
                f[j] = f[j] < f[j - a[i]] * (1 - b[i]) ? f[j] : f[j - a[i]] * (1 - b[i]);
            }
        }
        printf("%.1lf%%\n", (1 - f[n]) * 100);
    }
    return 0;
}
```

#### 4.14 ACboy 需要你的帮助

**题目来源**

[HDOJ 1712 ACboy needs your help](http://acm.hdu.edu.cn/showproblem.php?pid=1712)

**题目分析**

这实际上是一道分组背包的问题，其中每个课程内复习天数的价值为一组。只要用一个一维数组，三重循环就可以解决问题。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int n, m;
int **a;
int *dp;

int main() {
    while(scanf("%d%d", &n, &m) == 2) {
        if(m == 0 && n == 0) {
            break;
        }
        a = (int **)malloc(sizeof(int *) * (n + 1));
        for(int i = 0; i <= n; i++) {
            a[i] = (int *)malloc(sizeof(int) * (m + 1));
            memset(a[i], 0, sizeof(int) * (m + 1));
        }
        dp = (int *)malloc(sizeof(int) * (m + 1));
        memset(dp, 0, sizeof(int) * (m + 1));
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= m; j++) {
                scanf("%d", &a[i][j]);
            }
        }
        for(int i = 1; i <= n; i++) {
            for(int j = m; j >= 1; j--) {
                for(int k = 1; k <= j; k++) {
                    dp[j] = dp[j] > dp[j - k] + a[i][k] ? dp[j] : dp[j - k] + a[i][k];
                }
            }
        }
        printf("%d\n", dp[m]);
    }
}
```

#### 4.15 珍惜现在，感恩生活

**题目来源**

[HDOJ 2191 悼念512汶川大地震遇难同胞——珍惜现在，感恩生活](http://acm.hdu.edu.cn/showproblem.php?pid=2191)

**题目分析**

表面上物品是多件的，但因为件数有限，所以我们可以用 01 背包的思路解决。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int C;
int n, m;
int *p, *h, *c;
int *f;

int main() {
    scanf("%d", &C);
    while(C--) {
        scanf("%d%d", &n, &m);
        p = (int *)malloc(sizeof(int) * (m + 1));
        h = (int *)malloc(sizeof(int) * (m + 1));
        c = (int *)malloc(sizeof(int) * (m + 1));
        f = (int *)malloc(sizeof(int) * (n + 1));
        memset(f, 0, (n + 1)*sizeof(int));
        for(int i = 1; i <= m; i++) {
            scanf("%d%d%d", &p[i], &h[i], &c[i]);
        }
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= c[i]; j++) {
                for(int k = n; k >= p[i]; k--) {
                    f[k] = f[k] > f[k - p[i]] + h[i] ? f[k] : f[k - p[i]] + h[i];
                }
            }
        }
        printf("%d\n", f[n]);
    }
    return 0;
}
```

#### 4.16 饭卡

**题目来源**

[HDOJ 2546 饭卡](http://acm.hdu.edu.cn/showproblem.php?pid=2546)

**题目分析**

“每种菜可购买一次”，而且还是求最值问题，很容易让我们联想到 01 背包问题。我们可以记录所有菜中最贵的一种，把它放在最后买，然后用 01 背包求（m-5）元钱可买的菜的最大金额，最后`m-最大金额-最贵菜价格`即为所求，状态转移方程是`f[v] = max(f[v], f[v - s[i]] + s[i])`。值得注意的是，`m < 5`的特殊情况需要特别考虑。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int n, m;
int *s, *f;

int main() {
    while(scanf("%d", &n) == 1) {
        if(n == 0) {
            break;
        }
        s = (int *)malloc(sizeof(int) * (n + 1));
        int max = -1, maxId = 0;
        for(int i = 1; i <= n; i++) {
            scanf("%d", &s[i]);
            if(s[i] > max) {
                max = s[i];
                maxId = i;
            }
        }
        scanf("%d", &m);
        if(m < 5) {
            printf("%d\n", m);
        } else {
            f = (int *)malloc(sizeof(int) * (m - 4));
            for(int i = 0; i <= m - 5; i++) {
                f[i] = 0;
            }
            for(int i = 1; i <= n; i++) {
                for(int j = m - 5; j >= s[i]; j--) {
                    if(i == maxId) {
                        continue;
                    }
                    f[j] = f[j] > f[j - s[i]] + s[i] ? f[j] : f[j - s[i]] + s[i];
                }
            }
            printf("%d\n", m - f[m - 5] - max);
        }
    }
    return 0;
}
```

#### 4.17 拾骨者

**题目来源**

[HDOJ 2602 Bone Collector](http://acm.hdu.edu.cn/showproblem.php?pid=2602)

**题目分析**

典型的 01 背包问题，直接解决即可。

**实现代码**

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int t, n, v;
int *c, *w;
int *f;

int main() {
    scanf("%d", &t);
    while(t--) {
        scanf("%d%d", &n, &v);
        c = (int *)malloc(sizeof(int) * (n + 1));
        w = (int *)malloc(sizeof(int) * (n + 1));
        f = (int *)malloc(sizeof(int) * (v + 1));
        memset(f, 0, (v + 1) * sizeof(int));
        for(int i = 1; i <= n; i++) {
            scanf("%d", &w[i]);
        }
        for(int i = 1; i <= n; i++) {
            scanf("%d", &c[i]);
        }
        for(int i = 1; i <= n; i++) {
            for(int j = v; j >= c[i]; j--) {
                f[j] = f[j] > f[j - c[i]] + w[i] ? f[j] : f[j - c[i]] + w[i];
            }
        }
        printf("%d\n", f[v]);
    }
    return 0;
}
```

### 第 5 章 贪心算法

#### 5.1 肥鼠交易

**题目来源**

[HDOJ 1009 FatMouse' Trade](http://acm.hdu.edu.cn/showproblem.php?pid=1009)

**题目分析**

这道题目属于很简单的贪心算法题目，能够帮助我们很好地理解贪心算法。得到换算数组之后，我们可以求出每一个房间换算的比率，然后按照比率从大到小的方式进行换算。

**实现代码**

```c++
#include<stdio.h>
#include<stdlib.h>

typedef struct {
    int j;
    int f;
    double ratio;
} room;

int main() {
    int m, n;
    room *rooms;
    while(scanf("%d %d", &m, &n) == 2 && m != -1 && n != -1) {
        rooms = (room *)malloc(sizeof(room) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d %d", &rooms[i].j, &rooms[i].f);
            rooms[i].ratio = (double)rooms[i].j / rooms[i].f;
        }
        for(int i1 = 0; i1 < n - 1; i1++) {
            for(int i2 = 0; i2 < n - i1 - 1; i2++) {
                if(rooms[i2].ratio < rooms[i2 + 1].ratio) {
                    room temp = rooms[i2];
                    rooms[i2] = rooms[i2 + 1];
                    rooms[i2 + 1] = temp;
                }
            }
        }
        double sum = 0;
        for(int i = 0; i < n; i++) {
            if(m <= 0) {
                break;
            } else if(m > rooms[i].f) {
                m -= rooms[i].f;
                sum += rooms[i].j;
            } else {
                sum += m * rooms[i].ratio;
                break;
            }
        }
        printf("%.3lf\n", sum);
    }
    return 0;
}
```

#### 5.2 移动桌子

**题目来源**

[HDOJ 1050 Moving Tables](http://acm.hdu.edu.cn/showproblem.php?pid=1050)

**题目分析**

这是很经典的贪心算法问题，我们可以设一个数组存储桌子从每个门口经过的次数，最后遍历数组，记录最大经过次数，即可得到最少时间。

**实现代码**

```c
#include<stdio.h>
#include<string.h>
#define N 200

int main() {
    int t;
    int s[N];
    scanf("%d", &t);
    while(t--) {
        memset(s, 0, sizeof(s));
        int n;
        scanf("%d", &n);
        while(n--) {
            int a, b;
            scanf("%d%d", &a, &b);
            if(a > b) {
                int t = a;
                a = b;
                b = t;
            }
            for(int i = (a - 1) / 2; i <= (b - 1) / 2; i++) {
                s[i]++;
            }
        }
        int max = 0;
        for(int i = 0; i < N; i++) {
            if(s[i] > max) {
                max = s[i];
            }
        }
        printf("%d\n", max * 10);
    }
    return 0;
}
```

#### 5.3 木棍

**题目来源**

[HDOJ 1051 Wooden Sticks](http://acm.hdu.edu.cn/showproblem.php?pid=1051)

**题目分析**

简单的贪心算法题，对结构体进行排序后求出 LIS 的个数。

**实现代码**

```c++
#include<stdio.h>
#include<algorithm>
#define N 5000
using namespace std;

struct stick {
    int l;
    int w;
    int flag;
};

int t, n;
stick s[N];

bool cmp(stick x, stick y) {
    if(x.l == y.l)
        return x.w < y.w;
    else
        return x.l < y.l;
}

int main() {
    scanf("%d", &t);
    while(t--) {
        scanf("%d", &n);
        for(int i = 0; i < n; i++) {
            scanf("%d%d", &s[i].l, &s[i].w);
            s[i].flag = 0;
        }
        sort(s, s + n, cmp);
        int result = 0;
        int isFound, pos;
        while(1) {
            isFound = 0;
            for(int i = 0; i < n; i++) {
                if(s[i].flag == 0) {
                    isFound = 1;
                    pos = i;
                    result++;
                    s[i].flag = 1;
                    break;
                }
            }
            if(isFound) {
                for(int i = pos + 1; i < n; i++) {
                    if(s[i].w >= s[pos].w && s[i].flag == 0) {
                        pos = i;
                        s[i].flag = 1;
                    }
                }
            } else {
                break;
            }
        }
        printf("%d\n", result);
    }
    return 0;
}
```

#### 5.4 哈夫曼编码

**题目来源**

[HDOJ 1053 Entropy](http://acm.hdu.edu.cn/showproblem.php?pid=1053)

**题目分析**

这是一道典型的哈夫曼编码问题，只要掌握哈夫曼编码的操作就可以解决了，但要特别注意编码字符串所含字符的种类只有 1 的情况。

**实现代码**

```c++
#include<stdio.h>
#include<string.h>
#include<algorithm>
#define N 1005
using namespace std;

int main() {
    char s[N];
    int a[N];
    while(scanf("%s", s) && strcmp(s, "END")) {
        for(int i = 0; i < N; i++) {
            a[i] = 1;
        }
        sort(s, s + strlen(s));
        int id = 0;
        char c = s[0];
        for(unsigned int i = 1; i < strlen(s); i++) {
            if(s[i] == c) {
                a[id]++;
            } else {
                id++;
                c = s[i];
            }
        }
        int n = id + 1;
        sort(a, a + n);
        int sum;
        if(n == 1) {
            sum = a[0];
        } else {
            sum = 0;
            for(int i = 0; i < n - 1; i++) {
                sum += a[i] + a[i + 1];
                a[i + 1] += a[i];
                int j = i + 1;
                while(j + 1 < n && a[j] > a[j + 1]) {
                    int t = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = t;
                    j++;
                }
            }
        }
        printf("%d %d %.1lf\n", strlen(s) * 8, sum, strlen(s) * 8 / (double)sum);
    }
    return 0;
}
```

### 第 6 章 数学

#### 6.1 阶乘位数

**题目来源**

[HDOJ 1018 Big Number](http://acm.hdu.edu.cn/showproblem.php?pid=1018)

**题目分析**

这道问题表面上可以进行暴力求解，即计算出一个数的阶乘之后再计算该阶乘的位数。但实际上这是很不现实的做法，首先它耗时较久，其次计算机需要通过字符串而不是整型变量来存储计算的阶乘值，这将大大增大编程难度。这个时候，数学就体现作用了，斯特林公式就可以解决这个问题。对于这道题，我们不需要记住那些复杂的公式，我们只要知道一个数 n 的阶乘位数为 log10(n!) 的值取整加 1，即`阶乘位数 = log10(1) + log10(2) + … + log10(n) + 1`。利用这条公式，我们就可以写出我们的程序了。这里需要注意的是，为了计算的精确度，sum 和遍历的 i 都必须为 double 类型，不然会出现 WA 的情况。

**实现代码**

```c++
#include<stdio.h>
#include<math.h>

int main() {
    int n, num;
    double sum;
    scanf("%d", &n);
    while(n--) {
        scanf("%d", &num);
        sum = 1;
        for(double i = 1; i <= num; i++) {
            sum += log10(i);
        }
        printf("%d\n", (int)sum);
    }
    return 0;
}
```

#### 6.2 伊格和公主（二）

**题目来源**

[HDOJ 1027 Ignatius and the Princess II](http://acm.hdu.edu.cn/showproblem.php?pid=1027)

**题目分析**

这道题相对来说比较简单。如果用全排列的思维的话，那很可能超时，所以我们使用 C++ 算法库下的 next_permutation 函数就可以解决问题了。值得注意的是，这个函数调用的次数应该为 m-1 次，因为初始的排列算是一次。

**实现代码**

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

#### 6.3 伊格和公主（三）

**题目来源**

[HDOJ 1028 Ignatius and the Princess III](http://acm.hdu.edu.cn/showproblem.php?pid=1028)

**题目分析**

这题属于母函数（也称生成函数）里面比较简单和基础的一个题目了，但由于笔者之前接触较少，所以也是查阅了一些资料方能解决。首先我们先要了解[母函数和排列组合之间的关系](http://www.cnblogs.com/dolphin0520/archive/2012/11/07/2755080.html)，然后我们就可以根据题意得到我们的母函数，接着在模拟多项式的展开就可以了。关于这类问题，网上已经有固定的模板作为参考了，具体可参考[这篇文章](http://www.cnblogs.com/syxchina/archive/2010/09/19/2197360.html)。本文也是借鉴了这篇文章里面的模板，也方便大家理解。

**实现代码**

```c++
#include<stdio.h>
#define lmax 10000

// c1是用来存放展开式的系数的，而c2则是用来计算时保存的，
// 用下标来控制每一项的位置，比如 c2[3] 就是 x^3 的系数。
// 用c1保存，然后在计算时用c2来保存变化的值。

int c1[lmax + 1], c2[lmax + 1];

int main() {
    int n, i, j, k ;
    // 计算的方法还是模拟手动运算，一个括号一个括号的计算，从前往后
    while(scanf("%d", &n) == 1) {
        // 对于 1+x+x^2+x^3+ 他们所有的系数都是 1
        // 而c2全部被初始化为0是因为以后要用到 c2[i] += x ;
        for (i = 0; i <= n; i++) {
            c1[i] = 1;
            c2[i] = 0;
        }
        //第一层循环是一共有n个小括号，而刚才已经算过一个了，所以是从2 到 n
        for (i = 2; i <= n; i++) {
            // 第二层循环是把每一个小括号里面的每一项，都要与前一个小括号里面的每一项计算。
            for (j = 0; j <= n; j++)
                // 第三层小括号是要控制每一项里面X增加的比例，这就是为什么要用 k+= i ;
                for (k = 0; k + j <= n; k += i) {
                    // 合并同类项，他们的系数要加在一起，所以是加法，
                    c2[ j + k] += c1[ j];
                }
            // 刷新一下数据，继续下一次计算，就是下一个括号里面的每一项。
            for ( j = 0; j <= n; j++ ) {
                c1[j] = c2[j] ;
                c2[j] = 0 ;
            }
        }
        printf("%d\n", c1[n]);
    }
    return 0;
}
```

#### 6.4 求 N^N 最高位

**题目来源**

[HDOJ 1060 Leftmost Digit](http://acm.hdu.edu.cn/showproblem.php?pid=1060)

**题目分析**

这道题数据量这么大，所以绝对不可能直接模拟解决。实际上，这道题用数学方法一下子就可以解决了，具体如下：

1. 令 $$M = N^N$$
2. 两边取对数，$$log_{10}M = N\*log_{10}N$$，得到 $$M = 10^{N*log_{10}N}$$
3. 令 $$N*log_{10}N = a（整数部分） + b（小数部分）$$，则 $$M = 10^{a+b} = 10^a * 10^b$$，由于 10 的整数次幂的最高位必定是 1，所以 M 的最高位只需考虑 $$10^b$$
4. 最后对 $$10^b$$ 取整，输出取整的这个数就行了（因为 $$0<=b<1$$，所以 $$1<=10^b<=10$$。对其取整，得到的一定是个位数，也就是所求的数）

**实现代码**

```c
#include<stdio.h>
#include<math.h>

int main() {
    int count, n;
    scanf("%d", &count);
    while(count--) {
        scanf("%d", &n);
        double d = n * log10(n);
        d = d - (long long)d;
        d = pow(10, d);
        printf("%d\n", (int)d);
    }
    return 0;
}
```

#### 6.5 阶乘最后非零位

**题目来源**

[HDOJ 1066 Last non-zero Digit in N!](http://acm.hdu.edu.cn/showproblem.php?pid=1066)

**题目分析**

这道题实际上是 ACM 中典型的套模板的题目，主要是要找到规律，具体的规律分析可以查看[这里](http://blog.sina.com.cn/s/blog_59e67e2c0100a7yx.html)。主要掌握了规律，我们就可以直接套模板解决。由于这道题数据量较大，所以要采用到大数除法。

**实现代码**

```c
#include<stdio.h>
#include<string.h>

int mod[20] = {1, 1, 2, 6, 4, 2, 2, 4, 2, 8, 4, 4, 8, 4, 6, 8, 8, 6, 8, 2};
char n[1000];
int a[1000];

int main() {
    int i, c, t, len;
    while(scanf("%s", n) != EOF) {
        t = 1;
        len = strlen(n);
        for(i = 0; i < len; i++)
            a[i] = n[len - 1 - i] - '0';
        while(len) {
            len -= !a[len - 1];
            t = t * mod[a[1] % 2 * 10 + a[0]] % 10;
            for(c = 0, i = len - 1; i >= 0; i--)
                c = c * 10 + a[i], a[i] = c / 5, c %= 5;
        }
        printf("%d\n", t);
    }
    return 0;
}
```

### 第 7 章 非主流

#### 7.1 圆环套玩具游戏

**题目来源**

[HDOJ 1007 Quoit Design](http://acm.hdu.edu.cn/showproblem.php?pid=1007)

**题目分析**

这道题本质上是一道最近点对问题。如果采用暴力求解的方法的话，会导致 TLE，所以我们必须采用分治的方法解决。具体的做法是将所给平面上 n 个点的集合 S 分成两个子集 S1 和 S2，每个子集中约有 n/2 个点，然后在每个子集中递归地求最接近的点对。在这里，一个关键的问题是如何实现分治法中的合并步骤，即由 S1 和 S2 的最接近点对，如何求得原集合 S 中的最接近点对。如果这两个点分别在 S1 和 S2 中，问题就变得复杂了。

对于这个关键点，我们处理的方式是先将集合进行排序，然后再分别求得两个集合中的最小距离 d1 和 d2，设 d1 和 d2 的最小值为 mindist。如果两个点分别存在两个不同的集合中，则两个点与中心点的横坐标之差和纵坐标之差均小于 mindist（不可能是等于，如果是等于，其中一个集合可求出），且其中一个点必定是分界点，而另一个点则必定在没有包含分界点的集合里。利用这一点，我们可以在得到 mindist 后，在可能范围内进行遍历，这样便可以大大减少计算量。

如果对于上面的解释不太清楚的话，可以查看[这里](http://www.cnblogs.com/hxsyl/p/3230164.html)。对于这个问题，还有几个关键的地方，具体如下：

1. 程序排序用到的是 C++ 的库文件 algorithm 里面的 sort 算法，原本尝试过 ANSI C 的 stdlib.h 头文件中的 qsort 算法，但效果不佳，会出现 TLE。所以这个程序需要在 C++ 环境下运行
2. 浮点数（float，double）是不存在完全相等的。我们可以用 eps（一般为 1e-6 或 1e-8），利用 fabs（abs 是整数取绝对值）判断范围是否小于 eps，从而判断浮点数是否相等

**实现代码**

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

### 第 8 章 搜索

#### 8.1 诱惑者的骨

**题目来源**

[HDOJ 1010 Tempter of the Bone](http://acm.hdu.edu.cn/showproblem.php?pid=1010)

**题目分析**

这道题是常见的递归题，相对来说比较简单，但有两点必须注意：

1. 递归实际上是效率很低的做法，操作不当甚至可能导致程序崩溃，所以在某些情况下，我们要尽可能避免使用递归。本题由于相对简单，所以为了思路清晰，我们依旧使用递归。但为了加强递归的效率，我们必须做一些适当的剪枝，这也是算法题中经常出现的。下面的程序并没有做太多的剪枝，网上提供的其他版本程序提供的剪枝很多，比如奇偶剪枝，即当出发地与目的地之间的距离与所给的时间奇偶性不同的话，那么肯定无法走出迷宫
2. 这道题的另一个关键点是回溯法。由于题意要求，每走完一步，方格就不可走，所以我们在进行试探前需将方格设为不可走，试探后必须回溯，将方格设为可走

**实现代码**

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

#### 8.2 素数环

**题目来源**

[HDOJ 1016 Prime Ring Problem](http://acm.hdu.edu.cn/showproblem.php?pid=1016)

**题目分析**

这道题也是一道典型的深度优先搜索问题，使用到了回溯法。我一开始是参考了全排的算法来进行程序的编写的，可不知为什么，程序提交一直 WA（如果后面后进展，我再提及）。所以我最终还是参考了图深度优先遍历的算法来进行实现。

对于这道题，用以下的代码就可以 AC 了，可如果要再进行优化，可以通过以下几点进行考虑：

1. 由于涉及到素数判断很少，最大的判断数也就是 18 + 19 = 37 而已，所以可以通过建立素数表，用查表的方式来加快素数判断
2. 如果输入的 n 是奇数的话，那么 1 - n 之间一共有 n / 2 个偶数，n / 2 + 1 个奇数，也就是奇数比偶数多一个。那么把这 n 个数排成一个环，根据鸽巢原理，必然两个奇数相邻，而奇数之和是偶数，偶数不是素数，所以得出结论：如果 n 是奇数，则没有满足条件的排列。通过这一点，当 n 是奇数时，我们直接返回即可，这样可大大减少计算量

**实现代码**

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

#### 8.3 伊格和公主（一）

**题目来源**

[HDOJ 1026 Ignatius and the Princess I](http://acm.hdu.edu.cn/showproblem.php?pid=1026)

**题目分析**

这道题我先是用深搜的算法做的，可是却无奈超时了。查了一下网上的资料才发现，对于这道题，用广搜来做更加适合，因为广搜相比深搜，能够更快地得到最短的路径。不过有点奇怪的是，我这里原先是用数组来实现队列的操作的，可是会出现超时的情况，而我把它改成 C++ 下自带的队列后，却成功的 AC 了。

**实现代码**

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

#### 8.4 收集珠宝

**题目来源**

[HDOJ 1044 Collect More Jewels](http://acm.hdu.edu.cn/showproblem.php?pid=1044)

**题目分析**

本题我一开始是用 DFS 的思路做的，可是 TLE 了。分析了一下发现，深搜的时间复杂度达到 2 的 L 次方，耗时过长。查了一下资料，最终使用 BFS + DFS 的思路解决。具体做法是：用 BFS 得到起点、珠宝以及终点之间的最短路径（求无权图的最短路径一般使用 BFS），然后再用 DFS 搜索最大获得价值。另外，我们可以用 sum 保存所有珍宝的价值和来方便深搜剪枝。值得注意的是，根据题目数据，至少要定义 51 * 51 的二维数组才足以存储地图，因为接收每一行的字符串时不仅要接收数据部分，还要接收结束符。关于此题，网上还有一种 BFS + 状态压缩的方法，感兴趣的朋友可以去看下。

**实现代码**

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

#### 8.5 最大碉堡数

**题目来源**

[HDOJ 1045 Fire Net](http://acm.hdu.edu.cn/showproblem.php?pid=1045)

**题目分析**

本题在 HDOJ 里面被分在图论和贪心算法两个板块中，但实际上，这道题用 DFS 解决更加方便，除了以下本人的 DFS 算法之外，网上还有另一种普遍的 DFS 算法思路，感兴趣的朋友可以上网查查。如果想要练练二分图或者贪心算法，则可以试一试，不过难度也会随之上升。

**实现代码**

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

#### 8.6 噩梦

**题目来源**

[HDOJ 1072 Nightmare](http://acm.hdu.edu.cn/showproblem.php?pid=1072)

**题目分析**

这题需要求最少步骤，也可以说是求最短路径，因此用广度优先搜索（BFS）要比深度优先搜索（DFS）更好。这里由于每一个点都可以重复走，所以为了减少搜索工作量，我们设置了一个`vis`数组来记录每一个节点的所剩时间，初始时间都为 0（除了起点），如果走下去剩余时间没大于原先剩余时间，则不往下搜索。

**实现代码**

```c++
#include <iostream>
#include <queue>
using namespace std;

const int origin_time = 6;
const int directions[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };

int T;
int N, M;
int **labyrinth;
int **vis;
int start_x, start_y;
int result, flag;

struct node {
    int x, y;
    int time, truetime;
    node(int a, int b, int c, int d) {
        x = a;
        y = b;
        time = c;
        truetime = d;
    }
};

int main() {
    cin >> T;
    while(T--) {
        cin >> N >> M;
        labyrinth = new int*[N];
        vis = new int*[N];
        for(int i = 0; i < N; i++) {
            labyrinth[i] = new int[M];
            vis[i] = new int[M];
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++) {
                cin >> labyrinth[i][j];
                if(labyrinth[i][j] == 2) {
                    start_x = i;
                    start_y = j;
                }
                vis[i][j] = 0;
            }
        }
        queue<node> q;
        q.push(node(start_x, start_y, origin_time, 0));
        vis[start_x][start_y] = origin_time;
        flag = 0;
        while(!q.empty()) {
            node temp = q.front();
            q.pop();
            result = temp.truetime;
            if(labyrinth[temp.x][temp.y] == 3) {
                flag = 1;
                break;
            }
            if(temp.time >= 2) {
                for(int i = 0; i < 4; i++) {
                    int newx = temp.x + directions[i][0];
                    int newy = temp.y + directions[i][1];
                    if(newx >= 0 && newx < N && newy >= 0 && newy < M && labyrinth[temp.x][temp.y] && temp.time - 1 > vis[newx][newy]) {
                        if(labyrinth[newx][newy] == 4) {
                            q.push(node(newx, newy, origin_time, result + 1));
                            vis[newx][newy] = 6;
                        } else {
                            q.push(node(newx, newy, temp.time - 1, result + 1));
                            vis[newx][newy] = temp.time - 1;
                        }
                    }
                }
            }
        }
        if(flag) {
            cout << result << endl;
        } else {
            cout << -1 << endl;
        }
    }
    return 0;
}
```

### 第 9 章 排序

#### 9.1 伊格和公主（四）

**题目来源**

[HDOJ 1029 Ignatius and the Princess IV](http://acm.hdu.edu.cn/showproblem.php?pid=1029)

**题目分析**

实际上，这道题分在排序题有点勉强，因为它可以不用排序解决。排序的方法指的是先对数组排序，出现 (N + 1) / 2 次的元素总会在排序后该序列的中间位置。但排序的做法相对开销比较大，这里我们采用绝对众数的方法，这种方法复杂度为 O(n)，而且容易掌握，推荐大家使用。

**实现代码**

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

### 第 10 章 水题

#### 10.1 数字序列

**题目来源**

[HDOJ 1005 Number Sequence](http://acm.hdu.edu.cn/showproblem.php?pid=1005)

**题目分析**

对于这道题，我一开始的做法是根据进行 n 次迭代得到最后的结果，可是提交程序的结果是 TLE。于是我谷歌一查才发现，原来这是典型的找规律题，从迭代函数`f(n) = (A * f(n - 1) + B * f(n - 2)) mod 7`可以看出，f(n - 1) 的取值可能有 7 种，f(n-2) 也有 7 种，故 f(n - 1)f(n - 2) 的组合可能有 49 种，于是可以得到，在 49 次迭代内 f(n) 必有规律可寻，这便是此题的核心思路。值得注意的是，规律不一定是从 1 1 开始，因为序列可能是 1 1 2 3 2 3……，因此我们在每次迭代之后必须从第一个数遍历到当前迭代数，来得到规律开始的地方及规律的周期。

**实现代码**

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

#### 10.2 数根

**题目来源**

[HDOJ 1013 Digital Roots](http://acm.hdu.edu.cn/showproblem.php?pid=1013)

**题目分析**

这道题有两种做法：常规算法和九余数法。其中我在第一种做法上消耗了较长时间，主要是我分配的字符数组太小了。尽管早就知道不能直接用 int 变量接收输入的 n，而要用字符串来表示，但我给出的长度为 1000 的数组还是满足不了题目的要求（在 C++ 版本的程序中是可以的），最后数组的长度是 10000。另一种方法则更为简洁和常用，在数根的[维基百科英文字条](https://en.wikipedia.org/wiki/Digital_root)中就有提及到，另外知乎上也有关于其证明的[讨论](https://www.zhihu.com/question/30972581)。九余数算法的核心公式就是`dr(n) = 1 + ((n - 1) mod 9)`，只要利用这个公式，我们就可以大大简化我们的程序。

**实现代码**

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

#### 10.3 随机数生成器

**题目来源**

[HDOJ 1014 Uniform Generator](http://acm.hdu.edu.cn/showproblem.php?pid=1014)

**题目分析**

关于这道题，我一开始是直接根据题意进行解决的，具体的思路便是获取 x 为 0 - mod-1 时 seed 的数值，每次获取的时候与之前与获取到的数值进行比较，如果出现重复的数，则说明是一个 Bad Choice ，并退出获取操作。这个思路表面是可行的，可惜耗时太长，被 OJ 判了 TLE。那么我们应该怎么解决这个问题呢？方法就是验证 step 和 mod 的最大公约数是不是 1，如果是，则是 Good Choice，否则是 Bad Choice。这个方法可行的原因如下：seed(0) 为 0，第一次计算后结果为 step，第二次为 2 \* step ，第三次是 3 \* step ，一直到 (k \* step) % mod ，如果此时 k < mod ，则不合题意，所以要满足 Good Choice，则 step 和 mod 的最大公约数只能为1。从这道题我们也可以看出，编码前进行一定的分析能够提高我们解决问题的效率。

**实现代码**

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

#### 10.4 窃贼

**题目来源**

[HDOJ 1015 Safecracker](http://acm.hdu.edu.cn/showproblem.php?pid=1015)

**题目分析**

这道题实质上考察的是深度优先搜索，为了满足题意要求，我们先对输入的字符串中的字符降序排列，这样子只要找到一个满足题意的字符串，我们就可以退出操作了。

**实现代码**

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

#### 10.5 格子土地

**题目来源**

[HDOJ 1046 Gridland](http://acm.hdu.edu.cn/showproblem.php?pid=1046)

**题目分析**

此题很简单，只要找到规律即可，具体可以参考[这篇文章](http://blog.csdn.net/hurmishine/article/details/51992672)。

**实现代码**

```c
#include<stdio.h>
  
int main()  
{  
    int total;  
    int m,n;  
    int i;  
    scanf("%d",&total);  
    for(i=1; i <= total ; i++)  
    {  
        scanf("%d%d",&m,&n);  
        printf("Scenario #%d:\n",i);  
        if((m*n)%2 == 0)  
            printf("%d.00\n",m*n);  
        else  
            printf("%d.41\n",m*n);  
        printf("\n");  
    }  
    return 0;  
}
```

#### 10.6 田忌赛马

**题目来源**

[HDOJ 1052 Tian Ji -- The Horse Racing](http://acm.hdu.edu.cn/showproblem.php?pid=1052)

**题目分析**

这道水题的解决思路是贪心，具体做法是：对田忌和齐王的马从小到大各自排序，然后从两边扫描，如果小的能赢，就赢，如果不能，就比最大的马，能赢则赢，如果也不能赢，就用最小的和齐王最大的比掉，不管是否平局都保证了最优选择。

**实现代码**

```c++
#include<stdio.h>
#include<algorithm>
using namespace std;

int main() {
    int n;
    int a[1000], b[1000];
    while(scanf("%d", &n) == 1 && n) {
        for(int i = 0; i < n; i++) {
            scanf("%d", &a[i]);
        }
        for(int i = 0; i < n; i++) {
            scanf("%d", &b[i]);
        }
        sort(a, a + n);
        sort(b, b + n);
        int begin1, begin2, end1, end2, result;
        begin1 = begin2 = 0;
        end1 = end2 = n - 1;
        result = 0;
        while(begin1 <= end1) {
            if(a[begin1] > b[begin2]) {
                result++;
                begin1++;
                begin2++;
            } else if(a[end1] > b[end2]) {
                result++;
                end1--;
                end2--;
            } else {
                result -= (a[begin1] != b[end2]);
                begin1++;
                end2--;
            }
        }
        printf("%d\n", result * 200);
    }
    return 0;
}
```