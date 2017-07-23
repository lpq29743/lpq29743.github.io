---
layout:     post
title:      "北邮复试上机题"
subtitle:   "寄出梦想的明信片"
date:       2017-07-03 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 付出就有回报
>


## 前言

之前做了一下北航的复试题目，现在来试一下北邮的题目。

---

## 正文

**二进制数**

***题目描述***

大家都知道，数据在计算机里中存储是以二进制的形式存储的。小明学了C语言后，想知道一个类型为unsigned int类型的数字，存储在计算机中的二进制串是什么样子的。你能帮帮小明吗？并且，小明不想要二进制串中前面的没有意义的0串，即要去掉前导0

***输入描述***

第一行是数字T（T<=1000），表示要求数字个数。接下来有T行，每行一个数字n（0<=n<=10^8），表示要求的二进制串

***输出描述***

输出共T行。每行输出求得的二进制串

***输入例子***

```
5
23
535
2624
56275
989835
```

***输出例子***

```
10111
1000010111
101001000000
1101101111010011
11110001101010001011
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int n;
    scanf("%d", &n);
    for(int i = 0; i < n; i++) {
        int source;
        char result[30];
        scanf("%d", &source);
        int loc = 0;
        while(source != 0) {
            result[loc++] = source % 2 + '0';
            source /= 2;
        }
        for(int j = loc - 1; j >= 0; j--) {
            printf("%d", result[j] - '0');
        }
        printf("\n");
    }
    return 0;
}
```
**哈夫曼树**

***题目描述***

哈夫曼树，第一行输入一个数n，表示叶结点的个数。需要用这些叶结点生成哈夫曼树，根据哈夫曼树的概念，这些结点有权值，即weight，题目需要输出所有结点的值与权值的乘积之和。

***输入描述***

输入有多组数据。每组第一行输入一个数n，接着输入n个叶节点（叶节点权值不超过100，2<=n<=1000）

***输出描述***

输出权值

***输入例子***

```
5  
1 2 2 5 9
```

***输出例子***

```
37
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        for(int i = 0; i < n - 1; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j] < s[j + 1]) {
                    int temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int wpl = 0;
        for(int i = 0; i < n - 1; i++) {
            int sum = s[n - i - 1] + s[n - i - 2];
            wpl = wpl + sum;
            s[n - i - 2] = sum;
            for(int j = n - i - 2; j > 0; j--) {
                if(s[j] > s[j - 1]) {
                    int temp = s[j];
                    s[j] = s[j - 1];
                    s[j - 1] = temp;
                } else {
                    break;
                }
            }
        }
        printf("%d\n", wpl);
    }
    return 0;
}
```

**比较奇偶数个数**

***题目描述***

第一行输入一个数，为n，第二行输入n个数，这n个数中，如果偶数比奇数多，输出NO，否则输出YES

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数（1<=n<=1000）

***输出描述***

如果偶数比奇数多，输出NO，否则输出YES。

***输入例子***

```
5
1 5 2 4 3
```

***输出例子***

```
YES
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        int count = 0;
        for(int i = 0; i < n; i++) {
            if(s[i] % 2 == 0) {
                count++;
            } else {
                count--;
            }
        }
        printf("%s\n", count > 0 ? "NO" : "YES");
    }
    return 0;
}
```

**查找第K小数**

***题目描述***

查找一个数组的第K小的数，注意同样大小算一样大。 如 2 1 3 4 5 2 第三小数为3

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数(1<=n<=1000)，再输入k

***输出描述***

输出第k小的整数

***输入例子***

```
6
2 1 3 5 2 2
3
```

***输出例子***

```
3
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int n, k;
    int *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        scanf("%d", &k);
        for(int i = 0; i < n - 1; i++) {
            for(int j = 0; j < n - i - 1; j++) {
                if(s[j] > s[j + 1]) {
                    int temp = s[j];
                    s[j] = s[j + 1];
                    s[j + 1] = temp;
                }
            }
        }
        int count = 0;
        for(int i = 0; i < n; i++) {
            while(i + 1 < n && s[i] == s[i + 1]) {
                i++;
            }
            count++;
            if(count == k) {
                printf("%d\n", s[i]);
            }
        }
    }
    return 0;
}
```

**矩阵幂**

***题目描述***

给定一个n*n的矩阵，求该矩阵的k次幂，即P^k

***输入描述***

输入包含多组测试数据。数据的第一行为一个整数T(0<T<=10)，表示要求矩阵的个数。接下来有T组测试数据，每组数据格式如下： 第一行：两个整数n（2<=n<=10）、k（1<=k<=5），两个数字之间用一个空格隔开，含义如上所示。接下来有n行，每行n个正整数，其中，第i行第j个整数表示矩阵中第i行第j列的矩阵元素Pij且（0<=Pij<=10）。另外，数据保证最后结果不会超过10^8

***输出描述***

对于每组测试数据，输出其结果。格式为：n行n列个整数，每行数之间用空格隔开，注意，每行最后一个数后面不应该有多余的空格

***输入例子***

```
3
2 2
9 8
9 3
3 3
4 8 4
9 3 0
3 5 7
5 2
4 0 3 0 1
0 0 5 8 5
8 9 8 5 3
9 6 1 7 8
7 2 5 7 3
```

***输出例子***

```
153 96
108 81
1216 1248 708
1089 927 504
1161 1151 739
47 29 41 22 16
147 103 73 116 94
162 108 153 168 126
163 67 112 158 122
152 93 93 111 97
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

int main() {
    int t;
    scanf("%d", &t);
    for(int i = 0; i < t; i++) {
        int n, k;
        int **s1, **s2, **s3;
        scanf("%d %d", &n, &k);
        s1 = (int **)malloc(sizeof(int *) * n);
        s2 = (int **)malloc(sizeof(int *) * n);
        s3 = (int **)malloc(sizeof(int *) * n);
        for(int j = 0; j < n; j++) {
            s1[j] = (int *)malloc(sizeof(int) * n);
            s2[j] = (int *)malloc(sizeof(int) * n);
            s3[j] = (int *)malloc(sizeof(int) * n);
        }
        for(int j1 = 0; j1 < n; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                scanf("%d", &s1[j1][j2]);
                s2[j1][j2] = s1[j1][j2];
                s3[j1][j2] = 0;
            }
        }
        for(int j1 = 1; j1 < k; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                for(int j3 = 0; j3 < n; j3++) {
                    int sum = 0;
                    for(int j4 = 0; j4 < n; j4++) {
                        sum += s1[j2][j4] * s2[j4][j3];
                    }
                    s3[j2][j3] = sum;
                }
            }
            for(int j2 = 0; j2 < n; j2++) {
                for(int j3 = 0; j3 < n; j3++) {
                    s2[j2][j3] = s3[j2][j3];
                }
            }
        }
        for(int j1 = 0; j1 < n; j1++) {
            for(int j2 = 0; j2 < n; j2++) {
                if(j2 != 0) {
                    printf(" ");
                }
                printf("%d", s2[j1][j2]);
            }
            printf("\n");
        }
    }
    return 0;
}
```

**C翻转**

***题目描述***

首先输入一个5 * 5的数组，然后输入一行，这一行有四个数，前两个代表操作类型，后两个数x y代表需操作数据为以x y为左上角的那几个数据。 操作类型有四种：  1 2 表示：90度，顺时针，翻转4个数  1 3 表示：90度，顺时针，翻转9个数  2 2 表示：90度，逆时针，翻转4个数  2 3 表示：90度，逆时针，翻转9个数 

***输入描述***

输入有多组数据。每组输入一个5 * 5的数组，然后输入一行，这一行有四个数，前两个代表操作类型，后两个数x y代表需操作数据为以x y为左上角的那几个数据

***输出描述***

输出翻转后的数组

***输入例子***

```
1 2 3 4 5
6 7 8 9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
1 3 1 1
```

***输出例子***

```
11 6 1 4 5
12 7 2 9 10
13 8 3 14 15
16 17 18 19 20
21 22 23 24 25
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int matrix[5][5];
    int a, b, x, y;
    while(1) {
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                if(scanf("%d", &matrix[i][j]) != 1) {
                    break;
                }
            }
        }
        if(scanf("%d %d %d %d", &a, &b, &x, &y) != 4) {
            break;
        }
        // 1 2 表示：90度，顺时针，翻转4个数
        if(a == 1 && b == 2) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x][y - 1];
            matrix[x][y - 1] = matrix[x][y];
            matrix[x][y] = matrix[x - 1][y];
            matrix[x - 1][y] = temp;
        }
        // 1 3 表示：90度，顺时针，翻转9个数
        if(a == 1 && b == 3) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x + 1][y - 1];
            matrix[x + 1][y - 1] = matrix[x + 1][y + 1];
            matrix[x + 1][y + 1] = matrix[x - 1][y + 1];
            matrix[x - 1][y + 1] = temp;
            temp = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y - 1];
            matrix[x][y - 1] = matrix[x + 1][y];
            matrix[x + 1][y] = matrix[x][y + 1];
            matrix[x][y + 1] = temp;
        }
        // 2 2 表示：90度，逆时针，翻转4个数
        if(a == 2 && b == 2) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y];
            matrix[x][y] = matrix[x][y - 1];
            matrix[x][y - 1] = temp;
        }
        // 2 3 表示：90度，逆时针，翻转9个数
        if(a == 2 && b == 3) {
            int temp = matrix[x - 1][y - 1];
            matrix[x - 1][y - 1] = matrix[x - 1][y + 1];
            matrix[x - 1][y + 1] = matrix[x + 1][y + 1];
            matrix[x + 1][y + 1] = matrix[x + 1][y - 1];
            matrix[x + 1][y - 1] = temp;
            temp = matrix[x - 1][y];
            matrix[x - 1][y] = matrix[x][y + 1];
            matrix[x][y + 1] = matrix[x + 1][y];
            matrix[x + 1][y] = matrix[x][y - 1];
            matrix[x][y - 1] = temp;
        }
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                if(j != 0) {
                    printf(" ");
                }
                printf("%d", matrix[i][j]);
            }
            printf("\n");
        }
    }
    return 0;
}
```

**打牌**

***题目描述***

牌只有1到9，手里拿着已经排好序的牌a，对方出牌b，用程序判断手中牌是否能够压过对方出牌。  规则：出牌牌型有5种   [1]一张 如4 则5...9可压过 [2]两张 如44 则55，66，77，...，99可压过 [3]三张 如444 规则如[2][4]四张 如4444 规则如[2][5]五张 牌型只有12345 23456 34567 45678 56789五个，后面的比前面的均大

***输入描述***

输入有多组数据。每组输入两个字符串(字符串大小不超过100)a，b。a字符串代表手中牌，b字符串代表处的牌

***输出描述***

压过输出YES 否则NO

***输入例子***

```
12233445566677
33
```

***输出例子***

```
YES
```

***程序代码***

```c
#include<stdio.h>
#include<string.h>

int main() {
    char a[100], b[6], c[5];
    int num, isYes;
    while(scanf("%s %s", &a, &b) == 2) {
        num = b[0] - '0';
        isYes = 0;
        if(strlen(b) == 1 || strlen(b) == 2 || strlen(b) == 3 || strlen(b) == 4) {
            for(int i = num + 1; i <= 9; i++) {
                int j;
                for(j = 0; j < strlen(b); j++) {
                    c[j] = i + '0';
                }
                c[j] = '\0';
                if(strstr(a, c) != NULL) {
                    isYes = 1;
                    break;
                }
            }
        } else if(strlen(b) == 5) {
            for(int i = num + 1; i <= 5; i++) {
                int isAllFind = 1;
                for(int j = i; j < i + 5; j++) {
                    if(strchr(a, j + '0') == NULL) {
                        isAllFind = 0;
                        break;
                    }
                }
                if(isAllFind == 1) {
                    isYes = 1;
                    break;
                }
            }
        }
        printf("%s\n", isYes == 1 ? "YES" : "NO");
    }
    return 0;
}
```

**树查找**

***题目描述***

有一棵树，输出某一深度的所有节点，有则输出这些节点，无则输出EMPTY。该树是完全二叉树

***输入描述***

输入有多组数据。每组输入一个n(1<=n<=1000)，然后将树中的这n个节点依次输入，再输入一个d代表深度

***输出描述***

输出该树中第d层得所有节点，节点间用空格隔开，最后一个节点后没有空格

***输入例子***

```
4
1 2 3 4
2
```

***输出例子***

```
2 3
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main() {
    int n, d, *s;
    while(scanf("%d", &n) == 1) {
        s = (int *)malloc(sizeof(int) * n);
        for(int i = 0; i < n; i++) {
            scanf("%d", &s[i]);
        }
        scanf("%d", &d);
        if((int)pow(2, d - 1) > n) {
            printf("EMPTY\n");
        } else {
            for(int i = (int)pow(2, d - 1); i <= (int)pow(2, d) - 1 && i <= n; i++) {
                if(i != (int)pow(2, d - 1)) {
                    printf(" ");
                }
                printf("%d", s[i - 1]);
            }
            printf("\n");
        }
    }
    return 0;
}
```

**查找**

***题目描述***

读入一组字符串（待操作的），再读入一个记录n记下来有几条命令，总共有2种命令，翻转：从下标为i的字符到i+len-1之间的字符串倒序；替换：命中如果第一位为1，用命令的第四位开始到最后的字符串替换原读入的字符串下标 i 到 i+len-1的字符串。每次执行一条命令后新的字符串代替旧的字符串（即下一条命令在作用在得到的新字符串上）。命令格式：第一位0代表翻转，1代表替换；第二位代表待操作的字符串的起始下标int i；第三位表示需要操作的字符串长度int len

***输入描述***

输入有多组数据。每组输入一个字符串（不大于100）然后输入n，再输入n条指令（指令一定有效）

***输出描述***

根据指令对字符串操作后输出结果

***输入例子***

```
bac
2
003
112as
```

***输出例子***

```
cab
cas
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 1000

int main() {
    int n;
    int start, length;
    char source[N], command[N];
    while(scanf("%s %d", &source, &n) == 2) {
        while(n-- > 0) {
            scanf("%s", &command);
            start = command[1] - '0';
            length = command[2] - '0';
            if(command[0] == '0') {
                for(int j = start, k = start + length - 1; j < k; j++, k--) {
                    char temp = source[j];
                    source[j] = source[k];
                    source[k] = temp;
                }
            } else {
                char a[N], b[N], c[N];
                strncpy(a, source, start);
                a[start] = '\0';
                strncpy(b, command + 3, strlen(command) - 3);
                b[strlen(command) - 3] = '\0';
                strncpy(c, source + start + length, strlen(source) - start - length);
                c[strlen(source) - start - length] = '\0';
                strcat(a, b);
                strcat(a, c);
                strcpy(source, a);
            }
            printf("%s\n", source);
        }
    }
    return 0;
}
```

**复数集合**

***题目描述***

一个复数（x+iy）集合，两种操作作用在集合上：Pop表示读出集合中模值最大的复数，如集合为空输出empty，不为空就输出最大的那个复数并且从集合中删除那个复数，再输出集合大小SIZE；Insert a+ib（a，b表示实部和虚部）将a+ib加入到集合中，输出集合的大小SIZE

***输入描述***

输入有多组数据。每组输入一个n(1<=n<=1000)，然后再输入n条指令

***输出描述***

根据指令输出结果。模相等的输出b较小的复数。a和b都是非负数

***输入例子***

```
3
Pop
Insert 1+i2
Pop
```

***输出例子***

```
empty
SIZE = 1
1+i2
SIZE = 0
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct node {
    int a;
    int b;
};

int main() {
    int n;
    char command[10];
    while(scanf("%d", &n) == 1) {
        struct node s[1000];
        int size = 0;
        while(n-- > 0) {
            scanf("%s", &command);
            if(strcmp(command, "Pop") == 0) {
                if(size == 0) {
                    printf("empty\n");
                } else {
                    int loc = -1, max = -1, mod, ta, tb = -1;
                    for(int i = 0; i < size; i++) {
                        mod = s[i].a * s[i].a + s[i].b * s[i].b;
                        if(mod > max || (mod == max && s[i].b < tb)) {
                            loc = i;
                            max = mod;
                            ta = s[i].a;
                            tb = s[i].b;
                        }
                    }
                    for(int i = loc; i < size - 1; i++) {
                        s[i] = s[i + 1];
                    }
                    size--;
                    printf("%d+i%d\nSIZE = %d\n", ta, tb, size);
                }
            } else if(strcmp(command, "Insert") == 0) {
                char obj[100];
                scanf("%s", &obj);
                char *temp = strchr(obj, '+');
                char ta[100], tb[100];
                strncpy(ta, obj, strlen(obj) - strlen(temp));
                ta[strlen(obj) - strlen(temp)] = '\0';
                strncpy(tb, temp + 2, strlen(temp));
                tb[strlen(temp)] = '\0';
                s[size].a = atoi(ta);
                s[size].b = atoi(tb);
                size++;
                printf("SIZE = %d\n", size);
            }
        }
    }
    return 0;
}
```

**二叉排序树**

***题目描述***

输入一系列整数，建立二叉排序数，并进行前序，中序，后序遍历

***输入描述***

输入第一行包括一个整数n(1<=n<=100)。接下来的一行包括n个整数

***输出描述***

可能有多组测试数据，对于每组数据，将题目所给数据建立一个二叉排序树，并对二叉排序树进行前序、中序和后序遍历。每种遍历结果输出一行。每行最后一个数据之后有一个空格。输入中可能有重复元素，但是输出的二叉树遍历序列中重复元素不用输出

***输入例子***

```
5
1 6 5 9 8

```

***输出例子***

```
1 6 5 9 8 
1 5 6 8 9 
5 8 9 6 1 
```

***程序代码***

```c
#include<stdio.h>
#include<stdlib.h>

typedef struct node {
    int val;
    struct node *left;
    struct node *right;
} node;

node* insert(int val, node *root) {
    if(root == NULL) {
        root = (node *)malloc(sizeof(node));
        root->val = val;
        root->left = NULL;
        root->right = NULL;
    } else {
        if(root->val == val) {
            return root;
        } else if(root->val > val) {
            root->left = insert(val, root->left);
        } else {
            root->right = insert(val, root->right);
        }
    }
    return root;
}

void preOrder(node *root) {
    if(root == NULL) {
        return;
    } else {
        printf("%d ", root->val);
        preOrder(root->left);
        preOrder(root->right);
    }
}

void inOrder(node *root) {
    if(root == NULL) {
        return;
    } else {
        inOrder(root->left);
        printf("%d ", root->val);
        inOrder(root->right);
    }
}

void postOrder(node *root) {
    if(root == NULL) {
        return;
    } else {
        postOrder(root->left);
        postOrder(root->right);
        printf("%d ", root->val);
    }
}

int main() {
    int n;
    while(scanf("%d", &n) == 1) {
        node *root = NULL;
        int tmp;
        for(int i = 0; i < n; i++) {
            scanf("%d", &tmp);
            root = insert(tmp, root);
        }
        preOrder(root);
        printf("\n");
        inOrder(root);
        printf("\n");
        postOrder(root);
        printf("\n");
    }
    return 0;
}
```

**找最小数**

***题目描述***

第一行输入一个数n，1 <= n <= 1000，下面输入n行数据，每一行有两个数，分别是x y。输出一组x y，该组数据是所有数据中x最小，且在x相等的情况下y最小的

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数对

***输出描述***

输出最小的整数对

***输入例子***

```
5  
3 3  
2 2  
5 5  
2 1  
3 6
```

***输出例子***

```
2 1
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int n;
    while(scanf("%d", &n) == 1) {
        int minx, miny;
        scanf("%d%d", &minx, &miny);
        while(--n) {
            int x, y;
            scanf("%d%d", &x, &y);
            if(x < minx || (x == minx && y < miny)) {
                minx = x;
                miny = y;
            }
        }
        printf("%d %d\n", minx, miny);
    }
    return 0;
}
```

**查找**

***题目描述***

输入数组长度 n ，输入数组a[1...n]，输入查找个数m，输入查找数字b[1...m]。输出 YES or NO，查找有则YES，否则NO

***输入描述***

输入有多组数据。每组输入n，然后输入n个整数，再输入m，然后再输入m个整数（1<=m,n<=100）

***输出描述***

如果在n个数组中输出YES否则输出NO

***输入例子***

```
5
1 5 2 4 3
3
2 5 6
```

***输出例子***

```
YES
YES
NO
```

***程序代码***

```c
#include<stdio.h>

int main() {
    int n, m, a[101];
    while(scanf("%d", &n) == 1) {
        for(int i = 0; i < n; i++) {
            scanf("%d", &a[i]);
        }
        scanf("%d", &m);
        while(m--) {
            int isFind = 0, tmp;
            scanf("%d", &tmp);
            for(int i = 0; i < n; i++) {
                if(tmp == a[i]) {
                    isFind = 1;
                    break;
                }
            }
            printf("%s\n", isFind ? "YES" : "NO");
        }
    }
    return 0;
}
```

## 后记

继续前进，没有一滴汗水会白流。
