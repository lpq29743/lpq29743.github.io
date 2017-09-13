---
layout:     post
title:      "哈工大复试上机题"
subtitle:   "迈出成功的第一步"
date:       2017-09-13 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 相信自己
>


## 前言

之前已经刷过北航和北邮的复试题，这次来刷刷哈工大的。

---

## 正文

**字符串内排序**

***题目描述***

输入一个字符串，长度小于等于200，然后将输出按字符顺序升序排序后的字符串

***输入描述***

测试数据有多组，输入字符串

***输出描述***

对于每组输入，输出处理后的结果

***输入例子***

```
bacd
```

***输出例子***

```
abcd
```

***程序代码***

```c++
#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;

int main() {
    char s[205];
    while(scanf("%s", s) == 1) {
        sort(s, s + strlen(s));
        printf("%s\n", s);
    }
    return 0;
}
```
**数组逆置**

***题目描述***

输入一个字符串，长度小于等于200，然后将数组逆置输出

***输入描述***

测试数据有多组，每组输入一个字符串

***输出描述***

对于每组输入，请输出逆置后的结果

***输入例子***

```
hdssg
```

***输出例子***

```
gssdh
```

***程序代码***

```c
#include<stdio.h>
#include<string.h>

int main() {
    char s[205];
    while(scanf("%s", s) == 1) {
        for(int i = 0; i <= (strlen(s) - 1) / 2; i++) {
            char c = s[i];
            s[i] = s[strlen(s) - i - 1];
            s[strlen(s) - i - 1] = c;
        }
        printf("%s\n", s);
    }
    return 0;
}
```

## 后记

这几天来，收到的消息喜忧参半，心情也比较复杂，希望能够收拾好行囊，继续朝实现梦想的道路前进。
