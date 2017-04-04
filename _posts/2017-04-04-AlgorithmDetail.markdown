---
layout:     post
title:      "算法的细枝末节"
subtitle:   "程序的精髓——算法"
date:       2017-04-04 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 数据结构与算法
---

> 从2017年4月写起的算法笔记


## 前言

4月份的第一篇文章，送给程序员永远不可忽略的算法。算法是个有趣的东西，也是个折磨人的东西，在学习编码过程中，我会不时接触到关于算法的一些小问题，就用这篇文章来总结一下吧！

---

## 正文

**怎样快速得知一个数是否是2的幂，如果是，又是2的几次方**

2的幂次方写成二进制后是1后面跟n个0，所以我们可以用`(number & number - 1) == 0`（这个式子左边的两个运算符都是遵循从右到左的运算规则）来进行判断，而几次方则需利用循环或递归获取，具体代码如下：

```c
#include <stdio.h>
#include <stdlib.h>

// 递归判断一个数是2的多少次方
int log2(int value) {
    if (value == 1)
        return 0;
    else
        return 1 + log2(value >> 1);
}

int main(void) {
    int num;
    scanf("%d", &num);
    if(num & (num - 1))
        printf("%d不是2的幂次方\n", num);
    else
        printf("%d是2的%d次方\n", num, log2(num));
    return 0;
}
```

**求一个二进制数中1的个数**

通过上面的例子，我们可以发现，`(number & number - 1)`能移除掉二进制中最右边的1，循环移除，可以将1全部移除，利用这一点我们可以求一个二进制数中1的个数，代码如下：

```c
int Func(int data) { 
    int count = 0;  
    while (data) {  
        data = data & (data-1);  
        count++;  
    }  
    return count;  
}
```

利用上面这个函数，我们还可以求出A和B的二进制中有多少位不相同，具体做法为：

1. 将A和B异或得到C，即C=A^B
2. 计算C的二进制中有多少个1

**怎么在O(n)的时间复杂度要求下，完成对0-n二进制中1的计数**

这个问题是对上面那个问题的变种，如果按照上面的方法，求每个数的二进制形式中1的个数需要的时间将与其本身带有1的个数所决定，所以每个数不可能在O(1)的时间复杂度下完成1的计数，也就不可能在遍历下达到O(n)的时间复杂度，那么怎样解决呢？

第一种方法是让后面的数通过前面的数求出其二进制数中1的个数。通过上例我们知道，每做一次`k & (k - 1)`的操作，都会得到一个比本身小，且1的个数为本身1的个数减一的数，所以利用这一点我们可以写出以下程序解决问题：

```c
result[0] = 0
for(int k = 0; k <= n; k++)
    result[k] = result[k & (k - 1)] + 1;
```

还有一种方法是观察规律写出相应的程序。通过写出前面的一些数，我们可以发现1的个数和2的幂存在着一些关系，所以根据这点我们可以写出以下代码：

```c
result[0] = 0;
for (int powerOfTwo = 1; powerOfTwo < n; powerOfTwo *= 2) {
    for (int i = 0; i < powerOfTwo; i++) {
        result[powerOfTwo + i] = result[i] + 1;
    }
}
```

实际上，这道题的解决方法有好几个，我也是从stackoverflow上的问题得到了灵感，具体可以点击[问题地址](http://stackoverflow.com/questions/43007574/how-to-count-the-number-of-1-bits-set-in-0-1-2-n-in-on-time)查看。

## 后记

算法很容易被过于重视，也很容易被过于忽视，认识到自己需要的，才能更好的把握学习算法的时间投入。


