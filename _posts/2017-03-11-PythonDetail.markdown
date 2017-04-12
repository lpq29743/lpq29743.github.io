---
layout:     post
title:      "Python的细枝末节"
subtitle:   "献给一见钟情的Python"
date:       2017-03-11 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 不定期更新的Python笔记~~


## 前言

Python是在大二的暑假接触的，平时主要是用它来写一些爬虫，之后也会用它来进行一些机器学习、自然语言等的实验，可以说，Python给我带来的帮助还是蛮大的。今天，就让我用这篇文章来记录我与Python的点点滴滴。（注：本文中的Python版本默认是Python3）

---

## 正文

**如何判断变量的类型**

Python判断变量的类型主要有两种方法：type和isinstance。

type的使用方法如下：

```python
x = int(5)
if type(x) == int:
    print("x is interger.")
else: 
    print("false.")
```

isinstance的使用方法如下：

```python
isinstance({},dict)
isinstance(5,dict)
isinstance([],dict)
```

官方推荐的比较方法是后者，但是后者逻辑更为复杂，想要了解更多可以点击[这里](http://blog.csdn.net/handsomekang/article/details/10043633)。

**为什么Python没有++运算符**

Python并不支持自增运算符和自减运算符，这个问题已经在知乎上得到了很好的讨论，具体可以查看[这里](https://www.zhihu.com/question/20913064)。在Python中，使用后缀自增运算符会直接报错，而使用前缀自增运算符则会被识别为正负号，如`++i`的结果就是`i`而不是`i+1`。

## 后记

我想，Python应该会是陪我比较久的一个原因。希望随着时间的累计，这篇文章的质量能越来越高。


