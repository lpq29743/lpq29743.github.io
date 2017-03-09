---
layout:     post
title:      "Python历险记第四站"
subtitle:   "利用Python实现情感分析"
date:       2017-03-09 19:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 当Python遇到情绪分析~~~


## 前言

继上篇讲述Python实现词云后，这篇文章将围绕Python实现情绪分析进行展开。实验过程会运用到爬虫和情绪分析的知识。话不多说，让我们开始这篇文章！

---

## 正文

首先，我们要选择数据源。由于之前在[第二站](https://lpq29743.github.io/redant/2016/12/18/PythonQQMusicMS/)的时候由于网站的特殊原因并没有实现scrapy，所以笔者今天打算顺便学习一下scrapy（[相关教程](http://scrapy-chs.readthedocs.io/zh_CN/1.0/intro/tutorial.html)）这个爬虫框架。我选择了我最喜欢的电影《发条橙》的[豆瓣短评](https://movie.douban.com/subject/1292233/comments)作为实验对象。

## 后记


