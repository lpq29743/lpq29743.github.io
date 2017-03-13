---
layout:     post
title:      "URI、URL和URN"
subtitle:   "关于URI、URL和URN的点点滴滴"
date:       2017-03-13 21:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 漫谈
---

> URI、URL和URN的细枝末节


## 前言

之所以写这篇文章，是因为在网上看到了一个关于URL空格表示的问题，所以想通过这篇文章整理一下编程以及学习过程中经常打交道的URI、URL和URN。

---

## 正文

**URI、URL和URN有什么区别**

网上已经有很多围绕这个问题进行讨论的文章，具体可以看stackoverflow上的问题 [What is the difference between a URI, a URL and a URN?](http://stackoverflow.com/questions/176264/what-is-the-difference-between-a-uri-a-url-and-a-urn) 或者IBM网站的文章 [分清 URI、URL 和 URN](https://www.ibm.com/developerworks/cn/xml/x-urlni.html) 。

**各大浏览器规定的URL最长长度是多少**

要访问一个网址，我们一般需要输入特定的URL，那么在浏览器中，URL是不是不限字符，可以无限输下去呢？事实上不是这样的，这个问题已经在stackoverflow有了很好的回答，具体可以点击[这里](http://stackoverflow.com/questions/417142/what-is-the-maximum-length-of-a-url-in-different-browsers)。

**短链接是什么，它能给我们带来什么帮助**

短链接是一把双刃剑，它可以把较长的URL缩短，但也会带来安全隐患，想要了解更多，可以查看[维基百科相关字条](https://zh.wikipedia.org/wiki/%E7%B8%AE%E7%95%A5%E7%B6%B2%E5%9D%80%E6%9C%8D%E5%8B%99)。现在网上也是出现了各大公司的URL短链接服务，个人推荐[百度短链接服务](http://dwz.cn/)。关于短链接的实现，可以查看[知乎相关回答](https://www.zhihu.com/question/29270034/answer/46446911)以及[stackoverflow相关讨论](http://stackoverflow.com/questions/742013/how-to-code-a-url-shortener)，在Github上也是给出了[各个版本的实现方案](https://github.com/delight-im/ShortURL)。

**URL中的特殊字符应该怎么表示**

URL中不能直接添加空格等特殊字符，这也是很多人在编程中出现问题的主要原因。那么我们应该如何识别或者添加URL中的特殊字符呢？具体可以查看维基百科的[百分号编码词条](https://en.wikipedia.org/wiki/Percent-encoding)。实际上，空格的表示方法有`%20`和`+`两种，但一般推荐前者，关于这个问题，在stackoverflow上也有很好的[解答](http://stackoverflow.com/questions/1634271/url-encoding-the-space-character-or-20)。

**怎样的字符会让URL无效**

这个问题源自stackoverflow，这里直接给出[问题链接](http://stackoverflow.com/questions/1547899/which-characters-make-a-url-invalid)。

**URL中的 #! 号是什么意思**

之前博主在做一个QQ音乐的爬虫实验也遇到过相应的问题，在URL中，#号是不影响访问网页内容的。这个问题也在知乎上被讨论过，具体可以查看[这里](https://www.zhihu.com/question/19946782)。

**URL大小写敏感吗**

这个问题其实蛮有趣的，在现实生活中我们可能很少注意到这个问题，但实际上，在知乎和stackoverflow上都有过相应的讨论，这里我们分别给出它们的相应链接： [网址链接是否区分大小写？](https://www.zhihu.com/question/19572705) 和 [Should URL be case sensitive?](http://stackoverflow.com/questions/7996919/should-url-be-case-sensitive) 。

## 后记

URI、URL和URN与我们的生活息息相关，这些问题虽然小，但是有趣而且有用，我也随时根据自己的学习情况更新这篇文章。


