---
layout:     post
title:      "怎样系统地学习Android开发"
subtitle:   "写给自己的Android笔记"
date:       2017-03-03 20:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> 那些年与Android度过的岁月


## 前言

从大一开始接触Android，到现在已经两三年了，不算老鸟，但多多少少有自己的一点收获。这篇文章将会结合本人的学习经验和网络上的资料，总结出一个Android学习的框架。

---

## 正文

**开发工具**

- Android Studio：如果你现在还是在使用Eclipse，还在怀疑迁移AS带来的影响，那我可以很负责任地告诉你，可以换环境了。AS是官方推荐的开发环境，很多方面的体验都要优于Eclipse

**编程语言**

- Java：基本很多人学习Android开发之前都要学习Java，因为Android应用是基于Java编写的
- Kotlin：Android开发的Swift，是对现有Java的增强，可搭配Anko使用（具体看[这里](https://realm.io/cn/news/getting-started-with-kotlin-and-anko/)），推荐大家可以试试
- React Native：跨平台语言，可以同时开发IOS App和Android App

**开发框架**

- ButterKnife：依赖注入框架，提高代码可读性，具体可参考[这里](https://lpq29743.github.io/redant/2016/09/26/ButterKnife/)
- OkHttp：网络框架，相对成熟的解决方案
- Retrofit：网络框架，封装了OkHttp，也是相当优秀的框架，项目中可以考虑Retrofit+OkHttp的方案
- EventBus：事件总线框架，具体学习可以参考[这里](https://lpq29743.github.io/redant/2016/09/26/EventBus/)
- Glide：Picasso、Glide和Fresco这三大图片加载框架都很不错，前两者适合普通应用，后者适合图片型应用

## 后记


