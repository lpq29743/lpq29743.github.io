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
- Git：版本控制工具，有必要的话还可以把代码传到github上

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
- greenDao：数据库操作框架，还算蛮不错的
- RxJava+RxAndroid：响应式编程框架，具体学习可以查看[这篇文章](https://lpq29743.github.io/redant/2016/09/29/RxJava/)
- logger：日志输出框架，相当好用
- bugly：腾讯的崩溃统计平台，地址请点击[这里](https://bugly.qq.com/v2/)
- MVVM：Android架构设计方法，使代码更加清晰，实现高内聚低耦合
- Material Design：Android5.0新特性，使界面更加友好和美观
- FastJson：阿里开发的Json解析工具，在国内十分流行

**相关小技巧**

- XML Tools：具体使用方法可以查看本人[这篇文章](https://lpq29743.github.io/redant/2016/10/04/AndroidXmlTools/)
- 小知识点总结：本人根据自己的开发经验，不定期总结一些小知识点，发布在[这篇文章](https://lpq29743.github.io/redant/2017/01/23/AndroidDetail/)

**优秀项目**

- iosched：Google每年I/O大会都会出一款范例App，这是2016年的App，项目地址在[这里](https://github.com/google/iosched)
- u2020：Jake Wharton（ButterKnife等的开发者）的示例项目，项目地址在[这里](https://github.com/JakeWharton/u2020)
- android-architecture：Android官方MVP架构示例项目，项目地址在[这里](https://github.com/googlesamples/android-architecture)
- Philm：Chris Bannes的开源项目，实现了MVP，项目地址在[这里](https://github.com/chrisbanes/philm)

## 后记

关于Android开发的总结就先到这里了。随着技术的不断发展和本人技术的提高，这篇文章也会被不断的改善。
