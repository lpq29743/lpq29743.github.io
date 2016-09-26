---
layout:     post
title:      "Android依赖注入框架ButterKnife"
subtitle:   "帮你偷懒的ButterKnife"
date:       2016-09-26 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 开源框架
---

> “不会偷懒的程序员不是好的程序员！”
>


## 前言

依赖注入框架在Android项目开发过程中极为方便，本位主要围绕这ButterKnife进行讲述。

---

## 正文

本文所介绍的版本是ButterKnife8.0.1（2016/04/29更新版本），项目源地址：[https://github.com/JakeWharton/butterknife](https://github.com/JakeWharton/butterknife)

###配置###

***步骤一***

Project 的 build.gradle 添加：
`dependencies {`
  `classpath 'com.neenbedankt.gradle.plugins:android-apt:1.8'`
`}`


## 后记

做第一个Android项目的时候就会经常听到依赖注入这个名词，后面接触Java Web也常有耳闻。于是，后面做Android项目的时候就会经常考虑使用依赖注入框架。而ButterKnife作为Android依赖注入框架的代表，是每一个想要“偷懒”的程序员必须掌握的。
