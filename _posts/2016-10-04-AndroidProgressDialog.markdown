---
layout:     post
title:      "Android控件之ProgressDialog"
subtitle:   "进度框ProgressDialog"
date:       2016-10-04 16:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 控件
---

> ProgressBar的兄弟——ProgressDialog


## 前言

ProgressDialog相比ProgressBar能展现更多的信息，这一篇就让我们来讲一讲它。

---

## 正文

***圆形ProgressDialog***

```java
// 实例化
ProgressDialog mypDialog=new ProgressDialog(this);
// 设置进度条风格，风格为圆形，旋转的
mypDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
// 设置ProgressDialog 标题
mypDialog.setTitle("圆形ProgressDialog");
// 设置ProgressDialog 提示信息
mypDialog.setMessage("测试1");
// 设置ProgressDialog 标题图标
mypDialog.setIcon(R.drawable.android);
// 设置ProgressDialog 的一个Button
mypDialog.setButton("确定",this);
// 设置初始位置
progressDialog.setProgress(50);
// 设置ProgressDialog 的进度条是否不明确
progressDialog.setIndeterminate(false);
// 设置ProgressDialog 是否可以按退回按键取消
progressDialog.setCancelable(true);
// 让ProgressDialog显示
progressDialog.show();
```

***长形ProgressDialog***

```java
// 实例化
ProgressDialog mypDialog=new ProgressDialog(this);
// 设置进度条风格，风格为圆形，旋转的
mypDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
// 设置ProgressDialog 标题
mypDialog.setTitle("长形ProgressDialog");
// 设置ProgressDialog 提示信息
mypDialog.setMessage("测试2");
// 设置ProgressDialog 标题图标
mypDialog.setIcon(R.drawable.android);
// 设置ProgressDialog 的一个Button
mypDialog.setButton("确定",this);
// 设置初始位置
progressDialog.setProgress(50);
// 设置ProgressDialog 的进度条是否不明确
progressDialog.setIndeterminate(false);
// 设置ProgressDialog 是否可以按退回按键取消
progressDialog.setCancelable(true);
// 让ProgressDialog显示
progressDialog.show();
```

如果要取消显示

```java
// 取消ProgressDialog显示
progressDialog.cancel();
```

至于按钮监听，由于比较简单，本文不细讲。

## 后记

关于控件学习，还是那句话，官方文档是最好用的。下一节我们讲讲AlertDialog。
