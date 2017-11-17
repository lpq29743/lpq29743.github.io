---
layout:     post
title:      "Android控件之ProgressBar"
subtitle:   "进度条控件ProgressBar"
date:       2016-10-04 15:45:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 控件
---

> “网络加载中常用到的控件”


## 前言

继续前一篇文章，这一篇讲讲ProgressBar

---

## 正文

从官方文档上看，为了适应不同的应用环境，Android内置了几种风格的进度条，可以通过Style属性设置ProgressBar的风格。支持如下属性：

- @android:style/Widget.ProgressBar.Horizontal：水平进度条（可以显示刻度，常用）。
- @android:style/Widget.ProgressBar.Small：小进度条。
- @android:style/Widget.ProgressBar.Large：大进度条。
- @android:style/Widget.ProgressBar.Inverse：不断跳跃、旋转画面的进度条。
- @android:style/Widget.ProgressBar.Large.Inverse:不断跳跃、旋转动画的大进度条。
- @android:style/Widget.ProgressBar.Small.Inverse：不断跳跃、旋转动画的小进度条。

只有Widget.ProgressBar.Horizontal风格的进度条，才可以设置进度的递增，其他的风格展示为一个循环的动画，而设置Widget.ProgressBar.Horizontal风格的进度条，需要用到一些属性设置递增的进度，这些属性都有对应的setter、getter方法，这些属性如下：

- android:max：设置进度的最大值。
- android:progress:设置当前第一进度值。
- android:secondaryProgress：设置当前第二进度值。
- android:visibility：设置是否显示，默认显示。

对于Widget.ProgressBar.Horizontal风格的进度条而言，在代码中动态设置移动量，除了可以使用setProgress(int)方法外，Android还为我们提供了另外一个incrementProgressBy(int)方法，它与setProgress(int)的根本区别在于，setProgress(int)是直接设置当前进度值，而incrementProgressBy(int)是设置当前进度值的增量（正数为增，负数为减）。与setProgress(int)和incrementProgressBy(int)对应的还有setSecondaryProgress(int)和incrementSecondaryProgressBy(int)方法，用于设置第二进度值。

## 后记

关于ProgressBar，我们讲的很少，因为它比较常用而且很简单。下一篇我们将会讲一下它的兄弟ProgressDialog
