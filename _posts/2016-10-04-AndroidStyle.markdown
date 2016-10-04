---
layout:     post
title:      "Android style笔记"
subtitle:   "了解了解style"
date:       2016-10-04 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> style——自定义Theme和减少代码书写必备


## 前言

style在Android开发中经常会使用到，今天这篇博客的主题便是style。

---

## 正文

***通过style定义Theme***

```java
<style name="simple_dialog" parent="@android:style/Theme.Dialog">
	<item name="android:windowFrame">@null</item><!-- Dialog的WindowFrame框为无 -->
	<item name="android:windowNoTitle">true</item>
	<item name="android:windowIsFloating">true</item><!-- 是否悬浮在activity上 -->
	<item name="android:windowIsTranslucent">true</item><!-- 是否半透明 -->
	<item name="android:backgroundDimEnabled">false</item><!-- 背景是否模糊 -->
</style>
```

***应用Theme***

```java
// 应用到Application
<application android:theme="@style/CustomTheme">
  
// 应用到Activity
<activity android:theme="@android:style/Theme.Dialog">
```

***通过style减少代码书写***

```java
<style name="CodeFont" parent="@android:style/TextAppearance.Medium">
	<item name="android:layout_width">fill_parent</item>
	<item name="android:layout_height">wrap_content</item>
	<item name="android:textColor">#00FF00</item>
	<item name="android:typeface">monospace</item>
</style>
```

***应用到相应控件***

```java
<TextView
    style="@style/CodeFont"
    android:text="@string/hello" />
```

## 后记

看完了这篇博客，有没有感到style的方便之处呢？

今天一下子就码了几篇博客，是时候该休息一下了！毕竟一天内写太多博客，博客的质量也会降低。