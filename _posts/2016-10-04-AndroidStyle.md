---
layout: post
title: Android style 笔记
categories: Android
description: Android style笔记
keywords: Android, Android style, 自定义Theme
---

style在Android开发中经常会使用到，今天这篇博客的主题便是style。

**通过style定义Theme**

```xml
<style name="simple_dialog" parent="@android:style/Theme.Dialog">
	<item name="android:windowFrame">@null</item><!-- Dialog的WindowFrame框为无 -->
	<item name="android:windowNoTitle">true</item>
	<item name="android:windowIsFloating">true</item><!-- 是否悬浮在activity上 -->
	<item name="android:windowIsTranslucent">true</item><!-- 是否半透明 -->
	<item name="android:backgroundDimEnabled">false</item><!-- 背景是否模糊 -->
</style>
```

**应用Theme**

```java
// 应用到Application
<application android:theme="@style/CustomTheme">
  
// 应用到Activity
<activity android:theme="@android:style/Theme.Dialog">
```

**通过style减少代码书写**

```xml
<style name="CodeFont" parent="@android:style/TextAppearance.Medium">
	<item name="android:layout_width">fill_parent</item>
	<item name="android:layout_height">wrap_content</item>
	<item name="android:textColor">#00FF00</item>
	<item name="android:typeface">monospace</item>
</style>
```

**应用到相应控件**

```java
<TextView
    style="@style/CodeFont"
    android:text="@string/hello" />
```