---
layout: post
title: Android xml tools 使用详解
categories: Android
description: Android xml tools使用详解
keywords: Android, Android xml, Android xml tools
---
---
layout:     post
title:      "Android xml tools使用详解"
subtitle:   "xml预览工具"
date:       2016-10-04 15:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

在写Android布局文件时，经常会碰到要预览效果的问题。相信有一部分人跟笔者一样，一开始在做Android开发的时候，想要看TextView的效果，就设置个android.text="xxx"，然后等把效果改满意后，出于代码洁癖症，又把这一句删掉。这样一来，在xml设计中耗费的时间变多，开发者也很容易失去耐心。那么，怎么解决这个问题呢？当然是本文提到的tools。 

首先需要添加tools命名空间：

```java
xmlns:tools="http://schemas.android.com/tools"
```

接下来就可以使用了

**设置文本**

```java
tools:text="I am a title"
```

**忽略警告**

```java
tools:ignore="contentDescription"
```

**忽略目标API警告**

```java
tools:targetApi="LOLLIPOP"
```

**设置拼写检查**

```java
tools:locale="it"
```

**设置上下文**

```java
tools:context="com.android.example.MainActivity"
```

**设置菜单**

```java
// 当主题为Theme.AppCompat时，这个属性不起作用
tools:menu="menu_main,menu_edit"
  
// 如果你不希望在预览图中显示菜单则：
tools:menu=""
```

**设置app bar模式**

```java
/* 当主题是Theme.AppCompat (r21+, at least) 或者Theme.Material,或者使用了布局包含Toolbar的方式。  该属性也不起作用，只有holo主题才有效。*/
tools:actionBarNavMode="tabs"
```

**设置 listitem, listheader 和 listfooter 属性**

```java
tools:listheader="@layout/list_header"
tools:listitem="@layout/list_item"
tools:listfooter="@layout/list_footer"
```

**设置显示在某个布局里面**

```java
tools:showIn="@layout/activity_main"
```