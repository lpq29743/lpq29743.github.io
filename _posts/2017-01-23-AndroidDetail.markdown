---
layout:     post
title:      "Android的细枝末节"
subtitle:   "聊聊Android的小知识点"
date:       2017-01-23 19:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> 不定期更新的Android笔记~~


## 前言

本人在大一学年开始接触到Android开发，主要是因为做项目的需要，由于投入了较多的时间和精力，所以对这项开发技术多多少少存在一些情感。虽然现在小程序、WebApp等的出现给Android开发市场带来了不小的挑战，但出于兴趣，本人还是会坚持Android学习，尽管将来很大程度上不以Android为就业方向，但相信在这个学习、研究和实践的过程中，一定能够收获到很多东西。Android这门开发语言设计到的大框架、大技术值得每一位想要深入Android开发的朋友进行学习熟悉，但一些小的细节，一些常用到的东西也同样需要引起我们的关注。如果博主把这些小的知识点放在一起的话，未免显得有些奇怪，所以博主希望通过这篇文章，不定期的记录一些Android的小知识点。

---

## 正文

**EditView软键盘弹出**

对于跳转新界面就要弹出软键盘的情况，可能会出现由于界面未加载完全而无法弹出软键盘的情况。此时应该适当的延迟弹出软键盘（保证界面的数据加载完成）。实例代码如下：

```java
Timer timer = new Timer();
timer.schedule(new TimerTask() {

	public void run() {
		InputMethodManager inputManager = (InputMethodManager) mSearchEt.getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
		inputManager.showSoftInput(mSearchEt, 0);
	}

}, 300);
```

## 后记




