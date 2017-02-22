---
layout:     post
title:      "Java的细枝末节"
subtitle:   "聊聊Java的小知识点"
date:       2017-02-22 13:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Java
---

> 不定期更新的Java笔记~~


## 前言

上个月差不多这个时候，我创建了一篇关于Android知识点的博客，而最近由于阿里巴巴Java开发手册的发布和学习上遇到的问题，所以决定也弄一篇总结Java小知识点的博客。同样地，这篇文章也是不定期更新。

---

## 正文

**for和foreach**

这两种都是常见的遍历方式。相对来说，for使用更频繁，效率更高。容易被忽略的是：foreach只有读取作用，无法修改元素，具体可以看我曾提出的[这个问题](http://stackoverflow.com/questions/37135364/error-occured-while-using-java-foreach-statements)。另外，不要在foreach循环中进行元素的remove/add操作，具体如下：

```java
List<String> a = new ArrayList<>();
a.add("1");
a.add("2");
for (String temp : a) {
	if ("1".equals(temp)) {
		a.remove(temp);
	}
}
```

至于原因，可以查看[这篇博客](http://rongmayisheng.com/post/%E7%A0%B4%E9%99%A4%E8%BF%B7%E4%BF%A1java-util-arraylist%E5%9C%A8foreach%E5%BE%AA%E7%8E%AF%E9%81%8D%E5%8E%86%E6%97%B6%E5%8F%AF%E4%BB%A5%E5%88%A0%E9%99%A4%E5%85%83%E7%B4%A0)。所以，remove元素一般使用Iterator方式，如果并发操作，需要对Iterator加锁。

## 后记




