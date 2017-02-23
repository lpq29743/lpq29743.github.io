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

**a=a+b与a+=b的区别**

首先先看下面一段代码：

```java
public class Test {

	public static void main(String args[]) {
		int a = 2;
		float b = 6;
		a = a + b; // error
		a = (int) (a + b); // right
		a += b; // right
	}
}
```

上面的这段代码中，第三句无法编译，因为a+b后的数据类型时float，无法向下转换为int，所以需要进行类型强制转换，如第4句，而第五句则是采用+=符号，+=具有自动类型转换的功能，所以可以编译。

同样情况还出现在：`byte a = 127; byte b = 127; b = a + b; // error : cannot convert from int to byte b += a; // ok`。因为无论a+b为多少，都会将a、b提升为int，所以将int类型赋给byte就会编译出错。

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

**java中的++操作符线程安全吗**

不是线程安全的操作。它涉及到多个指令，如读取变量、增加和存储回内存，这个过程可能会出现多个线程交差的情况。

**调用System.gc()会发生什么**

通知GC开始工作，但GC真正开始时间不确定。

## 后记

Java这些年越来越火，运用越来越广，但切不可因为这样，而忽略那些最基本的东西！


