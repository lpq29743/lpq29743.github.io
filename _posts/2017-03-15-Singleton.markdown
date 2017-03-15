---
layout:     post
title:      "设计模式之单例模式"
subtitle:   "最后一个创建型模式"
date:       2017-03-15 20:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 对不起，我只给一个


## 前言

这篇文章我们将介绍最后一个创建型模式——单例模式。利用英语的一句话描述就是：Last but not the least！

---

## 正文

**什么是单例模式**

单例模式确保类只有一个实例，而且自行实例化并向系统提供这个实例。它包括以下要素：

- 私有的构造方法
- 指向自己实例的私有静态引用
- 以自己实例为返回值的静态的公有的方法

**怎么使用单例模式**

单例模式主要有两种：饿汉式单例和懒汉式单例。前者加载时实例化；后者调用取得实例方法时实例化。

***饿汉式单例***

```java
package com.singleton;

public class HungrySingleton {

	private static HungrySingleton singleton = new HungrySingleton();

	private HungrySingleton(){}

	public static HungrySingleton getInstance() {
		return singleton;
	}
	
}
```

***懒汉式单例***

```java
package com.singleton;

public class LazySingleton {
	private static LazySingleton singleton;

	private LazySingleton(){}

	public static synchronized LazySingleton getInstance() {
		if (singleton == null) {
			singleton = new LazySingleton();
		}
		return singleton;
	}
}
```

饿汉式单例和懒汉式单例由于构造方法是private的，所以都不可继承，但是很多单例模式是可继承的，如登记式单例。在Java中，饿汉式单例要优于懒汉式单例，而C++中一般使用懒汉式单例。

**单例模式有哪些优缺点**

***优点***

- 提供了对唯一实例的受控访问。单例类封装了它的唯一实例，所以它可以严格控制客户怎样以及何时访问它，并为设计及开发团队提供了共享的概念
- 由于内存中只存在一个对象，因此可以节约资源，对于需要频繁创建和销毁的对象，单例模式可以提高系统性能
- 允许可变数目的实例。我们可以基于单例模式进行扩展，使用与单例控制相似的方法来获得指定个数的对象实例

***缺点***

- 由于单例模式中没有抽象层，因此单例类扩展有很大困难
- 单例类职责过重，一定程度违背了单一职责原则。单例类既充当了工厂角色，提供了工厂方法，又充当了产品角色，包含业务方法，将产品创建和本身功能融合到一起
- 滥用单例将带来负面问题，如为了节省资源将数据库连接池对象设计为单例类，可能会导致共享连接池对象的程序过多而出现连接池溢出；另外有一种争议说法，如果实例化对象长时间不被利用，系统会认为它是垃圾，自动销毁并回收资源，下次利用又将重新实例化，这将导致对象状态丢失，关于这个争论可以查看[这里](http://wiki.jikexueyuan.com/project/java-design-pattern/singleton-discuss.html)

**单例模式适用于什么环境**

- 系统只需要一个实例对象，如要求一个唯一序列号生成器，或需要考虑资源消耗而只允许创建一个对象
- 客户调用类的单个实例只允许使用一个公共访问点，除了该公共访问点，不能通过其他途径访问该实例
- 要求类只有一个实例时应用单例模式。如果类可以有几个实例共存，就需对单例模式改进，使之成为多例模式

**有哪些例子属于单例模式**

- 一个具有自动编号主键的表可以有多个用户同时使用，但数据库中只能有一个地方分配下一个主键编号，否则会出现主键重复，因此该主键编号生成器必须具备唯一性，可以通过单例模式来实现


## 后记

单例模式虽然普遍认为是23种设计模式中最简单的一种，但是却应用广泛，所以必须得很好地掌握。
