---
layout:     post
title:      "设计模式之模板方法模式"
subtitle:   "让一切按规范来"
date:       2017-06-01 17:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 套上模板好做事
>


## 前言

为了提高代码的复用性和系统的灵活性，我们经常使用到模板方法模式，今天就让我们一起来学一下它！

---

## 正文

**什么是模板方法模式**

在模板方法模式中，一个抽象类公开定义了执行它的方法的模板。它的子类可按需要重写方法实现，但调用将以抽象类中定义的方式进行。模板方法模式包含如下两个角色：

- AbstractClass：抽象类。定义一系列基本操作并实现了一个模板方法
- ConcreteClass：具体子类。实现父类中声明的抽象操作，也可以覆盖在父类中已实现的具体操作

**怎么使用模板方法模式**

***步骤一：创建AbstractClass***

```java
package com.templatemethod;

public abstract class Game {
	abstract void initialize();

	abstract void startPlay();

	abstract void endPlay();

	public final void play() {
		initialize();
		startPlay();
		endPlay();
	}
}
```

***步骤二：创建ConcreteClass***

Cricket类：

```java
package com.templatemethod;

public class Cricket extends Game {

	@Override
	void endPlay() {
		System.out.println("Cricket Game Finished!");
	}

	@Override
	void initialize() {
		System.out.println("Cricket Game Initialized! Start playing.");
	}

	@Override
	void startPlay() {
		System.out.println("Cricket Game Started. Enjoy the game!");
	}
}
```

Football类：

```java
package com.templatemethod;

public class Football extends Game {

	@Override
	void endPlay() {
		System.out.println("Football Game Finished!");
	}

	@Override
	void initialize() {
		System.out.println("Football Game Initialized! Start playing.");
	}

	@Override
	void startPlay() {
		System.out.println("Football Game Started. Enjoy the game!");
	}
}
```

***步骤三：创建Client***

```java
package com.templatemethod;

public class Client {
	public static void main(String[] args) {
		Game game = new Cricket();
		game.play();
		System.out.println();
		game = new Football();
		game.play();
	}
}
```

**模板方法模式有哪些优缺点**

***优点***

- 封装不变部分，扩展可变部分
- 提取公共代码，便于维护
- 行为由父类控制，子类实现


***缺点***

- 每一个不同的实现都需要一个子类来实现，导致类的个数增加，使得系统更加庞大



**模板方法模式适用于什么环境**

- 有多个子类共有的方法，且逻辑相同
- 重要的、复杂的方法，可以考虑作为模板方法


## 后记

模板方法算是近期学习的设计模式里面最简单的了，但即使如此，也要好好掌握！
