---
layout:     post
title:      "设计模式之享元模式"
subtitle:   "一个可以提高性能的设计模式"
date:       2017-04-26 22:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 当对象和类变得越来越多~~~
>


## 前言

面向对象技术可以解决一些灵活性或可扩展性的问题，但需要在系统中增加类和对象的个数。当对象数量太多时，将导致运行代价过高，带来性能下降等问题。为了解决这个问题，我们引入享元模式这一结构型模式。

---

## 正文

**什么是享元模式**

享元模式运用共享技术有效支持大量细粒度对象的复用，包含了以下角色：

- Flyweight：抽象享元类
- ConcreteFlyweight：具体享元类
- UnsharedConcreteFlyweight：非共享具体享元类
- FlyweightFactory：享元工厂类

**怎么使用享元模式**

***步骤一：创建Flyweight***

```java
package com.flyweight;

public interface Shape {
	void draw();
}
```

***步骤二：创建ConcreteFlyweight***

```java
package com.flyweight;

public class Circle implements Shape {
	private String color;
	private int x;
	private int y;
	private int radius;

	public Circle(String color) {
		this.color = color;
	}

	public void setX(int x) {
		this.x = x;
	}

	public void setY(int y) {
		this.y = y;
	}

	public void setRadius(int radius) {
		this.radius = radius;
	}

	@Override
	public void draw() {
		System.out.println("Circle: Draw() [Color : " + color + ", x : " + x + ", y :" + y + ", radius :" + radius);
	}
}
```

***步骤三：创建FlyweightFactory***

```java
package com.flyweight;

import java.util.HashMap;

public class ShapeFactory {
	private static final HashMap<String, Shape> circleMap = new HashMap<String, Shape>();

	public static Shape getCircle(String color) {
		Circle circle = (Circle) circleMap.get(color);

		if (circle == null) {
			circle = new Circle(color);
			circleMap.put(color, circle);
			System.out.println("Creating circle of color : " + color);
		}
		return circle;
	}
}
```

***步骤四：创建Client***

```java
package com.flyweight;

public class Client {
	private static final String colors[] = { "Red", "Green", "Blue", "White", "Black" };

	public static void main(String[] args) {

		for (int i = 0; i < 20; ++i) {
			Circle circle = (Circle) ShapeFactory.getCircle(getRandomColor());
			circle.setX(getRandomX());
			circle.setY(getRandomY());
			circle.setRadius(100);
			circle.draw();
		}
	}

	private static String getRandomColor() {
		return colors[(int) (Math.random() * colors.length)];
	}

	private static int getRandomX() {
		return (int) (Math.random() * 100);
	}

	private static int getRandomY() {
		return (int) (Math.random() * 100);
	}
}
```

**享元模式有哪些优缺点**

***优点***

- 极大减少内存中对象的数量，使得相同对象或相似对象在内存中只保存一份
- 享元模式的外部状态相对独立，且不影响其内部状态，使得享元对象可以在不同环境中被共享

***缺点***

- 享元模式使系统更加复杂，需要分离出内部状态和外部状态，使得程序的逻辑复杂化
- 为了使对象可以共享，享元模式需将享元对象的状态外部化，而读取外部状态使得运行时间变长


**享元模式适用于什么环境**

- 一个系统有大量相同或相似对象，由于这类对象的大量使用，造成内存大量耗费
- 对象的大部分状态都可以外部化，可以将这些外部状态传入对象中
- 使用享元模式需维护一个存储享元对象的享元池，而这要耗费资源，因此应当在多次使用享元对象时才使用享元模式


## 后记

是否合理地使用享元模式对程序的性能起到了很重要的影响，所以必须根据实际情况判断是否使用享元模式。
