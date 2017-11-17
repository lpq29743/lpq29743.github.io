---
layout:     post
title:      "设计模式之装饰模式"
subtitle:   "为对象披上一层华丽的外衣"
date:       2017-04-13 21:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 对象也需要装饰！
>


## 前言

本文涉及到的装饰模式也是一个对象结构型模式，话不对说，让我们正式开始！

---

## 正文

**什么是装饰模式**

装饰模式通过装饰类动态的给对象添加额外的职责，包含以下角色：

- 抽象构件（component）：用来规范被装饰对象，一般用接口方式给出
- 具体构件（concrete component）：被装饰的类
- 抽象装饰类（decorator）：持有一个构件对象的实例，并定义一个跟抽象构件一致的接口

- 具体装饰类（concrete decorator）：给构件添加附加职责，实际使用中装饰角色和具体装饰角色可能由一个类承担

**怎么使用装饰模式**

***步骤一：创建抽象构件***

```java
package com.decorator;

public interface Shape {
	
	void draw();
	
}
```

***步骤二：创建具体构件***

Circle类：

```java
package com.decorator;

public class Circle implements Shape {

	@Override
	public void draw() {
		System.out.println("Shape: Circle");
	}
	
}
```

Rectangle类：

```java
package com.decorator;

public class Rectangle implements Shape {

	@Override
	public void draw() {
		System.out.println("Shape: Rectangle");
	}
	
}
```

***步骤三：创建抽象装饰类***

```java
package com.decorator;

public abstract class ShapeDecorator implements Shape {
	
	protected Shape decoratedShape;

	public ShapeDecorator(Shape decoratedShape) {
		this.decoratedShape = decoratedShape;
	}

	public void draw() {
		decoratedShape.draw();
	}
	
}
```

***步骤四：创建具体装饰类***

```java
package com.decorator;

public class RedShapeDecorator extends ShapeDecorator {

	public RedShapeDecorator(Shape decoratedShape) {
		super(decoratedShape);
	}

	@Override
	public void draw() {
		decoratedShape.draw();
		setRedBorder(decoratedShape);
	}

	private void setRedBorder(Shape decoratedShape) {
		System.out.println("Border Color: Red");
	}
	
}
```

***步骤五：创建Client***

```java
package com.decorator;

public class Client {

	public static void main(String[] args) {
		Shape circle = new Circle();
		Shape redCircle = new RedShapeDecorator(new Circle());
		Shape redRectangle = new RedShapeDecorator(new Rectangle());
		
		System.out.println("Circle with normal border");
		circle.draw();
		System.out.println("\nCircle of red border");
		redCircle.draw();
		System.out.println("\nRectangle of red border");
		redRectangle.draw();
	}

}
```

**装饰模式有哪些优缺点**

***优点***

- 装饰模式与继承关系的目的都是扩展对象功能，但装饰模式可提供比继承更多的灵活性
- 可以通过一种动态方式来扩展对象功能，通过配置文件可在运行时选择不同装饰器，从而实现不同行为
- 通过使用不同具体装饰类以及这些装饰类的排列组合，可以创造出很多不同行为的组合
- 具体构件类与具体装饰类可独立变化，用户可根据需要增加新的具体构件类和具体装饰类，使用时再对其进行组合，原有代码无须改变，符合开闭原则

***缺点***

- 使用装饰模式进行系统设计时将产生很多小对象，这些对象的区别在于它们之间相互连接的方式不同，而不是它们的类或者属性值不同，同时还将产生很多具体装饰类。这些装饰类和小对象的产生将增加系统复杂度
- 这种比继承更加灵活的特性，同时意味着更易出错，排错困难，对于多次装饰的对象，寻找错误要逐级排查


**装饰模式适用于什么环境**

- 在不影响其他对象的情况下，以动态透明的方式给单个对象添加职责
- 需要动态地给对象增加功能，这些功能也可以动态撤销
- 当不能采用继承扩充系统或继承不利于系统扩展维护时。不能采用继承的情况有两类：一是系统存在大量独立扩展，为支持每一种组合将产生大量子类，使得子类数目爆炸性增长；二是因为类定义不能继承（如final类）


## 后记

装饰模式和继承模式都有着各自的优劣，如何根据具体情况，选择合适的解决方法，也是我们需要掌握的。
