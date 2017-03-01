---
layout:     post
title:      "设计模式之工厂方法模式"
subtitle:   "第一个GoF23设计模式"
date:       2017-03-01 19:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 简单工厂模式进化版


## 前言

简单工厂模式虽然简单，但不优化的情况下违背了开闭原则，这种情况下，工厂方法模式应运而生。

---

## 正文

**什么是工厂方法模式**

在简单工厂模式中，所有产品都由一个工厂创建，职责较重，业务逻辑复杂，而工厂方法模式则可以解决这一问题。工厂方法模式又称工厂模式、多态工厂模式或虚拟构造器模式，它针对不同产品提供不同工厂，包含了以下角色：

- **Product（抽象产品）**：定义产品接口，是产品对象的公共父类
- **ConcreteProduct（具体产品）**：实现了抽象产品接口，具体产品由具体工厂创建
- **Factory（抽象工厂）**：声明了工厂方法(Factory Method)，用于返回一个产品，所有工厂类都必须实现该接口
- **ConcreteFactory（具体工厂）**：抽象工厂类的子类，实现了工厂方法，并由客户端调用，返回具体产品实例

**怎么使用工厂方法模式**

***步骤一：创建Product***

```java
package factorymethod;

public interface Shape {

	public void draw();

}
```

***步骤二：创建ConcreteProduct***

Circle类：

```java
package factorymethod;

public class Circle implements Shape {

	@Override
	public void draw() {
		System.out.println("circle");
	}

}
```

Square类：

```java
package factorymethod;

public class Square implements Shape {

	@Override
	public void draw() {
		System.out.println("square");
	}

}
```

***步骤三：创建Factory***

```java
package factorymethod;

public abstract class Factory {

	public abstract Shape createShape();

}
```

***步骤四：创建ConcreteFactory***

CircleFactory类：

```java
package factorymethod;

public class CircleFactory extends Factory {
	
	public Shape createShape() {
		return new Circle();
	}

}
```

SquareFactory类：

```java
package factorymethod;

public class SquareFactory extends Factory {
	
	public Shape createShape() {
		return new Square();
	}

}
```

***步骤五：创建Client***

```java
package factorymethod;

public class Client {

	public static void main(String[] args) {
		Factory factory = null;

		factory = new CircleFactory();
		factory.createShape().draw();

        factory = new SquareFactory();
        factory.createShape().draw();
	}

}
```

**工厂方法模式与OOP原则有什么关系**

***工厂方法模式已遵循原则***

- 依赖倒置原则
- 迪米特法则
- 里氏替换原则
- 接口隔离原则
- 单一职责原则（每个工厂只负责创建自己的具体产品，没有简单工厂中的逻辑判断）
- 开闭原则（增加新的产品，不像简单工厂那样需要修改已有的工厂，而只需增加相应的具体工厂类）

***工厂方法模式未遵循原则***

- 开闭原则（虽然工厂对修改关闭了，但更换产品时，客户代码还是需要修改）

**工厂方法模式有哪些优缺点**

***优点***

- 因为每个具体工厂类只负责创建产品，没有简单工厂中的逻辑判断，符合单一职责原则
- 与简单工厂模式不同，工厂方法不使用静态工厂方法，可以形成基于继承的等级结构
- 新增产品只需要增加相应的具体产品类和相应的工厂子类即可，更符合开闭原则

***缺点***

- 添加新产品时，除了增加新产品类外，还要提供具体工厂类，类个数成对增加，增加系统复杂度和开销
- 虽然保证了工厂方法内的对修改关闭，但对于使用工厂方法的类，如果要换用产品，仍需修改实例化的具体工厂
- 一个具体工厂只能创建一种具体产品

**工厂方法模式适用于什么环境**

- 客户端不知道它所需对象的类。在工厂方法模式中，客户端不需要知道具体类名，只需知道对应工厂即可
- 抽象工厂类通过子类指定创建哪个对象。在工厂方法模式中，抽象工厂类只提供创建接口，而由子类确定具体对象

**有哪些例子属于工厂方法模式**

- 日志记录器。某日志记录器要求支持多种日志形式，如文件记录、数据库记录，且可以动态选择日志记录方式

## 后记

工厂方法模式作为GoF的第一个模式，还是能解决蛮多问题的，下一篇我们将讲解抽象工厂模式。
