---
layout:     post
title:      "设计模式之抽象工厂模式"
subtitle:   "工厂模式里的BOSS"
date:       2017-03-01 20:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 终于来到了最后一个工厂模式~~~


## 前言

工厂方法模式的每个工厂只生产一类产品，会导致存在大量工厂类，增加系统开销，此时可以考虑使用抽象工厂模式。

---

## 正文

**什么是抽象工厂模式**

与工厂方法模式相比，抽象工厂模式的具体工厂不只创建一种产品，而是一族产品。其包含以下角色：

- **AbstractFactory（抽象工厂）**：声明了一组用于创建一族产品的方法，每一个方法对应一种产品
- **ConcreteFactory（具体工厂）**：实现了抽象工厂中创建产品的方法，生成一组具体产品
- **AbstractProduct（抽象产品）**：为每种产品声明接口，在抽象产品中声明了产品所具有的业务方法
- **ConcreteProduct（具体产品）**：定义具体工厂生产的具体产品对象，实现抽象产品接口中声明的业务方法

**怎么使用抽象工厂模式**

***步骤一：创建AbstractProduct***

Shape类：

```java
package com.abstractfactory;

public interface Shape {

	public void draw();

}
```

Color类：

```java
package com.abstractfactory;

public interface Color {

	public void fill();

}
```

***步骤二：创建ConcreteProduct***

Circle类：

```java
package com.abstractfactory;

public class Circle implements Shape {

	@Override
	public void draw() {
		System.out.println("circle");
	}

}
```

Square类：

```java
package com.abstractfactory;

public class Square implements Shape {

	@Override
	public void draw() {
		System.out.println("square");
	}

}
```

Red类：

```java
package com.abstractfactory;

public class Red implements Color {

	@Override
	public void fill() {
		System.out.println("red");
	}

}
```

White类：

```java
package com.abstractfactory;

public class White implements Color {

	@Override
	public void fill() {
		System.out.println("white");
	}

}
```

***步骤三：创建Factory***

```java
package com.abstractfactory;

public abstract class AbstractFactory {

	public abstract Color getColor(String color);

	public abstract Shape getShape(String shape);

}
```

***步骤四：创建ConcreteFactory***

ShapeFactory类：

```java
package com.abstractfactory;

public class ShapeFactory extends AbstractFactory{
	
    @Override
	public Color getColor(String color) {
        return null;
    }

    @Override
    public Shape getShape(String shapeType) {
        if(shapeType == null){
             return null;
          }        
          if(shapeType.equalsIgnoreCase("CIRCLE")){
             return new Circle();
          }else if(shapeType.equalsIgnoreCase("SQUARE")){
             return new Square();
          }
          return null;
    }
    
}
```

ColorFactory类：

```java
package com.abstractfactory;

public class ColorFactory extends AbstractFactory {

	@Override
	public Color getColor(String color) {
		if (color == null) {
			return null;
		}
		if (color.equalsIgnoreCase("RED")) {
			return new Red();
		} else if (color.equalsIgnoreCase("WHITE")) {
			return new White();
		}
		return null;
	}

	@Override
	public Shape getShape(String shape) {
		return null;
	}

}
```

***步骤五：创建FactoryProducer***

```java
package com.abstractfactory;

public class FactoryProducer {

	public static AbstractFactory getFactory(String choice) {

		if (choice.equalsIgnoreCase("SHAPE")) {
			return new ShapeFactory();
		} else if (choice.equalsIgnoreCase("COLOR")) {
			return new ColorFactory();
		}
		return null;
	}

}
```

***步骤六：创建Client***

```java
package com.abstractfactory;

public class Client {

	public static void main(String[] args) {
		AbstractFactory shapeFactory = FactoryProducer.getFactory("SHAPE");
		Shape shape1 = shapeFactory.getShape("CIRCLE");
		shape1.draw();

		Shape shape2 = shapeFactory.getShape("SQUARE");
		shape2.draw();

		AbstractFactory colorFactory = FactoryProducer.getFactory("COLOR");
		Color color1 = colorFactory.getColor("RED");
		color1.fill();

		Color color2 = colorFactory.getColor("WHITE");
		color2.fill();
	}

}
```

**抽象工厂模式与OOP原则有什么关系**

***抽象工厂模式已遵循原则***

- 依赖倒置原则
- 迪米特法则
- 里氏替换原则
- 接口隔离原则
- 单一职责原则（每个工厂只负责创建自己的具体产品族，没有简单工厂中的逻辑判断）
- 开闭原则（增加新的产品族，不像简单工厂那样需要修改已有的工厂，而只需增加相应的具体工厂类）

***抽象工厂模式未遵循原则***

- 开闭原则（虽然工厂对修改关闭了，但更换产品时，客户代码还是需要修改）

**抽象工厂模式有哪些优缺点**

***优点***

- 抽象工厂模式隔离了具体类的生成，客户不需知道什么被创建。由于这种隔离，更换具体工厂变得相对容易。所有具体工厂都实现了抽象工厂定义的公共接口，因此只需改变具体工厂实例，就可以改变整个系统的行为
- 当产品族中多个对象被设计成一起工作时，它保证客户端始终只使用同一个产品族中的对象
- 增加新的具体工厂和产品族很方便，无须修改已有系统，符合开闭原则

***缺点***

- 难以扩展抽象工厂来生产新种类的产品，因为抽象工厂角色规定了所有可能被创建的产品集合，要支持新产品意味着要对该接口进行扩展，而这涉及到抽象工厂角色及其子类的修改，会带来较大不便
- 开闭原则的倾斜性（增加新的工厂和产品族容易，增加新的产品等级结构麻烦）

**抽象工厂模式适用于什么环境**

- 一个系统不应当依赖于产品类实例如何被创建、组合和表达的细节，这对于所有类型的工厂模式都是重要的
- 系统中有多个产品族，而每次只使用其中某一产品族
- 属于同一个产品族的产品将在一起使用，这一约束必须在系统的设计中体现出来
- 系统提供一个产品类的库，所有的产品以同样接口出现，使客户端不依赖于具体实现

**有哪些例子属于工厂方法模式**

- 系统中需要更换界面主题，要求界面中的按钮、文本框等一起改变时，可使用抽象工厂模式


## 后记

抽象工厂模式实现了高内聚低耦合，因此应用广泛。这一篇写完后，只剩下两种创建型模式，我们接下来再找时间讲解。
