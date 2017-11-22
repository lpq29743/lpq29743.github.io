---
layout: wiki
title: Design Patterns
categories: Design Patterns
description: 设计模式
keywords: 设计模式
---

#### 第 1 部分 创建型模式

### 第 1 章 简单工厂模式

**什么是简单工厂模式**

简单工厂模式又称为静态工厂方法模式，它属于类创建型模式。在简单工厂模式中，可以根据参数返回不同类的实例。简单工厂模式定义一个类负责创建其他类的实例，被创建实例具有共同父类。在简单工厂模式有如下角色：

- Factory（工厂角色）：即工厂类，负责实现创建产品实例的内部逻辑；可以被外界调用创建所需对象；提供了静态工厂方法`factoryMethod()`，它的返回类型为抽象产品类型Product
- Product（抽象产品角色）：工厂类所创建对象的父类，封装了各种产品对象的公有方法
- ConcreteProduct（具体产品角色）：简单工厂模式的创建目标，需要实现抽象方法

**怎么使用简单工厂模式**

例：现在要创建多个不同图形，这些图形源自同个父类，继承父类后适当修改而呈现了不同外观。如果希望使用图形时，不需知道具体类名，只需知道表示该图形的参数，即可返回相应图形。此时就可以使用简单工厂模式。

***步骤一：创建Product***

```java
package com.simplefactory;

public abstract class Shape {
	
	public void methodSame() {
		// 公共方法的实现
	}

	public abstract void draw();

}
```

***步骤二：创建ConcreteProduct***

Circle类：

```java
package com.simplefactory;

public class Circle extends Shape {

	@Override
	public void draw() {
		System.out.println("circle");
	}

}
```

Square类：

```java
package com.simplefactory;

public class Square extends Shape {

	@Override
	public void draw() {
		System.out.println("square");
	}

}
```

***步骤三：创建Factory***

```java
package com.simplefactory;

public class SimpleFactory {
	
	public static Shape createProduct(String product) {
		if (product.equals("circle")) {
			return new Circle();
		} else if (product.equals("square")) {
			return new Square();
		} else {
			System.out.println("null");
			return null;
		}
	}
	
}
```

***步骤四：创建Client***

```java
package com.simplefactory;

public class Client {

	public static void main(String[] args) {
		SimpleFactory.createProduct("circle").draw();
		SimpleFactory.createProduct("square").draw();
	}

}
```

**怎么改进简单工厂方法**

当我们要修改创建产品时，我们都要修改客户端代码的参数，违反了开闭原则。对于这个问题，我们可用配置文件解决：

***步骤一：创建config.xml***

```xml
<?xml version="1.0"?>
<config>
    <type>circle</type>
</config>
```

***步骤二：创建XMLUtil***

```java
package com.simplefactory;

import javax.xml.parsers.*;
import org.w3c.dom.*;
import java.io.*;

public class XMLUtil {

	public static String getType() {
		try {
			DocumentBuilderFactory dFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder builder = dFactory.newDocumentBuilder();
			Document doc;
			doc = builder.parse(new File("config.xml"));

			NodeList nl = doc.getElementsByTagName("type");
			Node classNode = nl.item(0).getFirstChild();
			String type = classNode.getNodeValue().trim();
			return type;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
}
```

***步骤三：修改Client***

```java
package com.simplefactory;

public class Client {

	public static void main(String[] args) {
		String type = XMLUtil.getType();
		SimpleFactory.createProduct(type).draw();
	}

}
```

这样子我们就只需改配置文件，无须改任何代码，就能符合开闭原则了。但是新的问题又出现了，如果我们要扩展，就要修改工厂类，也会违反开闭原则。那我们应该怎么解决呢？答案是使用反射机制，具体如下：

***步骤一：创建factory.properties***

```properties
circle = com.simplefactory.Circle
square = com.simplefactory.Square
```

***步骤二：创建PropertiesUtil***

```java
package com.simplefactory;

import java.io.FileInputStream;
import java.util.Properties;

public class PropertiesUtil {
	
	public static Properties getPro() {  
        Properties pro = new Properties();  
        try {  
            pro.load(new FileInputStream("factory.properties"));  
        } catch (Exception e) {  
            e.printStackTrace();  
        }  
        return pro;  
    }  

}
```

***步骤三：修改SimpleFactory***

```java
package com.simplefactory;

import java.util.Properties;

public class SimpleFactory {

	public static Shape createProduct(String product) {
		Properties pro = PropertiesUtil.getPro();
		String className = pro.getProperty(product);
		try {
			Class<?> c = Class.forName(className);
			return (Shape) c.newInstance();
		} catch (ClassNotFoundException e) {
			System.out.println("This class doesn't exsist!");
			e.printStackTrace();
		} catch (InstantiationException e) {
			System.out.println("This class can't be instantiated!");
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		}
		return null;
	}

}
```

这里还可以用注解解决扩展问题，具体的可以上网查找相关资料。这种方式不需使用到类的全名，更为简洁。

**简单工厂模式与OOP原则有什么关系**

设计模式和设计原则的关系就像是三十六计与孙子兵法的关系，学习设计模式就必须学习OOP设计原则！

***OOP的方法论***

- 代码重用（通常用继承和聚合实现）
- 低耦合（模块与模块之间，类与类之间依赖程度低）
- 高内聚（模块或类内部依赖程度高，负责处理相关的工作）
- 易扩充（在不改变原有代码，或改变很小的情况下增加功能）
- 易维护（代码结构清晰，容易管理和修改代码）

***OOP设计原则***

- 开闭原则（OCP，Open-Closed Principle）：对扩展开放、对修改封闭
- 依赖倒置原则（DIP，Dependency-Inversion Principles）：高层模块不依赖底层模块，两者都依赖抽象
- 里氏替换原则（LSP，Liskov Substitution Principle）：当子类能够替换掉基类，基类才真正被复用
- 接口隔离原则（ISP，Interface Insolation Principle）：接口功能单一，避免接口污染
- 单一职责原则（SRP，Single Resposibility Principle）：一个类应该仅有一个引起他变化的原因
- 迪米特法则（LoD ，Law of Demeter）：对象只和最直接的类交互，对第三方可通过转达交互，降低对象间的耦合度
- 合成/聚合复用原则（CARP，Composite/Aggregate Reuse Principle）：尽量使用合成/聚合，不要使用继承

***简单工厂模式已遵循原则***

- 依赖倒置原则
- 迪米特法则
- 里氏替换原则
- 接口隔离原则

***简单工厂模式未遵循原则***

- 开闭原则（利用配置文件+反射或注解可避免这一点）
- 单一职责原则（工厂类即要负责逻辑判断又要负责实例创建）

**简单工厂模式有哪些优缺点**

***优点***

- 工厂类含有判断逻辑，决定创建哪个实例，客户端可以免除创建责任，而仅仅使用产品，实现了责任分割
- 客户端无须知道具体产品类的类名，只需知道对应参数，可以减少使用者记忆量
- 通过引入配置文件，可在不修改代码的情况下更换和增加新的具体产品类，高了系统的灵活性

***缺点***

- 由于工厂类集中了所有产品创建逻辑，一旦不能正常工作，整个系统都要受到影响

- 使用简单工厂模式将会增加系统中类的个数，在一定程序上增加了系统的复杂度和理解难度

- 系统扩展困难，一旦添加新产品就不得不修改工厂逻辑，产品类型较多时有可能造成工厂逻辑过于复杂

- 简单工厂模式由于使用了静态工厂方法，造成工厂角色无法形成基于继承的等级结构


**简单工厂模式适用于什么环境**

- 工厂类负责创建对象较少，不会造成工厂方法业务逻辑太过复杂
- 客户端只知道传入工厂类的参数，不需要关心创建细节

**有哪些例子属于简单工厂模式**

- JDBC。DriverManager是工厂类，应用程序直接使用DriverManager静态方法得到某数据库的Connection

- java.text.DateFormat。用于格式化本地日期或时间
- Java加密技术。获取不同加密算法的密钥生成器以及创建密码器

### 第 2 章 工厂方法模式

**什么是工厂方法模式**

在简单工厂模式中，所有产品都由一个工厂创建，职责较重，业务逻辑复杂，而工厂方法模式则可以解决这一问题。工厂方法模式又称工厂模式、多态工厂模式或虚拟构造器模式，它针对不同产品提供不同工厂，包含了以下角色：

- **Product（抽象产品）**：定义产品接口，是产品对象的公共父类
- **ConcreteProduct（具体产品）**：实现了抽象产品接口，具体产品由具体工厂创建
- **Factory（抽象工厂）**：声明了工厂方法(Factory Method)，用于返回一个产品，所有工厂类都必须实现该接口
- **ConcreteFactory（具体工厂）**：抽象工厂类的子类，实现了工厂方法，并由客户端调用，返回具体产品实例

**怎么使用工厂方法模式**

***步骤一：创建Product***

```java
package com.factorymethod;

public interface Shape {

	public void draw();

}
```

***步骤二：创建ConcreteProduct***

Circle类：

```java
package com.factorymethod;

public class Circle implements Shape {

	@Override
	public void draw() {
		System.out.println("circle");
	}

}
```

Square类：

```java
package com.factorymethod;

public class Square implements Shape {

	@Override
	public void draw() {
		System.out.println("square");
	}

}
```

***步骤三：创建Factory***

```java
package com.factorymethod;

public abstract class Factory {

	public abstract Shape createShape();

}
```

***步骤四：创建ConcreteFactory***

CircleFactory类：

```java
package com.factorymethod;

public class CircleFactory extends Factory {
	
	public Shape createShape() {
		return new Circle();
	}

}
```

SquareFactory类：

```java
package com.factorymethod;

public class SquareFactory extends Factory {
	
	public Shape createShape() {
		return new Square();
	}

}
```

***步骤五：创建Client***

```java
package com.factorymethod;

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

### 第 3 章 抽象工厂模式

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

### 第 4 章 文件系统

### 第 5 章 I/O 系统

### 第 6 章 死锁