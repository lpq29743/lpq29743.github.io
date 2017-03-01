---
layout:     post
title:      "设计模式之简单工厂模式"
subtitle:   "从最简单的设计模式学起"
date:       2017-03-01 16:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 开启设计模式之旅~~~


## 前言

简单工厂模式并不属于GoF23个经典设计模式，但通常将它作为学习其他工厂模式的基础，今天就让我们来聊一聊它！

---

## 正文

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

- 系统扩展困难，一旦添加新产品就不得不修改工厂逻辑，在产品类型较多时，有可能造成工厂逻辑过于复杂，不利于系统的扩展和维护

- 简单工厂模式由于使用了静态工厂方法，造成工厂角色无法形成基于继承的等级结构。

- 细心的朋友可能早就发现了，这么多if else判断完全是Hard Code啊，如果我有一个新产品要加进来，就要同时添加一个新产品类，并且必须修改工厂类，再加入一个 else if 分支才可以， 这样就违背了 “开放-关闭原则”中的对修改关闭的准则了。当系统中的具体产品类不断增多时候，就要不断的修改工厂类，对系统的维护和扩展不利。那有没有改进的方法呢？在工厂方法模式中会进行这方面的改进。

  　　2.一个工厂类中集合了所有的类的实例创建逻辑，违反了高内聚的责任分配原则，将全部的创建逻辑都集中到了一个工厂类当中，因此一般只在很简单的情况下应用，比如当工厂类负责创建的对象比较少时。

**简单工厂方法适用于什么环境**

- 工厂类负责创建对象较少，不会造成工厂方法业务逻辑太过复杂
- 客户端只知道传入工厂类的参数，不需要关心创建细节

**有哪些例子属于简单工厂方法**

- JDBC。DriverManager是工厂类，应用程序直接使用DriverManager静态方法得到某数据库的Connection

- java.text.DateFormat。用于格式化本地日期或时间
- Java加密技术。获取不同加密算法的密钥生成器以及创建密码器

## 后记

博主自认为这篇文章自己还是写的很不错的，也希望通过这篇文章给设计模式这个系列的文章开个好头。
