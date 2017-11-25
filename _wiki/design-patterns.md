---
layout: wiki
title: Design Patterns
categories: Design Patterns
description: 设计模式
keywords: 设计模式
---

### 第一部分 创建型模式

#### 第 1 章 简单工厂模式

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

#### 第 2 章 工厂方法模式

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

#### 第 3 章 抽象工厂模式

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

#### 第 4 章 建造者模式

**什么是建造者模式**

建造者模式将复杂对象的构建与表示分离，使得同样的构建过程可以创建不同表示。它一步一步创建复杂对象，允许用户通过指定复杂对象类型和内容构建它们，而不需知道具体细节。建造者模式有如下角色：

- Product：一般是复杂对象，即创建过程复杂，有较多代码量。产品类可以由一个抽象类与它的不同实现组成，也可以由多个抽象类与他们的实现组成
- Builder：将建造具体过程交与其子类来实现。一般至少有两个抽象方法，一个建造产品，一个返回产品
- ConcreteBuilder：实现抽象类未实现方法，一般是两项任务：组建产品；返回产品
- Director：负责调用适当建造者组建产品，与导演类直接交互的是建造者类

**怎么使用建造者模式**

例：建造者模式可以用于描述KFC如何创建套餐：套餐包含主食（如汉堡、鸡肉卷）和饮料（如果汁、可乐）等，不同套餐有不同组成部分，而服务员可以根据顾客要求，一步一步装配这些部分，构造完整套餐，然后返回给顾客

***步骤一：创建Product***

```java
package com.builder;

public class Meal {

	private String food;
	private String drink;

	public String getFood() {
		return food;
	}

	public void setFood(String food) {
		this.food = food;
	}

	public String getDrink() {
		return drink;
	}

	public void setDrink(String drink) {
		this.drink = drink;
	}

}
```

***步骤二：创建Builder***

```java
package com.builder;

public abstract class MealBuilder {

	Meal meal = new Meal();

	public abstract void buildFood();

	public abstract void buildDrink();

	public Meal getMeal() {
		return meal;
	}

}
```

***步骤三：创建ConcreteBuilder***

MealA类：

```java
package com.builder;

public class MealA extends MealBuilder {

	@Override
	public void buildFood() {
		meal.setDrink("一杯可乐");
	}

	@Override
	public void buildDrink() {
		meal.setFood("一盒薯条");
	}

}
```

MealB类：

```c
package com.builder;

public class MealB extends MealBuilder {

	@Override
	public void buildFood() {
		meal.setDrink("一杯柠檬果汁");
	}

	@Override
	public void buildDrink() {
		meal.setFood("三个鸡翅");
	}

}
```

***步骤四：创建Director***

```c
package com.builder;

public class KFCWaiter {
	private MealBuilder mealBuilder;

	public void setMealBuilder(MealBuilder mealBuilder) {
		this.mealBuilder = mealBuilder;
	}

	public Meal construct() {
		mealBuilder.buildFood();
		mealBuilder.buildDrink();
		return mealBuilder.getMeal();
	}
}
```

***步骤五：创建Client***

```c
package com.builder;

public class Client {
	public static void main(String[] args) {
		KFCWaiter waiter = new KFCWaiter();
		MealA a = new MealA();
		waiter.setMealBuilder(a);
		Meal mealA = waiter.construct();
		System.out.print("套餐A的组成部分:");
		System.out.println(mealA.getFood() + "和" + mealA.getDrink());
	}
}
```

**建造者模式有哪些优缺点**

***优点***

- 客户端不必知道产品内部组成细节，将产品本身与产品创建过程解耦，使得相同创建过程可以创建不同产品
- 具体建造者相对独立，可以很方便地替换具体建造者或增加新的具体建造者
- 将复杂产品的创建步骤分解在不同方法中，使得创建过程更加清晰，也方便使用程序来控制创建过程
- 增加新的具体建造者无须修改原有类库代码，指挥者类针对抽象建造者类编程，系统扩展方便，符合开闭原则

***缺点***

- 创建产品一般具有较多共同点，组成部分相似，如果产品差异性大，则不适合用建造者模式，因此使用范围有限
- 如果产品内部变化复杂，可能会导致需要定义很多具体建造者类来实现这种变化，导致系统庞大

**建造者模式适用于什么环境**

- 需要生成的产品对象有复杂的内部结构，包含多个成员属性
- 需要生成的产品对象的属性相互依赖，需要指定生成顺序
- 对象创建过程独立于创建对象的类。建造者模式引入了指挥者类，将创建过程封装在指挥者类中，而不在建造者类中
- 隔离复杂对象的创建和使用，并使得相同创建过程可以创建不同产品

**有哪些例子属于建造者模式**

- 在很多游戏软件中，地图包括天空、地面等组成部分，人物包括人体、服装等组成部分，可以用建造者模式进行设计，通过不同具体建造者创建不同类型的地图或人物

#### 第 5 章 原型模式

**什么是原型模式**

原型模式用原型实例指定创建对象种类，并通过拷贝原型创建对象，其核心是原型类，Prototype类需具备两个条件：

- 实现Cloneable接口。Java有个Cloneable接口，实现了此接口的类才能被拷贝，否则会抛出异常
- 重写Object的clone方法。Object类clone方法的作用域为protected类型，一般类无法调用，因此需将clone方法作用域改为public类型

**怎么使用原型模式**

例：勺子可分为汤勺、色拉勺子等等，我们可以用原型模式来实现

***步骤一：创建Prototype***

```java
package com.prototype;

public abstract class AbstractSpoon implements Cloneable {

	String spoonName;

	public void setSpoonName(String spoonName) {
		this.spoonName = spoonName;
	}

	public String getSpoonName() {
		return this.spoonName;
	}

	public Object clone() {
		Object object = null;
		try {
			object = super.clone();
		} catch (CloneNotSupportedException exception) {
			System.err.println("AbstractSpoon is not Cloneable");
		}
		return object;
	}
}
```

***步骤二：创建ConcretePrototype***

SoupSpoon类：

```java
package com.prototype;

public class SoupSpoon extends AbstractSpoon {

	public SoupSpoon() {
		setSpoonName("Soup Spoon");
	}

}
```

SaladSpoon类：

```java
package com.prototype;

public class SaladSpoon extends AbstractSpoon {

	public SaladSpoon() {
		setSpoonName("Salad Spoon");
	}

}
```

***步骤三：创建Client***

```java
package com.prototype;

public class Client {
	public static void main(String[] args) {
		SoupSpoon soupSpoon = new SoupSpoon();
		for (int i = 0; i < 5; i++) {
			SoupSpoon cloneSoupSpoon = (SoupSpoon) soupSpoon.clone();
			System.out.println(cloneSoupSpoon.getSpoonName());
		}

		SaladSpoon saladSpoon = new SaladSpoon();
		for (int i = 0; i < 10; i++) {
			SaladSpoon cloneSaladSpoon = (SaladSpoon) saladSpoon.clone();
			System.out.println(cloneSaladSpoon.getSpoonName());
		}
	}
}
```
值得注意的是，使用原型模式复制对象不会调用类的构造方法。因为对象复制是通过调用Object类的clone方法完成的，它直接在内存中复制数据，不会调用到类的构造方法。

**原型模式怎么实现深拷贝**

***Java的深拷贝和浅拷贝***

Java的拷贝情况分为深拷贝和浅拷贝，具体如下：

- 浅拷贝：拷贝对象时仅拷贝对象本身，不拷贝对象包含的引用指向的对象
- 深拷贝：不仅拷贝对象本身，而且拷贝对象包含的引用指向的所有对象

举例说明：对象A1中包含对B1的引用，B1中包含对C1的引用。浅拷贝A1得到A2，A2依然包含对B1的引用，B1依然包含对C1的引用。深拷贝则是对浅拷贝的递归，深拷贝A1得到A2，A2中包含对B2（B1的copy）的引用，B2中包含对C2（C1的copy）的引用。

在上面的例子中，Object类的clone方法只会拷贝对象中的基本数据类型，对于数组、容器对象、引用对象等都不会拷贝，这就是浅拷贝。那么如果我们要实现深拷贝，应该怎么办呢？

***方法一：另行拷贝***

```java
public class Prototype implements Cloneable {
	private ArrayList list = new ArrayList();

	public Prototype clone() {
		Prototype prototype = null;
		try {
			prototype = (Prototype) super.clone();
			prototype.list = (ArrayList) this.list.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return prototype;
	}
}
```

***方法二：深度序列化克隆***

另行拷贝需要对数组、容器对象、引用对象等一一进行拷贝，当数组、容器对象、引用对象等数目较多的时候，程序将会变得很庞大，所以我们采取序列化来实现深拷贝：

```java
package com.prototype;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public abstract class AbstractSpoon implements Serializable {

	/**
	 * 生成的序列号
	 */
	private static final long serialVersionUID = -8348698601667527754L;
  
	String spoonName;

	public void setSpoonName(String spoonName) {
		this.spoonName = spoonName;
	}

	public String getSpoonName() {
		return this.spoonName;
	}

	public Object deepClone() throws IOException, ClassNotFoundException {
		// 将对象写到流里
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream oos = new ObjectOutputStream(bos);
		oos.writeObject(this);
		// 从流里读回来
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		ObjectInputStream ois = new ObjectInputStream(bis);
		return ois.readObject();
	}
  
}
```

**原型模式有哪些优缺点**

***优点***

- 如果新对象比较复杂，原型模式可以简化创建过程，也能提高效率
- 可以使用深克隆保持对象状态
- 提供了简化的创建结构

***缺点***

- 实现深克隆的时候可能需要比较复杂的代码
- 需要为每个类配备克隆方法，而且克隆方法需要对类的功能进行通盘考虑，这对全新的类来说不是很难，但对已有类进行改造时，不一定是件容易的事，必须修改其源代码，违背了开闭原则

**原型模式适用于什么环境**

- 如果创建新对象成本较大，可以利用已有对象进行复制来获得
- 如果系统要保存对象的状态，而对象状态变化小，或本身占内存不大，也可以用原型模式配合备忘录模式。相反，如果对象状态变化大，或占用内存大，那么用状态模式会更好
- 需要避免使用分层次的工厂类来创建分层次的对象，并且类的实例对象只有一个或很少的几个组合状态，通过复制原型对象得到新实例可能比使用构造函数创建新实例更加方便

#### 第 6 章 单例模式

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

在上面的代码中，加关键字`synchronized`是为了避免多个线程同时运行到`if (singleton == null)`，都判断为null，导致创建多个实例，这样子就不是单例了。但是这样写会导致除获得同步锁的线程外的其他所有线程等待，对软件的效率造成了很大影响，所以我们对代码做以下改进：

```java
package com.singleton;

public class LazySingleton {
	private static LazySingleton singleton;

	private LazySingleton(){}

	public static LazySingleton getInstance() {
		if (singleton == null) {
            synchronized(LazySingleton.class) {
                if (singleton == null) {
                    singleton = new LazySingleton();
                }
            }
		}
		return singleton;
	}
}
```

这种方法叫做双重检查，第一个if判断是为了解决上面的效率问题，只有instance为null的时候，才进入synchronized代码段，而第二个if判断是为了避免多个实例的产生。

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

### 第二部分 结构型模式

#### 第 7 章 适配器模式

**什么是适配器模式**

适配器模式将一个类的接口适配成用户所期待的。一个适配器通常允许因为接口不兼容而不能一起工作的类能够在一起工作，做法是将类自己的接口包裹在一个已存在的类中。

**怎么使用适配器模式**

适配器模式有类的适配器模式和对象的适配器模式两种不同的形式。首先我们先来实现一下类适配器：

***步骤一：创建Target***

```java
package com.adapter;

public interface Target {
	
	// 使用的接口
	public void request();
	
	// 已经存在的接口，这个接口需要适配
	public void specificRequest();
	
}
```

***步骤二：创建Adaptee***

```java
package com.adapter;

public class Adaptee {

	// 原本存在的方法
	public void specificRequest() {
		System.out.println("specific request");
	}

}
```

***步骤三：创建Adapter***

```java
package com.adapter;

public class Adapter extends Adaptee implements Target {

	@Override
	public void request() {
		System.out.println("request");
	}

}
```

***步骤四：创建Client***

```java
package com.adapter;

public class Client {
	public static void main(String[] args) {
		Target adapter = new Adapter();
		adapter.request();
	}
}
```

这样子，我们就实现了类适配器，但Java遵循单继承，所以类适配器往往受到限制，在面向对象设计原则中，有条叫做组合/聚合复用原则，指尽可能使用组合和聚合达到复用目的而不是继承，所以一般推荐用对象适配器。其具体实现如下：

***步骤一：创建Target和Adaptee***

这个步骤与上面的步骤一和步骤二基本一致，所以这里省略。

***步骤二：创建Adapter***

```java
package com.adapter;

public class Adapter implements Target {

	private Adaptee adaptee;

	public Adapter(Adaptee adaptee){
		this.adaptee = adaptee;
	}
	
	@Override
	public void request() { 
        System.out.println("request");
	}
	
	@Override
	public void specificRequest() {
		this.adaptee.specificRequest();
	}

}
```

***步骤三：创建Client***

```java
package com.adapter;

public class Client {
	public static void main(String[] args) {
		Adaptee adaptee = new Adaptee();
		Target adapter = new Adapter(adaptee);
		adapter.request();
	}
}
```

**适配器模式有哪些优缺点**

***优点***

- 将目标类和适配者类解耦，通过引入一个适配器类来重用现有的适配者类，而无须修改原有代码
- 增加了类的透明性和复用性，将具体实现封装在适配者类中，对于客户端类来说是透明的，且提高了适配者的复用性
- 灵活性和扩展性好，通过配置文件可以更换适配器，也可以在不修改代码的基础上增加新适配器类，符合开闭原则
- 类适配器模式优点：由于适配器类是适配者类子类，故可在适配器类中置换适配者方法，使得适配器灵活性更强
- 对象适配器模式优点：一个对象适配器可把多个不同适配者适配到同一目标，即可把适配者类及其子类都适配

***缺点***

- 类适配器模式缺点：对于不支持多继承的语言，一次只能适配一个适配者类，而且目标抽象类只能为抽象类，不能为具体类，使用有一定局限性，不能将一个适配者类和它的子类都适配到目标接口
- 对象适配器模式缺点：与类适配器模式相比，要想置换适配者类的方法不容易。如果一定要置换，只好先做一个适配者类的子类，将适配者类方法置换掉，再把适配者类子类当做真正适配者适配

**适配器模式适用于什么环境**

- 系统需要使用现有类，而这些类的接口不符合系统需要
- 想建立一个可重复使用的类，用于与一些彼此之间没太大关联的一些类，包括一些可能将来引进的类一起工作

**有哪些例子属于适配器模式**

- JDBC

#### 第 8 章 桥接模式

**什么是桥接模式**

桥接模式将抽象部分与实现部分分离，使它们独立变化。它是一种对象结构型模式。

**怎么使用桥接模式**

例：车从种类的角度可以分为火车和汽车，从用途的角度可分为客车和货车

***步骤一：创建桥接实现接口***

```java
package com.bridge;

public interface Transport {

	public void transport(); 
	
}
```

***步骤二：创建实现桥接接口的类***

Goods类：

```java
package com.bridge;

public class Goods implements Transport{  
	  
    @Override  
    public void transport() {
        System.out.println("运货");  
    }  
  
}
```

Guest类：

```java
package com.bridge;

public class Guest implements Transport{  
	  
    @Override  
    public void transport() {
        System.out.println("运客");
    }  
  
}
```

***步骤三：创建使用桥接接口的抽象类***

```java
package com.bridge;

public abstract class Vehicle {  
	  
    private Transport implementor;  
      
    public void transport(){  
        implementor.transport();  
    }
    
    public Vehicle(Transport implementor) {
		super();
		this.implementor = implementor;
	}

}
```

***步骤四：创建继承抽象类的实体类***

Car类：

```java
package com.bridge;

public class Car extends Vehicle {  
	  
    public Car(Transport implementor) {
		super(implementor);
	}

	@Override  
    public void transport() {
        System.out.print("汽车");  
        super.transport();  
    }
  
}
```

Train类：

```java
package com.bridge;

public class Train extends Vehicle {

	public Train(Transport implementor) {
		super(implementor);
	}

	@Override
	public void transport() {
		System.out.print("火车");
		super.transport();
	}

}
```

***步骤五：创建Client***

```java
package com.bridge;

public class Client {  
	  
    public static void main(String[] args) {
    	
        Train train1 = new Train(new Goods());
        train1.transport();
        Train train2 = new Train(new Guest());
        train2.transport();  
          
        Car car1 = new Car(new Goods());
        car1.transport();
        Car car2 = new Car(new Guest());
        car2.transport();   
  
    }
  
}
```

**桥接模式有哪些优缺点**

***优点***

- 分离抽象接口及其实现部分
- 桥接模式有时类似多继承，但多继承违背类的单一职责原则（即一个类只有一个变化），复用性较差，且类个数庞大，桥接模式比多继承更好
- 桥接模式提高系统的可扩充性，在两个变化维度中任意扩展一个维度，都不需修改原有系统
- 实现细节对客户透明，可对用户隐藏实现细节

***缺点***

- 桥接模式引入会增加系统的理解与设计难度，由于聚合关联关系建立在抽象层，要求开发者针对抽象设计与编程
- 桥接模式要求正确识别出系统中两个独立变化的维度，因此使用范围具有一定局限性

**桥接模式适用于什么环境**

- 如果系统需要在构件的抽象化角色和具体化角色之间增加更多灵活性，避免在两个层次之间建立静态的继承联系，通过桥接模式可以使它们在抽象层建立一个关联关系
- 抽象化角色和实现化角色可以以继承的方式独立扩展而互不影响，在程序运行时可以动态将一个抽象化子类的对象和一个实现化子类的对象进行组合，即系统需要对抽象化角色和实现化角色进行动态耦合
- 一个类存在两个独立变化的维度，且这两个维度都需要扩展
- 虽然系统中使用继承没有问题，但由于抽象化角色和具体化角色需要独立变化，设计要求需要独立管理这两者
- 对于那些不希望使用继承或因为多继承导致系统类的个数急剧增加的系统，桥接模式尤为适用

#### 第 9 章 组合模式

**什么是组合模式**

组合模式把一组相似对象当作一个单一对象，依据树形结构组合对象，用来表示部分以及整体层次，包括以下角色：

- 抽象构建角色（component）：作为抽象角色，给组合对象的统一接口
- 树叶构建角色（leaf）：代表组合对象中的树叶对象
- 树枝构建角色（composite）：参加组合的所有子对象的对象，并给出树枝构建对象的行为

**怎么使用组合模式**

例：公司员工包括普通员工和领导

***步骤一：创建Component***

```java
package com.bridge;

public interface Transport {

	public void transport(); 
	
}
```

***步骤二：创建Leaf***

```java
package com.composite;

public class Employee implements Worker {

	private String name;

	public Employee(String name) {
		super();
		this.name = name;
	}

	@Override
	public void doSomething() {
		System.out.println(toString());
	}

	@Override
	public String toString() {
		return "我叫" + getName() + "，就一普通员工!";
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

}
```

***步骤三：创建Composite***

```java
package com.composite;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class Leader implements Worker {
	private List<Worker> workers = new CopyOnWriteArrayList<Worker>();
	private String name;

	public Leader(String name) {
		super();
		this.name = name;
	}

	public void add(Worker worker) {
		workers.add(worker);
	}

	public void remove(Worker worker) {
		workers.remove(worker);
	}

	public Worker getChild(int i) {
		return workers.get(i);
	}

	@Override
	public void doSomething() {
		System.out.println(toString());
		Iterator<Worker> it = workers.iterator();
		while (it.hasNext()) {
			it.next().doSomething();
		}

	}

	@Override
	public String toString() {
		return "我叫" + getName() + "，我是一个领导,有 " + workers.size() + "下属。";
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

}
```

***步骤四：创建Client***

```java
package com.composite;

public class Client {

    public static void main(String[] args) {
        Leader leader1 = new Leader("张三");  
        Leader leader2 = new Leader("李四");  
        Employee employe1 = new Employee("王五");  
        Employee employe2 = new Employee("赵六");  
        Employee employe3 = new Employee("陈七");  
        Employee employe4 = new Employee("徐八");  
        leader1.add(leader2);  
        leader1.add(employe1);  
        leader1.add(employe2);  
        leader2.add(employe3);  
        leader2.add(employe4);  
        leader1.doSomething();     
    }

}
```

**组合模式有哪些优缺点**

***优点***

- 清楚定义分层次的复杂对象，表示对象的全部或部分层次，使得增加新构件更容易
- 客户端调用简单，可以一致的使用组合结构或单个对象
- 定义了包含叶子对象和容器对象的类层次结构，叶子对象可以被组合成更复杂的容器对象，而这个容器对象又可以被组合，不断递归，形成复杂的树形结构
- 更容易在组合体内加入对象构件，客户端不必因为加入了新的对象构件而更改原有代码

***缺点***

- 使设计更抽象，如果对象业务规则很复杂，则实现组合模式具有挑战性，且不是所有方法都与叶子对象子类有关联

**桥接模式适用于什么环境**

- 需要表示对象整体或部分层次，在具有整体和部分的层次结构中，希望忽略整体与部分的差异，可以一致对待
- 让客户能够忽略不同对象层次的变化，可以针对抽象构件编程，无须关心对象层次结构的细节

#### 第 10 章 装饰模式

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

#### 第 10 章 外观模式

**什么是外观模式**

外观模式为子系统中的一组接口提供一个一致的界面，定义了一个高层接口，使得子系统更易使用。

**怎么使用外观模式**

例：现在有一辆汽车，我们要启动它，就要发动引擎，使四个车轮转动。但实际上我们只要踩下油门，汽车就可以被启动了。

***步骤一：创建子系统***

Wheel类：

```java
package com.decorator;

public interface Shape {
	
	void draw();
	
}
```

Engine类：

```java
package com.facade;

public class Engine {
	public String EngineWork() {
		return "BMW's Engine is Working";
	}

	public String EngineStop() {
		return "BMW's Engine is stoped";
	}
}
```

Body类：

```java
package com.facade;

public class Body {
	public Wheel[] wheels = new Wheel[4];
	public Engine engine = new Engine();

	public Body() {
		for (int i = 0; i < wheels.length; i++) {
			wheels[i] = new Wheel();
		}
	}
}
```

***步骤二：创建Facade类***

```java
package com.facade;

public class CarFacade {
	Body body = new Body();

	public void Run() {
		System.out.println(body.engine.EngineWork());
		for (int i = 0; i < body.wheels.length; i++) {
			System.out.println(body.wheels[i].WheelCircumrotate());
		}
	}

	public void Stop() {
		System.out.println(body.engine.EngineStop());
		for (int i = 0; i < body.wheels.length; i++) {
			System.out.println(body.wheels[i].WheelStop());
		}
	}
}
```

***步骤三：创建Client***

```java
package com.facade;

public class Client {

	public static void main(String[] args) {
		CarFacade car = new CarFacade();
        car.Run();
        car.Stop();
	}

}
```

**外观模式有哪些优缺点**

***优点***

- 对客户屏蔽子系统组件，减少了客户处理的对象数目并使子系统更加容易使用。通过引入外观模式，客户端代码变得很简单，与之关联的对象也很少
- 实现了子系统与客户之间的松耦合关系，子系统的组件变化不会影响到调用它的客户类，只需调整外观类即可
- 降低了大型软件系统中的编译依赖性，并简化了系统在不同平台之间的移植过程，因为编译一个子系统一般不需要编译所有其他的子系统。子系统的修改对其他子系统没有影响，子系统的内部变化也不会影响到外观对象
- 只是提供了一个访问子系统的统一入口，不影响用户直接使用子系统类

***缺点***

- 不能很好地限制客户使用子系统类，如果对客户访问子系统类做太多的限制，则减少了可变性和灵活性
- 在不引入抽象外观类的情况下，增加新的子系统可能需要修改外观类或客户端的源代码，违背了开闭原则

**外观模式适用于什么环境**

- 当要为一个复杂子系统提供一个简单接口时。该接口可以满足大多数用户的需求，而且用户也可以越过外观类直接访问子系统
- 客户程序与多个子系统之间存在很大依赖性。引入外观类将子系统与客户及其他子系统解耦，提高子系统的独立性和可移植性
- 在层次化结构中，可以使用外观模式定义系统中每层的入口，层与层之间不直接产生联系，而通过外观类建立联系，降低层之间的耦合度

#### 第 11 章 享元模式

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

#### 第 12 章 代理模式

**什么是代理模式**

当客户不想或不能直接引用一个对象时，可以通过代理实现间接引用，这就是代理模式，代理模式包括以下角色：

- Subject：抽象主题角色
- Proxy：代理主题角色
- RealSubject：真实主题角色

**怎么使用代理模式**

***步骤一：创建Subject***

```java
package com.proxy;

public interface Image {
	void display();
}
```

***步骤二：创建RealSubject***

```java
package com.proxy;

public class RealImage implements Image {

	private String fileName;

	public RealImage(String fileName) {
		this.fileName = fileName;
		loadFromDisk(fileName);
	}

	@Override
	public void display() {
		System.out.println("Displaying " + fileName);
	}

	private void loadFromDisk(String fileName) {
		System.out.println("Loading " + fileName);
	}
	
}
```

***步骤三：创建Proxy***

```java
package com.proxy;

public class ProxyImage implements Image {

	private RealImage realImage;
	private String fileName;

	public ProxyImage(String fileName) {
		this.fileName = fileName;
	}

	@Override
	public void display() {
		if (realImage == null) {
			realImage = new RealImage(fileName);
		}
		realImage.display();
	}
	
}
```

***步骤四：创建Client***

```java
package com.proxy;

public class Client {

	public static void main(String[] args) {
		Image image = new ProxyImage("test.jpg");
		image.display();
		image.display();
	}

}
```

**代理模式有哪些优缺点**

***优点***

- 协调调用者和被调用者，一定程度降低了系统耦合度
- 远程代理使得客户端可以访问远程机器上的对象，远程机器可能具有更好的计算性能与处理速度，可以快速响应并处理客户端请求
- 虚拟代理通过使用一个小对象来代表一个大对象，可以减少系统资源消耗，对系统进行优化并提高运行速度
- 保护代理可以控制对真实对象的使用权限

***缺点***

- 由于在客户端和真实主题之间增加了代理对象，因此有些类型的代理模式可能会造成请求的处理速度变慢
- 实现代理模式需要额外的工作，有些代理模式的实现非常复杂

**代理模式适用于什么环境**

- 远程代理：为一个位于不同地址空间的对象提供一个本地代理对象
- 虚拟代理：如果要创建一个资源消耗大的对象，先创建一个消耗较小的对象表示，真实对象只在需要时才被创建
- Copy-on-Write代理：虚拟代理的一种，把复制操作延迟到客户端真正需要时。一般来说，对象的深克隆是开销较大，Copy-on-Write代理可以让操作延迟，只有对象被用到时才克隆
- 保护代理：控制对对象的访问，可以给不同用户提供不同级别的使用权限
- 缓冲代理：为某一目标操作的结果提供临时存储空间，以便多个客户端共享这些结果
- 防火墙代理：保护目标不让恶意用户接近
- 同步化代理：使几个用户能够同时使用一个对象而没有冲突
- 智能引用代理：当对象被引用时，提供额外的操作，如将此对象被调用的次数记录下来等

### 第三部分 行为型模式

#### 第 13 章 责任链模式

**什么是责任链模式**

责任链模式为请求创建了一个接收者对象的链，在这种模式中，通常每个接收者都包含对另一个接收者的引用。如果一个对象不能处理该请求，那么它会把相同的请求传给下一个接收者，依此类推。责任链模式包括以下角色：

在这种模式中，通常每个接收者都包含对另一个接收者的引用。如果一个对象不能处理该请求，那么它会把相同的请求传给下一个接收者，依此类推。，代理模式包括以下角色：

- 抽象处理者(Handler)角色：处理请求的接口。如果需要，接口可以定义一个方法以设定和返回对下家的引用
- 具体处理者(ConcreteHandler)角色：具体处理者接到请求后，可以处理请求或传给下家。由于具体处理者持有对下家的引用，因此可以访问下家

**怎么使用责任链模式**

***步骤一：创建Handler***

```java
package com.chainofresponsibility;

public abstract class AbstractLogger {
	public static int INFO = 1;
	public static int DEBUG = 2;
	public static int ERROR = 3;

	protected int level;

	protected AbstractLogger nextLogger;

	public void setNextLogger(AbstractLogger nextLogger) {
		this.nextLogger = nextLogger;
	}

	public void logMessage(int level, String message) {
		if (this.level <= level) {
			write(message);
		}
		if (nextLogger != null) {
			nextLogger.logMessage(level, message);
		}
	}

	abstract protected void write(String message);

}
```

***步骤二：创建ConcreteHandler***

ConsoleLogger类

```java
package com.chainofresponsibility;

public class ConsoleLogger extends AbstractLogger {

	public ConsoleLogger(int level) {
		this.level = level;
	}

	@Override
	protected void write(String message) {
		System.out.println("Standard Console::Logger: " + message);
	}
	
}
```

ErrorLogger类：

```java
package com.chainofresponsibility;

public class ErrorLogger extends AbstractLogger {

	public ErrorLogger(int level) {
		this.level = level;
	}

	@Override
	protected void write(String message) {
		System.out.println("Error Console::Logger: " + message);
	}
	
}
```

FileLogger类：

```java
package com.chainofresponsibility;

public class FileLogger extends AbstractLogger {

	public FileLogger(int level) {
		this.level = level;
	}

	@Override
	protected void write(String message) {
		System.out.println("File::Logger: " + message);
	}

}
```

***步骤三：创建Client***

```java
package com.chainofresponsibility;

public class Client {

	private static AbstractLogger getChainOfLoggers() {
		AbstractLogger errorLogger = new ErrorLogger(AbstractLogger.ERROR);
		AbstractLogger fileLogger = new FileLogger(AbstractLogger.DEBUG);
		AbstractLogger consoleLogger = new ConsoleLogger(AbstractLogger.INFO);
		
		errorLogger.setNextLogger(fileLogger);
		fileLogger.setNextLogger(consoleLogger);
		
		return errorLogger;
	}

	public static void main(String[] args) {
		AbstractLogger loggerChain = getChainOfLoggers();
		loggerChain.logMessage(AbstractLogger.INFO, "This is an information.");
		loggerChain.logMessage(AbstractLogger.DEBUG, "This is an debug level information.");
		loggerChain.logMessage(AbstractLogger.ERROR, "This is an error information.");
	}

}
```

**责任链模式有哪些优缺点**

***优点***

- 降低耦合度。将请求的发送者和接受者解耦
- 简化对象。对象不需知道链的结构
- 增强给对象指派职责的灵活性。通过改变链内的成员或者调动它们的次序，允许动态新增或删除责任
- 增加新的请求处理类很方便

***缺点***

- 不能保证请求一定被接收
- 系统性能将受到一定影响，且进行代码调试时不方便，可能会造成循环调用
- 可能不容易观察运行时的特征，有碍于除错

**责任链模式适用于什么环境**

- 有多个对象可以处理同一个请求，具体哪个对象处理由运行时刻自动确定
- 在不明确指定接收者的情况下，向多个对象中的一个提交请求
- 动态指定一组对象处理请求

#### 第 14 章 命令模式

**什么是命令模式**

命令模式将一个请求封装成一个对象，从而使你可用不同的请求对客户进行参数化，它对请求排队或记录请求日志，以及支持可撤销的操作。命令模式包括以下角色：

- Command：抽象类，声明需要执行的命令，一般要对外公布一个execute方法用来执行命令
- ConcreteCommand：Command类的实现类，对抽象类中声明的方法进行实现
- Invoker：调用者，负责调用命令
- Receiver：接收者，负责接收命令并执行命令
- Client：最终的客户端调用类

**怎么使用命令模式**

***步骤一：创建Receiver***

```java
package com.command;

public class Receiver {

    public void action(){
        System.out.println("执行操作");
    }
    
}
```

***步骤二：创建Command***

```java
package com.command;

public interface Command {
	void execute();
}
```

***步骤三：创建ConcreteCommand***

```java
package com.command;

public class ConcreteCommand implements Command {

	private Receiver receiver = null;

	public ConcreteCommand(Receiver receiver) {
		this.receiver = receiver;
	}

	@Override
	public void execute() {
		receiver.action();
	}

}
```

***步骤四：创建Invoker***

```java
package com.command;

public class Invoker {

	private Command command = null;

	public Invoker(Command command) {
		this.command = command;
	}

	public void action() {
		command.execute();
	}

}
```

***步骤五：创建Client***

```java
package com.command;

public class Client {

	public static void main(String[] args) {
		Receiver receiver = new Receiver();
		Command command = new ConcreteCommand(receiver);
		Invoker invoker = new Invoker(command);
		invoker.action();
	}

}
```

**命令模式有哪些优缺点**

***优点***

- 降低系统耦合度
- 新的命令可以很容易地加入到系统中
- 可以比较容易地设计一个命令队列和宏命令（组合命令）
- 可以方便地实现对请求的Undo和Redo

***缺点***

- 可能会导致某些系统有过多具体命令类。因为每个命令都需要一个具体命令类，因此系统可能需要大量具体命令类

**命令模式适用于什么环境**

- 系统需要将请求调用者和请求接收者解耦，使调用者和接收者不直接交互
- 系统需要在不同时间指定请求、将请求排队和执行请求
- 系统需要支持命令的撤销(Undo)操作和恢复(Redo)操作
- 系统需要将一组操作组合在一起，即支持宏命令

#### 第 15 章 解释器模式

**什么是解释器模式**

给定一个语言之后，解释器模式可以定义出其文法的一种表示，并同时提供一个解释器。客户端可以使用这个解释器来解释这个语言中的句子。解释器模式包括以下角色：

- Expression：抽象表达式。声明一个所有具体表达式角色都需实现的接口，包含interpret()方法
- Terminal Expression：终结符表达式。实现抽象表达式接口，每个终结符都有一个具体终结表达式与之相对应
- Nonterminal Expression：非终结符表达式。文法中的每条规则都需要一个具体的非终结符表达式
- Context：上下文。用来存放文法中各个终结符所对应的具体值

**怎么使用解释器模式**

***步骤一：创建Expression***

```java
package com.interpreter;

public interface Expression {
	public boolean interpret(String context);
}
```

***步骤二：创建TerminalExpression***

```java
package com.interpreter;

public class TerminalExpression implements Expression {

	private String data;

	public TerminalExpression(String data) {
		this.data = data;
	}

	@Override
	public boolean interpret(String context) {
		if (context.contains(data)) {
			return true;
		}
		return false;
	}
	
}
```

***步骤三：创建NonterminalExpression***

OrExpression类：

```java
package com.interpreter;

public class OrExpression implements Expression {

	private Expression expr1 = null;
	private Expression expr2 = null;

	public OrExpression(Expression expr1, Expression expr2) {
		this.expr1 = expr1;
		this.expr2 = expr2;
	}

	@Override
	public boolean interpret(String context) {
		return expr1.interpret(context) || expr2.interpret(context);
	}
	
}
```

AndExpression类：

```java
package com.interpreter;

public class AndExpression implements Expression {

	private Expression expr1 = null;
	private Expression expr2 = null;

	public AndExpression(Expression expr1, Expression expr2) {
		this.expr1 = expr1;
		this.expr2 = expr2;
	}

	@Override
	public boolean interpret(String context) {
		return expr1.interpret(context) && expr2.interpret(context);
	}
	
}
```

***步骤四：创建Client***

```java
package com.interpreter;

public class InterpreterPatternDemo {

	public static Expression getMaleExpression() {
		Expression robert = new TerminalExpression("Robert");
		Expression john = new TerminalExpression("John");
		return new OrExpression(robert, john);
	}

	public static Expression getMarriedWomanExpression() {
		Expression julie = new TerminalExpression("Julie");
		Expression married = new TerminalExpression("Married");
		return new AndExpression(julie, married);
	}

	public static void main(String[] args) {
		Expression isMale = getMaleExpression();
		Expression isMarriedWoman = getMarriedWomanExpression();

		System.out.println("John is male? " + isMale.interpret("John"));
		System.out.println("Julie is a married women? " + isMarriedWoman.interpret("Married Julie"));
	}

}
```

**解释器模式有哪些优缺点**

***优点***

- 可扩展性较好，灵活
- 增加了新的解释表达式的方式
- 易于实现文法

***缺点***

- 执行效率较低，可利用场景较少
- 对于复杂的文法比较难维护

**解释器模式适用于什么环境**

- 可将一个需要解释执行的语言中的句子表示为一个抽象语法树
- 一些重复出现的问题可以用一种简单的语言来进行表达
- 文法较为简单

#### 第 16 章 迭代器模式

**什么是迭代器模式**

迭代器模式提供一种方法顺序访问一个聚合对象中各个元素，而又不需暴露该对象的内部表示，它包括以下角色：

- Iterator：迭代器。定义遍历元素所需接口
- ConcreteIterator：具体迭代器。实现了Iterator接口，并保持迭代过程中的游标位置
- Aggregate：聚合。此抽象角色给出创建迭代器对象的接口
- ConcreteAggregate：具体聚合。实现了创建迭代器对象的接口，返回一个合适的具体迭代器实例

**怎么使用迭代器模式**

***步骤一：创建Iterator***

```java
package com.iterator;

public interface Iterator {
	
	public void first();

	public void next();

	public boolean isDone();

	public Object currentItem();
	
}
```

***步骤二：创建Aggregate***

```java
package com.iterator;

public abstract class Aggregate {
    public abstract Iterator createIterator();
}
```

***步骤三：创建ConcreteAggregate***

```java
package com.iterator;

public class ConcreteAggregate extends Aggregate {

	private Object[] objArray = null;

	public ConcreteAggregate(Object[] objArray) {
		this.objArray = objArray;
	}

	@Override
	public Iterator createIterator() {

		return new ConcreteIterator(this);
	}

	public Object getElement(int index) {

		if (index < objArray.length) {
			return objArray[index];
		} else {
			return null;
		}
	}

	public int size() {
		return objArray.length;
	}

}
```

***步骤四：创建ConcreteIterator***

```java
package com.iterator;

public class ConcreteIterator implements Iterator {

	private ConcreteAggregate agg;
	private int index = 0;
	private int size = 0;

	public ConcreteIterator(ConcreteAggregate agg) {
		this.agg = agg;
		this.size = agg.size();
		index = 0;
	}

	@Override
	public Object currentItem() {
		return agg.getElement(index);
	}

	@Override
	public void first() {
		index = 0;
	}

	@Override
	public boolean isDone() {
		return (index >= size);
	}

	@Override
	public void next() {
		if (index < size) {
			index++;
		}
	}

}
```

***步骤五：创建Client***

```java
package com.iterator;

public class Client {

	public void operation() {
		Object[] objArray = { "One", "Two", "Three", "Four", "Five", "Six" };
		Aggregate agg = new ConcreteAggregate(objArray);
		Iterator it = agg.createIterator();
		while (!it.isDone()) {
			System.out.println(it.currentItem());
			it.next();
		}
	}

	public static void main(String[] args) {
		Client client = new Client();
		client.operation();
	}

}
```

**迭代器模式有哪些优缺点**

***优点***

- 支持以不同方式遍历聚合对象
- 简化了聚合类
- 在同一聚合上可以有多个遍历
- 在迭代器模式中，增加新的聚合类和迭代器类很方便，无须修改原有代码

***缺点***

- 由于迭代器模式将存储数据和遍历数据的职责分离，增加新的聚合类需增加新的迭代器类，类的个数成对增加，一定程度增加了系统复杂性

**迭代器模式适用于什么环境**

- 访问聚合对象的内容而无须暴露它的内部表示
- 需要为聚合对象提供多种遍历方式
- 为遍历不同聚合结构提供统一接口

#### 第 17 章 中介者模式

**什么是中介者模式**

中介者模式值得是用一个中介对象来封装一系列的对象交互。中介者使各对象不需要显式地相互引用，从而使其耦合松散，而且可以独立地改变它们之间的交互。中介者模式包括以下角色：

- Mediator：中介者。定义好同事类对象到中介者对象的接口，用于各个同事类之间的通信
- ConcreteMediator：具体中介者。从抽象中介者继承而来，实现抽象中介者中定义的事件方法。
- Colleague Class：同事类。如果一个对象会影响其他对象，同时也会被其他对象影响，那么这两个对象称为同事类。在中介者模式中，同事类之间必须通过中介者才能进行消息传递

**怎么使用中介者模式**

***步骤一：创建Mediator***

```java
package com.mediator;

public abstract class Mediator {
	public abstract void constact(String message, Person person);
}
```

***步骤二：创建ConcreteMediator***

```java
package com.mediator;

public class MediatorStructure extends Mediator {

	private HouseOwner houseOwner;
	private Tenant tenant;

	public HouseOwner getHouseOwner() {
		return houseOwner;
	}

	public void setHouseOwner(HouseOwner houseOwner) {
		this.houseOwner = houseOwner;
	}

	public Tenant getTenant() {
		return tenant;
	}

	public void setTenant(Tenant tenant) {
		this.tenant = tenant;
	}

	public void constact(String message, Person person) {
		if (person == houseOwner) {
			tenant.getMessage(message);
		} else {
			houseOwner.getMessage(message);
		}
	}
}
```

***步骤三：创建抽象同事类***

```java
package com.mediator;

public abstract class Person {
	
	protected String name;
	protected Mediator mediator;

	Person(String name, Mediator mediator) {
		this.name = name;
		this.mediator = mediator;
	}

}
```

***步骤四：创建具体同事类***

HouseOwner类：

```java
package com.mediator;

public class HouseOwner extends Person {

	HouseOwner(String name, Mediator mediator) {
		super(name, mediator);
	}

	public void constact(String message) {
		mediator.constact(message, this);
	}

	public void getMessage(String message) {
		System.out.println("房主:" + name + ",获得信息：" + message);
	}
}
```

Tenant类：

```java
package com.mediator;

public class Tenant extends Person {

	Tenant(String name, Mediator mediator) {
		super(name, mediator);
	}

	public void constact(String message) {
		mediator.constact(message, this);
	}

	public void getMessage(String message) {
		System.out.println("租房者:" + name + ",获得信息：" + message);
	}

}
```

***步骤五：创建Client***

```java
package com.mediator;

public class Client {

	public static void main(String[] args) {
		MediatorStructure mediator = new MediatorStructure();
		HouseOwner houseOwner = new HouseOwner("张三", mediator);
		Tenant tenant = new Tenant("李四", mediator);
		mediator.setHouseOwner(houseOwner);
		mediator.setTenant(tenant);
		tenant.constact("听说你那里有三室的房主出租.....");
		houseOwner.constact("是的!请问你需要租吗?");
	}

}
```

**中介者模式有哪些优缺点**

***优点***

- 简化对象之间的关系，将系统的各个对象之间的相互关系进行封装，将各个同事类解耦，使系统成为松耦合系统
- 减少了子类的生成
- 可以减少各同事类的设计与实现

***缺点***

- 由于中介者对象封装了系统中对象之间的相互关系，导致其变得非常复杂，使得系统维护比较困难

**迭代器模式适用于什么环境**

- 系统中对象之间存在比较复杂的引用关系，导致他们之间的依赖关系结构混乱且难以复用该对象
- 想通过一个中间类封装多个类中的行为，而又不想生成太多子类

#### 第 18 章 责任链模式

#### 第 19 章 责任链模式

#### 第 20 章 责任链模式

#### 第 21 章 责任链模式

#### 第 22 章 责任链模式

#### 第 23 章 责任链模式

#### 第 24 章 责任链模式