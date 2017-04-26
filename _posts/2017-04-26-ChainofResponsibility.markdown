---
layout:     post
title:      "设计模式之责任链模式"
subtitle:   "第一个行为型模式"
date:       2017-04-26 23:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> “击鼓传花”
>


## 前言

从这篇文章开始，我们就从结构型模式转移到了行为型模式，而第一个介绍的就是责任链模式。

---

## 正文

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


## 后记

责任链模式的优缺点的十分突出，如何根据具体情况选择是否使用此模式，同样需要我们掌握。
