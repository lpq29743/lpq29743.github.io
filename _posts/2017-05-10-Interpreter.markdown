---
layout:     post
title:      "设计模式之解释器模式"
subtitle:   "很难却很少使用的设计模式"
date:       2017-05-10 19:40:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 没有哪个设计模式是可以缺少的
>


## 前言

正如小标题所说，解释器模式相对其他设计模式使用较少，且相对较难，但为了设计模式系列文章的完整，笔者还是决定撰写此文。

---

## 正文

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


## 后记

虽然解释器模式运用少，实现难，但对它的学习还是能帮助我们理解很多问题的。如果暂时学习不下，也可以先停一下，等学完后面的设计模式再回过头来学习。
