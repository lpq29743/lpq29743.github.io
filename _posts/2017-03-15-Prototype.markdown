---
layout:     post
title:      "设计模式之原型模式"
subtitle:   "没有什么是拷贝解决不了的"
date:       2017-03-15 18:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 设计模式里的“真假美猴王”


## 前言

这篇文章我们将介绍原型模式，它也是属于创建型模式的一种。话不多说，让我们开始这篇文章吧！

---

## 正文

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

## 后记

原型模式是一种比较简单的模式，也容易理解，实现一个接口，重写一个方法即完成了原型模式。实际应用中，原型模式很少单独出现，经常与其他模式混用，原型类Prototype也常用抽象类来替代。
