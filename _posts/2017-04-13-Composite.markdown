---
layout:     post
title:      "设计模式之组合模式"
subtitle:   "设计模式里面的树结构"
date:       2017-04-13 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 设计模式
---

> 又一个对象结构型模式~~~
>


## 前言

上一篇我们了解了桥接模式，这一篇让我们来了解同样是对象结构型模式的组合模式

---

## 正文

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


## 后记

组合模式跟我们数据结构中的树巧妙地联系在了一起，显得十分有趣，也有利于我们对它的学习。
