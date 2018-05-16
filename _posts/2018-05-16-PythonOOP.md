---
layout: post
title: Python 面向对象编程
categories: Python
description: Python面向对象编程
keywords: Python
---

首先先从一个简单地定义一个类 Human：

```python
class Human(object):
    pass
```

`Human`为类名，括号里的`object`为继承的父类。在 Python 中，`object`为所有类都会继承到的类。当然，这里语法也允许括号留空或者去掉括号。如果要使用这个类，就要创建一个实例，具体为`Tom = Human()`。

人类会有很多的属性，那如何在类中加入属性呢？继续看下一个程序：

```python
class Human:
    num_of_head = 1
    def eat():
        return 'eat something!'
```

类的属性可以是变量，也可以是函数。具体使用如下：

```python
Tom = Human()
print(Tom.num_of_head)
print(Tom.eat())
```

在创建实例的时候，我们也可以完成一些初始化的操作，这时候类要如此定义：

```python
class Human:
    def __init__(self, gender):
        self.gender = gender
```

`__init__`是 Python 中类特有的初始化方法，它的第一个参数固定是`self`，代表创建的实例本身，不需要传入。在类的定义中加入初始化操作后，我们可以这样创建一个实例：`Tom = Human('man')`。与`__init__`相似的，还有一个`__del__`方法，其相当于 C++ 里面的析构函数，在实例回收的时候执行。

Python 类还有一些内置的属性，具体如下：

- `__dict__`：类的属性（包含一个字典，由类的数据属性组成）
- `__doc__`：类的文档字符串
- `__name__`：类名
- `__module__`：类定义所在的模块
- `__bases__`：类的所有父类构成元素（包含了一个由所有父类组成的元组）

Python 中定义的属性都默认为 Public 状态，可以被外界访问和修改。如果要改成内部属性，则应该在属性前面加上两个下划线，如`gender`改为`__gender`，这样子就变成了私有变量，只有内部可以访问。一个有趣的事情是：不能在外部访问`__gender`的原因是因为 Python 解释器对外把`__gender`变量改成了`_Human__gender`，所以实际上外部是可以通过`_Human__gender`来访问`__gender`的。但我们一般不建议这么做，因为不同版本的 Python 解释器可能会把`__gender`改成不同的变量名。

面向对象的三大特性是封装、继承和多态。刚讲完封装，接下来就让我们看看 Python 类中的继承。

我们可以写一个类 Man 继承类 Human，具体实现如下：

```python
class Man(Human):
    pass
```

继承父类的子类可以使用或者重写父类的属性和方法，也可以新增属性和方法。

另外，Python 还支持多继承，具体语法为`class Son(Father, Mother)`。
