---
layout: post
title: Python 之数据结构
categories: Python
description: Python之数据结构
keywords: Python, Python数据结构
---

十二月份的第一篇文章，献给我的宠妃Python。学习Python是在写博客之前，那时候没有做好笔记，现在也认识到做笔记的重要性了。今天第一篇就讲讲Python的数据结构。

需要特别说明的是，本博客与Python相关的文章所用的都是Python3版本。

本文主要描述的python数据结构有：

- list
- dict
- tuple
- set

### list

list是一种有序的集合，可以随时添加和删除其中的元素。list里面的元素的数据类型也可以不同。具体操作如下：

```python
>>> classmates = ['Michael', 'Bob', 'Tracy']
>>> classmates
['Michael', 'Bob', 'Tracy']
>>> len(classmates)
3
>>> classmates[0]
'Michael'
>>> classmates[-1]
'Tracy'
>>> classmates.append('Adam')
>>> classmates
['Michael', 'Bob', 'Tracy', 'Adam']
>>> classmates.insert(1, 'Jack')
>>> classmates
['Michael', 'Jack', 'Bob', 'Tracy', 'Adam']
>>> classmates.pop()
'Adam'
>>> classmates
['Michael', 'Jack', 'Bob', 'Tracy']
>>> classmates.pop(1)
'Jack'
>>> classmates
['Michael', 'Bob', 'Tracy']
>>> classmates[1] = 'Sarah'
>>> classmates
['Michael', 'Sarah', 'Tracy']
>>> L = []
>>> len(L)
0
```

### dict

dict（字典）使用键-值（key-value）存储，具有极快的查找速度。具体操作如下：

如果用dict实现，只需要一个“名字”-“成绩”的对照表，直接根据名字查找成绩，无论这个表有多大，查找速度都不会变慢。用Python写一个dict如下：

```python
>>> d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
>>> d['Michael']
95
>>> d['Adam'] = 67
>>> d['Adam']
67
>>> 'Thomas' in d
False
>>> d.get('Thomas')
>>> d.get('Thomas', -1)
-1
>>> d.pop('Bob')
75
>>> d
{'Michael': 95, 'Tracy': 85}
```

### tuple

tuple和list非常类似，但是tuple一旦初始化就不能修改，具体操作如下：

```python
>>> classmates = ('Michael', 'Bob', 'Tracy')
>>> t = ('a', 'b', ['A', 'B'])
>>> t[2][0] = 'X'
>>> t[2][1] = 'Y'
>>> t
('a', 'b', ['X', 'Y'])
```

### set

set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。set是无序的。set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作。具体操作如下：

```python
>>> s = set([1, 2, 3])
>>> s
{1, 2, 3}
>>> s = set([1, 1, 2, 2, 3, 3])
>>> s
{1, 2, 3}
>>> s.add(4)
>>> s
{1, 2, 3, 4}
>>> s.add(4)
>>> s
{1, 2, 3, 4}
>>> s.remove(4)
>>> s
{1, 2, 3}
>>> s1 = set([1, 2, 3])
>>> s2 = set([2, 3, 4])
>>> s1 & s2
{2, 3}
>>> s1 | s2
{1, 2, 3, 4}
```

### 附：可变对象和不可变对象

- 可变对象：list, dictionary, set, byte array
- 不可变对象：int，long, complex, string, float, tuple, frozen set