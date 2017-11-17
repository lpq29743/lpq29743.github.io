---
layout:     post
title:      "Python工具库之Matplotlib"
subtitle:   "Python下的绘图工具"
date:       2017-04-10 19:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 用Matplotlib画一些简单的图


## 前言

Matplotlib是Python一个强大的绘图工具，它可以画各式各样的图，其官方地址在[这里](http://matplotlib.org/index.html)，在此地址下也可以找到[API](http://matplotlib.org/api/)。本文就让我们从简到繁了解一些这个工具库的使用。

---

## 正文

**实例一：画点**

画点需要用到Matplotlib的pyplot，具体实现方式如下：

```python
import matplotlib.pyplot as plt

x=[1,2]
y=[2,4]
plt.scatter(x,y)
# 调整点的颜色为红色，调整点的形状为x
# color参数包含红(red, r)、绿(green, g)、蓝(blue, b)、黑(k)、青色(c)、洋红色(m)、黄色(y)、白色(w)及其衍生色，还可以使用16进制数表示
# plt.scatter(x, y, color='red', marker='x')
# []里的4个参数分别表示X轴起始点，X轴结束点，Y轴起始点，Y轴结束点
# plt.axis([0,10,0,10])
plt.show()
```

**实例二：画线**

画点是不是太简单了呢？接下来我们再学学画线，具体代码如下：

```python
import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8]
y=[2,1,3,5,2,6,12,7]
plt.plot(x,y)
# 调整线的类型为虚线，加上图标签
# linestyle参数主要包含虚线、点化虚线、粗虚线、实线
# plt.plot(x, y, linestyle='--', label='picture')
plt.show()
```

**实例三：划分绘画框**

在实际工作中，我们常常需要将两张或多张绘图展现在同一个绘画框里，那么如何实现绘画框的划分呢？具体如下：

```python
import matplotlib.pyplot as plt

fig=plt.figure()

# 211表示将绘画框划分为2行1列，最后的1表示第一幅图
p1=fig.add_subplot(211)
x=[1,2,3,4,5,6,7,8]
y=[2,1,3,5,2,6,12,7]
p1.plot(x,y)

# 212表示将绘画框划分为2行1列，最后的2表示第二幅图
p2=fig.add_subplot(212)
a=[1,2]
b=[2,4]
p2.scatter(a,b)

plt.show()
```

**实例三：画3D图**

以上的图都是2D图，现在让我们开始来尝试一下3D图：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# 将绘画框划分为1个子图，并指定为3D图
ax = fig.add_subplot(111, projection='3d')

X = [1, 1, 2, 2]
Y = [3, 4, 4, 3]
Z = [1, 100, 1, 1]

ax.plot_trisurf(X, Y, Z)
plt.show()
```

**实例四：画直方图**

实际上，Matplotlib经常是与numpy等库结合使用的，接下来我们就用一个直方图的例子来展示一下：

```python
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

x_size = [1, 3, 5, 7, 9, 11, 13]
y_time_point = [0, 200, 415, 628, 714, 862, 0]
x_zhe = [3, 5, 7, 9, 11]
y_zhe = [200, 415, 628, 714, 862]

plt.xticks(x_zhe, (u"3", u"5", u"7", u"9", u"11"))
plt.bar(left = (x_size), height = (y_time_point), width = 1.0,
	align="center", facecolor = 'lightskyblue',edgecolor = 'white')

for x, y in zip(x_zhe, y_zhe):
    plt.text(x, y+10, '%.0f' % y, ha='center', va= 'bottom')

ylim(0, 1000)

plt.plot(x_zhe, y_zhe, 'y.-')
plt.title('Time values change with Size')
plt.xlabel('Size')
plt.ylabel('Time (s)')
plt.grid(True)
plt.show()
```

## 后记

实际上，Python下还有很多用于绘图的工具库，其中大部分都是根据Matplotlib进行封装，感兴趣的朋友可以自行去了解一下。
