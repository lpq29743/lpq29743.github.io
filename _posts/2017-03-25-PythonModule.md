---
layout: post
title: Python 模块——有逻辑地组织 Python 代码
categories: Python
description: Python模块
keywords: Python, Python模块
---

不知道你是不是经常在python中见到`import re`或者`__init__`这些代码，那么这些代码到底是什么？又有什么用呢？今天我们就通过这篇文章来了解一下。

### 什么是Python模块

Python模块，就是一个Python文件，包含了Python对象定义和Python语句，如下例support.py：

```python
def print_func( par ):
    print("Hello : ", par)
    return
```

### 怎么使用Python模块

模块定义好后，可以用import引入模块，语法如下：

```python
import module1[, module2[,... moduleN]
```

如要引用模块math，就可以在文件最开始的地方用import math引入。调用math模块的函数时，必须这样引用：

```
模块名.函数名
```

当解释器遇到import时，如果模块在当前搜索路径就会被导入。对模块位置的搜索顺序是：

1. 当前目录
2. 如果不在当前目录，搜索在shell 变量PYTHONPATH下的目录
3. 如果找不到，察看默认路径。UNIX下一般为/usr/local/lib/python/

PYTHONPATH由装在一个列表里的许多目录组成。Windows的PYTHONPATH为 `set PYTHONPATH=c:\python27\lib;` ，UNIX的PYTHONPATH为 `set PYTHONPATH=/usr/local/lib/python` 。import语句具体使用如下例test.py：

```python
#!/usr/bin/python
# -- coding: UTF-8 -- 

# 导入模块
import support 
# 现在可以调用模块里包含的函数了
support.print_func("Runoob")
```

以上实例输出结果：

```
Hello : Runoob
```

Python的from语句可以从模块中导入指定部分到当前命名空间中。语法如下：

```
from modname import name1[, name2[, ... nameN]]
```

例如，要导入模块fib的fibonacci函数，使用如下语句可以不把整个fib模块导入，而只导入fibonacci：

```
from fib import fibonacci
```

### 几个关于导入模块的方法

**dir()函数**

dir()函数返回一个排好序的字符串列表，包含模块定义的所有模块，变量和函数。具体使用如下例：

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
# 导入内置math模块
import math
 
content = dir(math)
 
print content;
```

以上实例输出结果：

```
['__doc__', '__file__', '__name__', 'acos', 'asin', 'atan', 
'atan2', 'ceil', 'cos', 'cosh', 'degrees', 'e', 'exp', 
'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'ldexp', 'log',
'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 
'sqrt', 'tan', 'tanh']
```

**globals()和locals()函数**

globals()和locals()用来返回全局和局部命名空间里的名字。如果在函数内部调用locals()，返回所有能在该函数里访问的命名；如果调用globals()，返回所有在该函数里能访问的全局名字。返回类型都是字典。所以能用keys()函数摘取。

**reload()函数**

当模块被导入，模块顶层部分的代码只执行一次。如果想重新执行模块顶层部分代码，可用reload() 函数，语法如下：

```
reload(module_name)
```

### Python中的包

包是分层次的文件目录结构，简单说就是文件夹，但该文件夹必须存在\_\_init\_\_.py文件，用于标识是包，文件内容可为空。具体使用如下例：

#### 步骤一：设置目录结构

```
test.py
package_runoob
|-- __init__.py
|-- runoob1.py
|-- runoob2.py
```

#### 步骤二：编写package_runoob/runoob1.py

```python
#!/usr/bin/python
# -- coding: UTF-8 -- 
def runoob1():   
    print("I'm in runoob1")
```

#### 步骤二：编写package_runoob/runoob2.py

```python
#!/usr/bin/python
# -- coding: UTF-8 -- 
def runoob1():   
    print("I'm in runoob1")
```

#### 步骤三：编写\_\_init\_\_.py

```python
#!/usr/bin/python
# -- coding: UTF-8 -- 
if __name__ == '__main__':
    print '作为主程序运行'
else:
    print 'package_runoob 初始化'
```

#### 步骤四：编写test.py

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
# 导入 Phone 包
from package_runoob.runoob1 import runoob1
from package_runoob.runoob2 import runoob2
 
runoob1()
runoob2()
```

#### 步骤五：查看输出结果

```
package_runoob 初始化
I'm in runoob1
I'm in runoob2
```

### Python内置属性及特殊模块

- \_\_doc\_\_：模块的docstring，如果不知道什么是文档字符串，请点击[这里](http://www.cnblogs.com/jlsme/articles/1394003.html)
- \_\_file\_\_：模块文件在磁盘上的绝对路径
- \_\_name\_\_：模块的名称。独立运行时值是\_\_main\_\_，被import时值是模块的名称
- \_\_init\_\_：用于初始化的特殊模块
- \_\_del\_\_：对象销毁时所调用的模块
- \_\_main\_\_：常用\_\_name\_\_ == '\_\_main\_\_'语句来判断是否单独运行