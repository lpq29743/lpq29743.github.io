---
layout: post
title: TensorFlow 入门
categories: Tool
description: TensorFlow入门
keywords: TensorFlow, 深度学习, 深度学习框架
---

本文参考斯坦福大学公开课 [CS 20SI: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/index.html) 前半部分的内容做了简单的笔记。如果大家想要进一步学习的话，建议参与一些小项目再学习后半部分的内容。

### 图和会话

数据流图的执行分为两个阶段：

1. 构建一个图
2. 用一个会话执行图中的操作

比如执行一个计算加法的图：

```python
import tensorflow as tf
a = tf.add(3, 5)
sess = tf.Session()
print(sess.run(a))
sess.close()
```

更加 Pythonic 的写法如下：

```python
import tensorflow as tf
a = tf.add(3, 5)
with tf.Session() as sess:
    print(sess.run(a))
```

尝试执行更加复杂的数据流图：

```python
import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z, not_useless = sess.run([pow_op, useless])
    print(z, not_useless)
```

TensorFlow 允许用户建立多个图，但一般不建议这样做。因为一方面多个图需要多个会话，而每个会话都会默认尽最大可能去使用所有可用资源，另一方面，如果要在两个图中传输数据，则必须要使用 numpy 库。因此，我们的建议是在一张图中建立多个互相不连通的子图。

### 基本操作、常量和变量

我们可以通过下面的方式创建常量：

```python
# 语法：创建一个常量
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
# 例子：创建一个向量和一个矩阵
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
```

还可以创建一些特殊的常量：

```python
# 语法：创建一个常量，它的所有元素都为 0
tf.zeros(shape, dtype=tf.float32, name=None)
# 例子：创建一个 2 * 3 的矩阵，它的所有元素都为 0
tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]
# 语法：创建一个与输入张量类型大小一致的常量，它的所有元素都为 0
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# 例子：输入向量 input_tensor 为 [0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]
# 同样的还有以下两个
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
# 语法：创建一个常量，它的所有元素都为指定值
tf.fill(dims, value, name=None)
# 例子：创建一个 2 * 3 的矩阵，它的所有元素都为 8
tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```

也可以创建序列常量：

```python
# 语法：在 [start, stop] 的范围内产生 num 个数的等差数列。注意 start 和 stop 要用浮点数表示
tf.linspace(start, stop, num, name=None)
# 例子：在 [10.0, 13.0] 的范围内产生 4 个数的等差数列
tf.linspace(10.0, 13.0, 4, name="linspace") ==> [10.0 11.0 12.0 13.0]
# 语法：在 [start, limit) 的范围内以步进值 delta 产生等差数列。注意不包括 limit 在内
tf.range(start, limit=None, delta=1, dtype=None, name='range')
# 例子：在 [3, 18) 的范围内以步进值 3 产生等差数列
tf.range(start=3, limit=18, delta=3) ==> [3, 6, 9, 12, 15]
# 例子：在 [0, 5) 的范围内以步进值 1 产生等差数列
tf.range(limit=5) ==> [0, 1, 2, 3, 4]
```

值得注意的是，这里产生的序列是**不能迭代**的。

另外，TensorFlow 还可以定义随机数常量：

```python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

变量的定义方式如下：

```python
# 定义一个标量
a = tf.Variable(2, name="scalar")
# 定义一个向量
b = tf.Variable([2, 3], name="vector")
# 定义一个矩阵
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# 定义一个张量
W = tf.Variable(tf.zeros([784,10]))
```

这里有一个有意思的地方，定义变量的时候 Variable 首字母大写，而常量 constant 首字母小写。这是因为前者是一个类，而后者是一个操作。作为类，那它必然包含一些操作：

```python
x = tf.Variable(...)
x.initializer # 初始化
x.value() # 读取操作
x.assign(...) # 写操作
x.assign_add(...) # 追加操作
```

**在使用变量之前必须对其进行初始化。**最简单的初始化方式便是一次性初始化所有的变量：

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

也可以只对部分变量进行初始化：

```python
init_ab = tf.variable_initializer([a, b], name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
```

或是单独对一个变量进行初始化：

```python
w = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(w.initializer)
```

如果要取出变量的值，有以下两种方法：

```python
w = tf.Variable(tf.truncated_normal([10, 10], name='normal'))
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval()) # 方法一
    print(sess.run(w)) # 方法二
```

下面看看这个程序：

```python
w = tf.Variable(10)
w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval())
```

上面这个程度会得到10，而不是100。这是因为虽然定义了 assign 操作，但是 TensorFlow 是在 session 中执行操作，所以我们需要执行 assign 操作：

```
w = tf.Variable(10)
assign_op = w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(assign_op)
    print(w.eval())
```

之前我们提过数据流图的执行的两个步骤，但是对于图的定义，我们经常会遇到暂时不知道值的情况。对此，我们可以先定义为占位符，之后再用`feed_dict`去赋值。

定义占位符的语法如下：

```python
tf.placeholder(dtype, shape=None, name=None)
```

我们可以用字典的形式进行赋值：

```python
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    # print(sess.run(c, feed_dict={a: [1, 2, 3]})) 也可以
    print(sess.run(c, {a: [1, 2, 3]}))
```

### 一个简单的线性回归例子

例子：探寻火灾与盗窃之间的线性关系

代码：

```python
""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = 'data/fire_theft.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * w + b 

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	
	writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
	
	# Step 8: train the model
	for i in range(50): # train the model 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs train_op and fetch values of loss
			_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
			total_loss += l
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	# close the writer when you're done using it
	writer.close() 
	
	# Step 9: output the values of w and b
	w, b = sess.run([w, b]) 

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
```

### 结构化你的 TensorFlow 模型

之前我们讲过模型实现的两个主要步骤：构建计算图和执行计算图。

这里我们把构建计算图的过程进行细化：

1. 定义输入和输出的占位符
2. 定义模型中需要用到的权重
3. 定义推断模型，构建网络
4. 定义损失函数作为优化对象
5. 定义优化器进行优化

再把执行计算图的步骤进行细化：

1. 第一次进行运算的时候，初始化模型的所有参数
2. 传入训练数据，可以打乱顺序
3. 网络前向传播，计算出当前参数下的网络输出
4. 根据网络输出和目标计算出 loss
5. 通过 loss 方向传播更新网络中的参数
