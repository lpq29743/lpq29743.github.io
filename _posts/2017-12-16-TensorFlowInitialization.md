---
layout: post
title: TensorFlow 参数初始化
categories: ArtificialIntelligence
description: TensorFlow参数初始化
keywords: TensorFlow, 深度学习, 深度学习框架, 神经网络
---

参数初始化方法的选择对于神经网络的训练有着很大程度的影响。初始点的选择能够决定算法的收敛与否以及收敛的速度，也能影响模型的泛化能力。这篇文章，就让我们结合 TensorFlow 提供的初始化方法，来一起学一下神经网络中的参数初始化。

### 常量初始化

常量初始化就是将参数都初始化为常数，在 TensorFlow 下对应的函数为 `tf.constant_initializer()`，可简写为`tf.Constant()`。由这个函数可衍生出两个初始化方法：`tf.zeros_initializer()`和`tf.ones_initializer()`，分别可简写为`tf.Zeros()`和`tf.Ones()`，表示将参数初始化为 0 或 1。

这种初始化方法一般只用于偏置的初始化，常见的偏置初始化方式是将所有偏置项设为 0。对于 ReLU 激活单元，常见方式是将偏置项初始化为较小的常数值，对于 LSTM 网络，常见方式是将偏置项初始化为 1。

常量初始化不适用于权值的初始化。因为如果神经网络中的每个神经元计算得到相同的输出，那么它们在反向传播算法中也是计算相同的梯度并经历完全相同的参数更新，这样子神经元之间就不存在不对称的来源，相当于每一层只有一个神经元，失去了神经网络进行特征扩展和优化的本意了。

### 小随机数初始化

在数据归一化后，我们可以合理地假设一半的权值为正值，一半的权值为负值。在这个理论基础上，我们可以知道把参数都初始化为 0 是最简便的方式。但由上面的分析，我们可以知道，这种方式是不科学的。为了解决这个问题，我们可以把每个参数都初始化为一个独立随机的接近 0 的数。这种思路的实现方式主要有两种：

1. 高斯分布初始化

   我们可以令权值的初始取值满足 0.001 * N(0, 1) 的分布，其中 N(0, 1) 表示的是标准高斯分布。在 TensorFlow 中，我们可以使用`tf.random_normal_initializer()`来生成一个满足标准高斯分布的随机数，这个函数等价于`tf.RandomNormal()`，也可以使用`random_normal(shape, mean = 0.0, stddev = 1.0, dtype = tf.float32, seed = None, name = None)`函数来生成一个满足标准高斯分布的张量。

2. 均匀分布初始化

   我们也可以把 1 中的标准高斯分布改为均匀分布。在 TensorFlow 中，我们可以使用`tf.random_uniform_initializer()`来生成一个满足均匀分布的随机数，这个函数等价于`tf.RandomUniform()`，也可以使用`tf.random_uniform(shape, minval = 0, maxval = None, dtype = tf.float32, seed = None, name = None)`函数来生成一个满足均匀分布的张量。

值得注意的是，并不是用越小的数进行参数初始化效果越好，因为小的数通过网络向后流动是会降低“梯度信号”。

### 校准方差

上面的做法存在一个问题：当输入的数据量不断增大，随机初始化的神经元的输出数据分布中的方差也不断增大。为了解决这个问题，即让输出方差为 1（不太大也不太小），我们可以用原始初始值除以输入数据量平方根的结果来进行初始化，从而使神经元输出分布的方差归一化。

2010 年，Glorot 的论文[《Understanding the difficulty of training deep feedforward neural networks》](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf) 做了类似的分析，并提出了 Xavier 初始化方法。而紧接着，He 等人又在 2015 年发表了文章[《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》](https://arxiv.org/pdf/1502.01852.pdf)。这篇文章针对 ReLu 神经元的特殊初始化，给出了结论：网络中神经元的方差应该是 2.0 / n 。这也就是 msra 初始化。

### 正交初始化

在 RNN 网络中，会有多次重复的矩阵相乘，如果初始化不当，就会导致梯度消失或梯度爆炸。针对这个问题，我们一般采用正交初始化（QR 分解可得一个正交矩阵）来对 RNN 权值进行初始化。