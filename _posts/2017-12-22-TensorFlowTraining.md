---
layout: post
title: TensorFlow 训练技巧
categories: DeepLearning
description: TensorFlow训练技巧
keywords: TensorFlow, 深度学习, 深度学习框架, 神经网络
---

### Learning rate

神经网络在利用梯度下降算法进行优化的时候，需要定义一个系数 η 来表示权重更新的速度，这个系数就是 learning rate。learning rate 的设置十分重要，设置过大会使结果超过最优值，太小则会使收敛过慢。通常在训练刚开始的时候，我们会使用较大的 learning rate， 随着训练的进行，再慢慢减小 learning rate。我们把这种训练策略叫做 weight decay。TensorFlow 提供了以下两种衰减策略：

- 指数衰减

  ```python
  # learning_rate：初始值
  # global_step：全局 step 数（每个 step 对应一次 batch）
  # decay_steps：learning rate 更新的 step 周期，即每隔多少 step 更新一次 learning rate 的值
  # decay_rate：指数衰减参数（对应 α ^ t 中的 α）
  # staircase：是否阶梯性更新 learning rate，也就是 global_step / decay_steps 的结果是 float 型还是向下取整
  # 计算公式：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
  tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
  ```

- 多项式衰减

  ```python
  # learning_rate、global_step、decay_steps 意义与以上一致
  # end_learning_rate：衰减最终值
  # power：多项式衰减系数（对应 (1 - t) ^ α 的 α）
  # cycle：step 超出 decay_steps 之后是否继续循环 t
  # 计算公式（cycle = False）：global_step = min(global_step, decay_steps)；decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
  # 计算公式（cycle = True）：decay_steps = decay_steps * ceil(global_step / decay_steps)；decayed_learning_rate =(learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
  tf.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, name=None)
  ```

如何确定 learning rate 的初始值也是一个关键的问题。在实践中，我们可以先把 learning rate 设置为 0.01，然后观察 training cost 的走向。如果 cost 在减小，则可以逐步地调大学习速率，试试 0.1，1.0等等；如果 cost 在增大，则得减小学习速率，试试 0.001，0.0001等等。经过一番尝试后，我们就可以大概确定 learning rate 的合适值。

### Momentum

梯度下降的迭代公式为 $$ 𝑤 ← 𝑤 − \eta \frac{𝜕𝐿}{𝜕w} $$，但是我们可以对这个公式进行进一步优化，即采用动量策略。采用动量策略之后的梯度下降的公式变为 $$ 𝑤 ← \gamma 𝑤 − \eta \frac{𝜕𝐿}{𝜕w} $$，其中的系数通常为 0.9。这种策略可以加速学习过程。

### Other optimizers

上面已经提及了一些关于梯度下降的优化。实际上近几年，也出现了很多灵活的优化算法，包括 AdaGrad、AdaDelta、RMSProp 以及 Adam 等等。

AdaGrad（adaptive gradient）允许 learning rate 基于参数进行调整，而无需人为调整。具体就是根据不常用的参数进行较大幅度的 learning rate 更新，根据常用的参数进行较小幅度的 learning rate 更新。TensorFlow 使用 AdaGrad 优化器的具体代码如下：

```python
tf.train.AdagradOptimizer(
    learning_rate,
    initial_accumulator_value=0.1,
    use_locking=False,
    name='Adagrad'
)
```

AdaDelta 使用最近历史梯度值缩放 learning rate，并且和经典的动量算法相似，累积历史的更新以加速学习。它有效地克服了 AdaGrad 中 learning rate 收敛至零的缺点。TensorFlow 使用 AdaDelta 优化器的具体代码如下：

```python
tf.train.AdadeltaOptimizer(
    learning_rate=0.001,
    rho=0.95,
    epsilon=1e-08,
    use_locking=False,
    name='Adadelta'
)
```

RMSProp 则是在 AdaGrad 的基础上加入了 decay factor，防止历史梯度求和过大。TensorFlow 使用 RMSProp 优化器的具体代码如下：

```python
tf.train.RMSPropOptimizer(
    learning_rate,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
    use_locking=False,
    centered=False,
    name='RMSProp'
)
```

Adam 算法综合了 AdaGrad 和 RMSProp 的优点。其收敛非常快，是目前所知的最优算法。TensorFlow 使用 Adam 优化器的具体代码如下：

```python
tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
```

### Epoch、 iteration and batch size

很多初学者（包括我）遇到这三个名词经常会搞混，这三个名词的意义如下：

- batch size：每次训练在训练集中取 batch size 个样本训练
- iteration：1 个 iteration 等于使用 batch size 个样本训练一次，迭代的次数等于总样本数除以 batch size 后向上取整
- epoch：1 个 epoch 等于使用训练集中的全部样本训练一次

那么为什么会有这三个名词的出现呢？它们的出现又有什么意义呢？

当数据集比较小的时候，我们就可以直接采用全数据集进行训练（Full Batch Learning），这样子更准确地朝向极值所在的方向。但当数据集变大的时候，Full Batch Learning 就会受到内存的限制。与 Full Batch Learning 完全相反的方法，是每次只训练一个样本，即 batch_size = 1，这种方法也称为在线学习（Online Learning）。在线学习方法虽然不会受到内存的限制，计算速度也很快，但是却存在难以达到收敛的问题。结合这两者的优缺点，便有了 Mini-batches Learning。

那么我们应该如何来调节 batch size 的值呢？根据 [CUDA GPU 中 warp 的架构](http://www.cnblogs.com/1024incn/p/4541313.html)，我们在用 GPU 训练的时候可以将 batch size 设为 32 的整数，一般可以从 128 开始，再进行上下调整。在合理的范围内，我们应该尽可能增大 batch size 的值，这样子可以提高内存利用率，减少一次 epoch 所需的迭代次数，并且使得梯度下降的方向更加准确。但是如果盲目增大此值，要想达到同样的精度，则会花费更多时间，参数的修正显得更加缓慢，下降方向也基本不再变化。

接下来，我们再来看看 epoch。在神经网络的训练中，随着 epoch 数量的增加，权重的更新次数也会相应增加，从而模型会从欠拟合走向过拟合。对于 epoch 的数目，不同的情景会有很大的差别，需要结合模型的具体情况进行尝试。

### Regularization

正则化技术是防止神经网络出现过拟合现象的有效方法。过去数年，研究者提出了多种正则化方法，如数据增强、L2 正则化、L1 正则化、Dropout 等。

过拟合现象在参数数目多于训练样本数的神经网络中普遍出现，所以我们可以通过数据增强的方式来避免过拟合现象的出现。数据增强是指通过向训练数据添加转换或扰动来人工增加训练数据集。在图像处理中，常见的数据增强技术有水平或垂直翻转图像、裁剪、色彩变换、扩展和旋转等。

L2 正则化可以说是最常见的正则化技术，其具体操作是在代价函数后面加上一个系数 λ 与 L2 正则化项的乘积，L2 正则化项为权值向量 w 中各个元素的平方和。其中 λ 越大，权重衰减地越快。

L1 正则化与 L2 正则化类似，不过它的正则化项是权值向量 w 中各个元素的绝对值之和，它也可以在一定程度上防止过拟合。

Dropout 是指暂时丢弃一部分神经元及其连接来防止过拟合。每个神经元被丢弃的概率为 p（通常设置为 0.5，如果模型不是很复杂，可以设为 0.2）。Dropout 在显著降低过拟合现象发生概率的同时，也提高了模型的学习速度。

### Loss functions

TensorFlow 提供了多种交叉熵损失函数的实现：

1. ```python
   # 对最后一层输出 logits 进行 softmax 函数分类后，计算其与实际标签 labels 之间的交叉熵
   tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
   ```

2. ```python
   # labels 的每一行是 one-hot 表示，即只有一个分量为 1
   tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
   ```

3. ```python
   # TensorFlow 最早实现的交叉熵算法，通常用于多目标分类问题
   tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
   ```

4. ```python
   # 在上一个版本的基础上，正样本算出的值乘以某个系数
   tf.nn.weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None)
   ```

### Activation functions

激活函数的作用是为神经网络加入一些非线性因素。它必须是可微的，这样才能够使用梯度下降法。常见的激活函数包括 Sigmoid、tanh、ReLU（Rectified Linear Unit）和 Softplus 等等。

Sigmoid 函数的表达式为 $$ f(x) = \frac{1}{1+e^{-x}} $$。它的输出映射在 (0, 1) 内，单调连续，求导比较容易，但具有软饱和性，容易产生梯度消失，而且输出不是以 0 为中心。TensorFlow 实现 Sigmoid 函数的方法是`tf.nn.sigmoid()`。

tanh 函数的表达式为 $$ f(x) = \frac{sinhx}{coshx} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e{-2x}} $$。tanh 函数的输出是以 0 为中心，收敛速度比 Sigmoid 函数要快，但还是没有解决因为软饱和性而产生的梯度消失问题。TensorFlow 实现 tanh 函数的方法是`tf.nn.tanh()`。

ReLU 函数是目前使用最多的也是最受欢迎的激活函数，其表达式为 $$ f(x) = max(x, 0) $$。ReLU 函数在 x < 0 的时候是硬饱和的，而在 x > 0 的时候可以保持梯度不衰减，从而缓解梯度消失问题，并且快速收敛。随着训练的进行，部分输入会落到硬饱和区，导致对应权重无法更新，这种情况称为“神经元死亡”。TensorFlow 实现 tanh 函数的方法是`tf.nn.relu()`。

TensorFlow 还在 ReLU 函数的基础上，定义了`tf.nn.relu6()`，其表达式为 $$ min(max(x, 0), 6) $$。另外还有`tf.nn.crelu()`，关于 CReLU 函数，可以查看[这篇文章](https://arxiv.org/pdf/1603.05201v2.pdf)。

Softplus 函数可以看成是 ReLU 函数的平滑版本，其表达式为 $$ f(x) = log(1 + e^x) $$。TensorFlow 中实现 Softplus 的方法是 `tf.nn.softplus()`。
