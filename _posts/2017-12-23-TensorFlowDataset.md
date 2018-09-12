---
layout: post
title: TensorFlow Dataset API
categories: ArtificialIntelligence
description: TensorFlow Dataset API
keywords: TensorFlow, 深度学习, 深度学习框架, 神经网络
---

Dataset API 是在 TensorFlow 1.3 版本中引入，使用的方法为调用`tf.contrib.data.Dataset`，但在 TensorFlow 1.4 中，改成了`tf.data.Dataset`。本文以 TensorFlow 1.4 版本为例进行讲解。

Dataset 可看作是相同类型元素的有序列表，这里的元素可以是向量、字符串、图片、tuple 或 dict 等等。

先看最简单的例子，即当元素是数字的时候。通过语句`dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))`，我们就可以创建一个包含五个数字的 dataset。创建完 dataset 后，下一步毫无疑问就是使用了。在这里我们读取元素的方式是从 dataset 中实例化一个 Iterator，然后通过 Iterator 进行迭代，具体如下：

```python
# 实例化了一个 “one shot iterator”，即只能从头到尾读取一次
iterator = dataset.make_one_shot_iterator()
# 从 iterator 里取出一个元素。非 Eager 模式下，one_element 是一个 Tensor，并不是一个值
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        # 取出具体的值
        print(sess.run(one_element))
```

这里展示的是非 Eager 模式下的迭代方法，关于 Eager 模式的了解学习会留到以后的文章。在 Eager 模式中，传统的读取数据的方式都无法使用，必须使用 Dataset API。

如果 dataset 中元素已读取完，再运行`sess.run(one_element)`，会抛出`tf.errors.OutOfRangeError`异常。我们可以在外界捕捉这个异常以判断数据是否读取完，具体如下：

```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

实际上，tf.data.Dataset.from_tensor_slices 的真正作用是切分传入 Tensor 的第一个维度，生成相应的 dataset。例如在语句`dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))`中，传入的是一个形状为 (5, 2) 的矩阵，tf.data.Dataset.from_tensor_slices 会切分它形状上的第一个维度，最后生成的 dataset 中含有 5 个元素，每个元素的形状是 (2, )，即矩阵的一行。

Dataset 中的元素还可以更复杂，如 tuple 或 dict。如在图像识别中，元素可以是 {"image": image_tensor, "label": label_tensor} 的形式，这样处理起来更方便。我们来看下面这个示例：

```python
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }
)
```

示例中，函数会分别切分 a 中的数值和 b 中的数值，最终 dataset 中形式类似 {"a": 1.0, "b": [0.9, 0.1]}。

Dataset 还支持 Transformation 操作。一个 Dataset 可以通过 Transformation 操作变成一个新的Dataset。通常我们通过 Transformation 完成数据变换、打乱、组成 batch 以及生成 epoch 等。常用的 Transformation 有：

- map

  map 接收一个函数，Dataset 中的元素都会被当作函数输入，并将函数返回值作为新的 Dataset，具体如下：

  ```python
  # 将原 dataset 的元素值加 1，得到新的 dataset
  dataset = dataset.map(lambda x: x + 1)
  ```

- batch

  batch 是将多个元素组合成 batch，具体如下：

  ```python
  # 将原 dataset 中的元素组成大小为 32 的 batch
  dataset = dataset.batch(32)
  ```

- shuffle

  shuffle 的功能是打乱 dataset 中的元素，其参数 buffersize 表示打乱时使用的 buffer 大小，具体如下：

  ```python
  dataset = dataset.shuffle(buffer_size=10000)
  ```

- repeat

  repeat 的功能是将整个序列重复多次，主要用来处理 epoch，假设原先数据是一个 epoch，使用 repeat(5) 就可以将之变成 5 个 epoch，具体如下：

  ```python
  dataset = dataset.repeat(5)
  ```

  如果只是单纯调用 repeat() 而不带参数的话，生成序列会无限重复下去，也不会抛出 tf.errors.OutOfRangeError 异常：

除了 tf.data.Dataset.from_tensor_slices，Dataset API 还提供了三种创建 Dataset 的方法：

- tf.data.TextLineDataset()：输入为文件列表，输出为 dataset。dataset 的每个元素对应了文件中的一行，可以用此函数来读入 CSV 文件
- tf.data.FixedLengthRecordDataset()：输入是文件列表和 record_bytes，输出是 dataset。dataset 的每个元素对应文件中固定字节数 record_bytes 的内容，可以用此函数来读取以二进制形式保存的文件，如 CIFAR10 数据集
- tf.data.TFRecordDataset()：用来读取 TFRecord 文件。dataset 的每个元素为一个 TFExample

Iterator 的创建也有更丰富的方法，主要有以下三种：

- initializable iterator
- reinitializable iterator
- feedable iterator

这里我们主要来了解一下 initializable iterator。initializable iterator 使用前必须通过`sess.run()`初始化。使用 initializable iterator，可以将 placeholder 代入 Iterator，从而快速定义新的 Iterator。具体示例如下：

```python
limit = tf.placeholder(dtype=tf.int32, shape=[])
dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
	sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value
```

initializable iterator 还可以用来读大数组。使用tf.data.Dataset.from_tensor_slices(array) 时，实际上是将 array 作为一个常量保存到计算图中。当 array 很大时，会导致计算图变得很大。这时我们可以用 placeholder 代替 array，并使用 initializable iterator 将 array 传进去，具体如下：

```python
# 从硬盘中读入两个 Numpy 数组
with np.load("/var/data/training_data.npy") as data:
	features = data["features"]
    labels = data["labels"]
features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
```

关于 Dataset API，我们就讲到这里，这篇文章也会随着我学习的深入进行更新。