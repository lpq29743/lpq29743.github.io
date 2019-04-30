---
layout: post
title: 深度学习实践中的 GPU 使用
categories: ArtificialIntelligence
description: 深度学习实践中的GPU使用
keywords: 机器学习, 深度学习, GPU, CUDA, TensorFlow, Pytorch
---

GPU 是 NVIDIA 在发布 GeForce 256 时提出的概念。主要的公司包括[NVIDIA（英伟达）](https://zh.wikipedia.org/wiki/%E8%8B%B1%E4%BC%9F%E8%BE%BE#%E7%B9%AA%E5%9C%96%E8%99%95%E7%90%86%E5%99%A8)、AMD、Qualcomm（高通）和 Intel（英特尔）。今天这篇文章主要讨论在深度学习使用中如何使用 GPU。

##### CUDA

CUDA 安装

```bash
# 查看 cuda 版本 
cat /usr/local/cuda/version.txt
# 查看 cudnn 版本 
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# 查看 GPU 使用情况
nvidia-smi
# 一直刷新 GPU 使用情况
nvidia-smi -l
# 定时查看 GPU 使用情况
watch -n 0.1 nvidia-smi
```

##### TensorFlow GPU

在了解 TensorFlow 下 GPU 的使用之前，我们先来看一下`tf.ConfigProto`。`tf.ConfigProto`一般用在创建 Session 时，对 Session 进行参数配置，具体使用如下：

```python
with tf.Session(config=tf.ConfigProto())
```

以下为`tf.ConfigProto`几个重要参数：

```python
# allow_soft_placement 表示如果指定设备不存在，是否允许 TF 自动分配设备
# log_device_placement 表示是否打印设备分配日志
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# 让 TF 在运行过程中动态申请显存，需要多少就申请多少
config.gpu_options.allow_growth = True
# TF 会默认占满内存，这里保证只占用 40% 显存
config.gpu_options.per_process_gpu_memory_fraction = 0.4
```

控制使用哪块 GPU 的方法如下：

```python
# 运行时设置
~/ CUDA_VISIBLE_DEVICES=0  python your.py # 使用 GPU 0
~/ CUDA_VISIBLE_DEVICES=0,1 python your.py # 使用 GPU 0, 1

# 程序中设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
```

如果要在特定的设备执行操作，可以这样做：

```python
with tf.device('/cpu:0'):
```

具体使用的设备标识如下：

- `"/cpu:0"`: 使用 CPU
- `"/device:GPU:0"`: 使用第 0 块 GPU
- `"/device:GPU:1"`: 使用第 1 块 GPU

最后附上两个多 GPU 训练的例子：

1. 简单的例子：https://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/notebooks/4_multi_gpu/multigpu_basics.ipynb
2. 复杂的例子：https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

##### Pytorch GPU

在 Pytorch 中，GPU 是按需分配的。使用多 GPU 训练，只要用`nn.DataParallel`包装模型，并提高 batch size 就可以了。具体使用可以参考[官方文档 1](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)和[官方文档 2](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)。