##### GPU 历史

GPU 是 NVIDIA 在发布 GeForce 256 时提出的概念。

公司：[NVIDIA（英伟达）](https://zh.wikipedia.org/wiki/%E8%8B%B1%E4%BC%9F%E8%BE%BE#%E7%B9%AA%E5%9C%96%E8%99%95%E7%90%86%E5%99%A8)、AMD、Qualcomm（高通）、Intel（英特尔）

##### CUDA

CUDA 安装

```bash
# 查看 cuda 版本 
cat /usr/local/cuda/version.txt
# 查看 cudnn 版本 
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
nvidia-smi
nvidia-smi -l
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

##### Pytorch GPU