---
layout:     post
title:      "Python历险记第六站"
subtitle:   "利用Python实现人脸识别"
date:       2017-03-29 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 当Python遇到OpenCV和Dlib


## 前言

Python有一个很强大的图片处理库PIL，但我们本文要用到的是OpenCV和Dlib，话不多说，让我们正式开始。

---

## 正文

本文的实验环境是在Windows下，关于Windows下Python安装opencv库的具体详情可以查看[这里](https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/)，接下来我们用一段简洁的代码开始此篇文章，代码如下：

```python
import cv2

face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
sample_image = cv2.imread('test1.webp');
faces = face_patterns.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

for (x, y, w, h) in faces:
    cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('result1.png', sample_image);
```

这里的haarcascade_frontalface_default.xml是一个存储训练人脸识别数据的分类器，此文件是opencv默认有的，如果需要的话，也可以训练自己想要的分类器；接下来这一步是加载图片，注意图片必须清晰，不然人脸部分过于模糊将会导致无法识别出脸部；detectMultiScale函数主要是用来进行多尺度检测，scaleFactor参数是图像的缩放因子，minNeighbors是每个矩形应该保留的邻近个数，可以理解为一个人周边有几个人脸，minSize是检测窗口的大小；for循环是为了给每个人脸加一个框；最后一步是将结果输出。

## 后记


