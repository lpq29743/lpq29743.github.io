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

> 当Python遇到人脸识别


## 前言

Python提供了很多人脸识别的库，今天就让我们一起来学习学习。

---

## 正文

本文的实验环境是在Windows下，关于Windows下Python安装opencv库的具体详情可以查看[这里](https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/)（Windows下用pip安装库时，经常会遇到报错的情况，这种情况可以去Python为Windows提供的[扩展库](http://www.lfd.uci.edu/~gohlke/pythonlibs)下找自己需要的库手动安装），接下来我们用一段简洁的代码开始此篇文章，代码如下：

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

关于图像识别，OpenCV更有名，但在提取面部图像部分，dlib更加精确，而且有识别眼睛，鼻子等功能。关于两者在人脸识别方面的比较可以参考这个视频：[dlib vs OpenCV face detection](http://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DLsK0hzcEyHI)。

Dlib是一个基于C++的强大机器学习库，其官网为[dlib.net](http://dlib.net/)，Github项目地址可以点击[这里](https://github.com/davisking/dlib)。Dlib的官网上提供了一些例子，有基于C++的，也有基于Python的，我参考了[face_detector](http://dlib.net/face_detector.py.html)这个例子写了以下代码：

```python
import cv2
import sys
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()

for f in sys.argv[1:]:
    img = io.imread(f)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
    cv2.imwrite('result-dlib-%s.png' % f.split('.')[0], img);
```

这里与前面不同的是，处理图片是用参数的形式传进来的，其中detector中的第二个参数1表示的是采样图像1次，这允许我们检测更多的面孔；cv2的使用主要是为了保存图片。在官网还有几个基于Python的面部识别例子，感兴趣的朋友可以去尝试一下！

工程上一向崇尚用最短的时间做最有效的事情。有一位大神[Adam Geitgey](https://github.com/ageitgey)就利用dlib写了个Python库[face_recognition](https://github.com/ageitgey/face_recognition)，只要用pip就可以下载到该库了。

face_recognition安装后，会有一个自带的名为face_recognition的命令行工具，它能识别照片或照片文件夹中的所有人脸，使用的步骤具体如下：

1. 提供一个包含照片的文件夹，且已知照片中的人是谁，每个人都有一张照片文件，且文件以该人姓名命名
2. 准备另一个文件夹，装有想要识别人脸的照片
3. 运行`face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/`

face_recognition的Github站点提供了几个Demo，这里我们先尝试一下最简单的识别人脸：

```python
import cv2
import face_recognition

image = face_recognition.load_image_file("test2.webp")
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))
for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 2)
cv2.imwrite('result2.png', image);
```

这个库封装了dlib，所以效果与上面的一样，如果想再进一步实验，可以去项目Github主页上查看Demo。

## 后记

本文只是浅显地讲解了Python下几种人脸识别的方法，更加深入的并没有提及。如果大家需要的话，可以选择相应的方法进行深入。
