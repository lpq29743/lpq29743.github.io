---
layout:     post
title:      "Android反编译总结"
subtitle:   "走上探究app源码的道路"
date:       2016-10-06 22:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Docker
    - 云计算
---

> App：如果你一层一层地剥开我的心~~


## 前言

在进行Android开发的过程中，我们会接触到很多优秀app。作为一个Android开发者，我们或多或少会好奇其中某个功能的实现。那么在仅仅拥有apk文件的情况下，我们应该怎样操作才能获取到其中的源码呢？

---

## 正文

“工欲善其事，必先利其器”。这句千古名言在Android反编译中同样有效，那么Android反编译需要用到哪些工具呢？具体如下：

- apktool：用于资源文件获取。（[下载地址一](https://bitbucket.org/iBotPeaches/apktool/downloads)、[下载地址二](http://download.csdn.net/detail/vipzjyno1/7025111)）
- dex2jar：用于将apk反编译成java源码，即classes.dex转化成jar文件。（[下载地址一](http://sourceforge.net/projects/dex2jar/files/)、[下载地址二](http://download.csdn.net/detail/vipzjyno1/7025127)）
- jd-gui：查看APK中classes.dex转化成出的jar文件，即源码文件。（[下载地址一](http://jd.benow.ca/)、[下载地址二](http://download.csdn.net/detail/vipzjyno1/7025145)）

那么有了工具之后我们改怎么使用呢？

#### apktool的使用

如果只是为了一些图片资源，直接把apk修改为zip，然后解压。我们可以用android的调试工具monitor获取view的id，然后在解压后的zip中全局搜索id即可。

直接解压apk也能得到图片、xml等资源文件，但是得到的xml文件是乱码，而图片资源也不是原封不动的，特别是.9图片，全都变成了一般的图片。这个情况下往往就需要apktool出手：

***反编译出apk（获取资源文件）***

```shell
# -f 如果目标文件夹已存在，则强制删除现有文件夹（默认如果目标文件夹已存在，则解码失败）
# -o 指定解码目标文件夹的名称（默认使用APK文件的名字来命名目标文件夹）
# -s 不反编译dex文件，也就是说classes.dex文件会被保留（默认会将dex文件解码成smali文件）
# -r 不反编译资源文件，也就是说resources.arsc文件会被保留（默认会将resources.arsc解码成具体的资源文件）
java -jar apktool_2.0.1.jar d -f test.apk -o test
```

***对反编译出的文件进行逆向形成apk（常用于软件汉化、破解等）***

```shell
# 打包过后的apk文件由于没有签名文件，所以不能安装，要自己对软件重新签名才可以，也就是传说中的盗版
apktool b test
# 通过jarsigner重新签名，jarsigner命令是存放在jdk的bin目录下，需将bin目录配置环境变量才可随处使用
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore 签名文件名 -storepass 签名密码 待签名的APK文件名 签名的别名
# 使用zipalign对apk进行对齐操作，使程序运行更快。该工具存放于<Android SDK>/build-tools/<version>目录下
# 其中4是固定值不能改变，后面指定待对齐的APK文件名和对齐后的APK文件名
zipalign 4 New_Demo.apk New_Demo_aligned.apk
```

#### dex2jar的使用

首先要将apk后缀名改为.rar或.zip，解压得到其中的classes.dex文件（它就是java文件编译再通过dx工具打包而成的）

```shell
# 生成classes_dex2jar.jar
dex2jar.bat classes.dex
```

对于我们而言，jar文件也不是可读的，因此要借助jd-gui这个工具来将jar文件转换成java代码。 

## 后记

看完这篇文章之后，相信大家对Android反编译有了更好的了解了吧，不过我现在要很遗憾的告诉大家，Android反编译在实际开发中其实很少用到。原因主要有两方面：一方面，对于那些源码不想被你看见的开发者，他会通过代码混淆的等方式把源码保护地非常好，通过这些工具你根本无法获取到源码，关于代码混淆等app保护方式，博主之后也会用一篇博客的形式进行讲解；另一方面，对于那些想要分享代码、公开代码的开发者来说，自然有很多渠道可以获取到他们的代码，比如github。

那么，这篇博客是不是就毫无意义了呢？并不是，知识这东西从来没有毫无意义这一说法。通过对反编译的了解，我们可以更好地认识到app安全的重要性，加强app保护意思；与此同时，我们也能对应用的本质有更深的认识。