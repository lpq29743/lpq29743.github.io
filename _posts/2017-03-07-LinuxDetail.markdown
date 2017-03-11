---
layout:     post
title:      "Linux的细枝末节"
subtitle:   "谈谈使用Linux过程中遇到的小知识点"
date:       2017-03-07 20:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Linux
---

> 不定期更新的Linux笔记~~


## 前言

从大一第二学期开始简单地学习一些Linux命令，到大二第二学期把自己的电脑装成Ubuntu单系统以强迫自己学习Linux，再到后面装成了Win7+CentOS的双系统。与Linxu相识已有几年了，认识它越多，就发现自己不认识它的地方越多。希望能通过这篇文章，能够让我对它越来越了解！

---

## 正文

**查看系统版本**

```shell
# 这种方法适用于所有的Linux
lsb_release -a
# 针对Redhat、CentOS等系统
cat /etc/redhat-release
```

**Linux添加中文字体**

Linux默认支持的中文字体不多，需要我们做一定的配置，具体配置可以参考[这篇文章](http://5iqiong.blog.51cto.com/2999926/1188961)，而Linux的字体目录为/usr/share/fonts/。使用命令`fc-list :lang=zh`可以查看当前系统中有哪些中文字体。

**怎么给Linux下Eclipse、Pycharm等软件添加桌面快捷方式**

要给这类软件加快捷方式，首先要创建相应文件：`vim /usr/share/applications/eclipse.desktop`，然后在文件中输入以下文本：

```shell
[Desktop Entry]
Encoding=UTF-8
Name=Eclipse
Comment=Eclipse IDE
# 软件启动位置
Exec=/usr/local/Android/eclipse/eclipse
# 软件显示图标
Icon=/usr/local/android/eclipse/icon.xpm
Terminal=false
StartupNotify=true
Type=Application
Categories=Application;Development;
```

然后在`/usr/share/applications/`目录下就可以找到对应快捷方式了，右键copy到桌面即可。

**更改CentOS yum源**

更改CentOS yum源可以使yum的安装更新速度更快，具体细节可以参考[这篇文章](http://www.jianshu.com/p/d8573f9d1f96)。

## 后记

这篇文章会不定期地更新我使用Linux过程中遇到的问题，也欢迎大家提出自己遇到的问题。


