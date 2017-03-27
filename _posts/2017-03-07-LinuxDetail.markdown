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

**Linux主要有哪几类系统**

CentOS、Ubuntu、Debian、OpenSUSE………，这些名词是不是让你一头雾水，那么Linux到底有几类系统呢？实际上，主要的Linux系统包括[Debian](https://zh.wikipedia.org/wiki/Debian)（及其派生版本Ubuntu、Linux Mint）、[Fedora](https://zh.wikipedia.org/wiki/Fedora)（及其相关版本Red Hat Enterprise Linux、CentOS）和[openSUSE](https://zh.wikipedia.org/wiki/OpenSUSE)等。这些类型的系统的在软件包管理等方面都存在较大差异，其中Debian系和Fedora系相对较多。

**查看系统版本**

```shell
# 这种方法适用于所有的Linux
lsb_release -a
# 针对Redhat、CentOS等系统
cat /etc/redhat-release
```

**who、who am i以及whoami有什么区别**

实践出真知，关于这个问题大家可以自己实验一下，具体的结论如下：

- who：查看当前有哪些用户登录到了本台机器
- who am i：显示的是实际的用户名，即登录的用户ID。此命令相当于who -m
- whoami：显示的是当前操作用户的用户名

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

**Fedora如何启动网卡**

装好虚拟机后，经常会遇到网络无法连接的情况，那么应该怎样解决呢？实际上，原因主要是网卡未启动。启动网卡（假设网卡名为eth0）主要有以下两种方法：

1. 执行命令`ifup eth0`
2. 一劳永逸的做法。修改文件/etc/sysconfig/network-scripts/ifcfg-eth0，将ONBOOT设为yes，让网卡开机启动

**Fedora下：xx is not in the sudoers file问题解决**

1. 用命令`whereis sudoers`找出文件所在位置，默认是/etc/sudoers
2. 以root登录，用`chmod u+w /etc/sudoers`添加写权限
3. 编辑文件，在root ALL=(ALL)ALL行下添加XXX ALL=(ALL)ALL，XXX为用户名，保存退出
4. 用`chmod u－w /etc/sudoers`回到文件原权限

**CentOS下有什么合适的中文输入法**

相对于小伙伴Ubuntu，CentOS支持的中文输入法相对较少，自带的输入法体验又相当糟糕，通过一番资料查询，我终于找到了一种不错的中文输入法———tong，它的具体安装及使用可以参考[这里](http://seisman.info/install-yong-chinese-input-method-under-centos-7.html)

**更改CentOS yum源**

更改CentOS yum源可以使yum的安装更新速度更快，具体细节可以参考[这篇文章](http://www.jianshu.com/p/d8573f9d1f96)。

**yum install报错：Another app is currently holding the yum lock解决方法**

顾名思义，这个报错信息的意思是有另外一个应用在使用yum，被占用锁定了，所以我们可以选择直接结束占用yum的进程来解决问题，也可以通过强制关掉yum进程：`sudo rm -f /var/run/yum.pid`来解决。

**怎么找到yum安装的软件路径**

查找yum安装的软件的具体路径，需要用到rpm命令，具体使用如下：

```shell
# 查询软件包
rpm -qa|grep xxx
# 查询软件包安装路径
rpm -ql xxx
```

rpm命令是RPM软件包的管理工具，遵循GPL规则且功能强大方便，具体的使用可参考[这里](http://man.linuxde.net/rpm)。

## 后记

这篇文章会不定期地更新我使用Linux过程中遇到的问题，也欢迎大家提出自己遇到的问题。


