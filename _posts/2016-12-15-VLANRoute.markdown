---
layout:     post
title:      "计算机网络实验之VLAN间路由"
subtitle:   "单臂路由器配置"
date:       2016-12-15 15:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 计算机网络
---

> VLAN间路由实现之旅


## 前言

VLAN间路由的实现也是计算机网络中很重要的一部分，本文我们就围绕着它进行展开。

---

## 正文

实现VLAN间路由有以下三种方式：

- 单臂路由器
- 三层交换机的路由端口
- 三层交换机的SVI端口

**单臂路由器配置**

```shell
#单臂路由器的ISL配置
Router(config)# interface f0/0.1
Router(config-if)# encapsulation ISL 1
Router(config-if)# ip address 10.1.1.1 255.255.255.0
Router(config-if)# interface f0/0.2
Router(config-if)# encapsulation ISL 2
Router(config-if)# ip address 10.2.2.1 255.255.255.0

#单臂路由器的802.1Q配置
Router(config)# interface f0/0
Router(config-if)# ip address 10.1.1.1 255.255.255.0
Router(config-if)# interface f0/0.2
Router(config-if)# encapsulation dot1Q 2
Router(config-if)# ip address 10.2.2.1 255.255.255.0
#或：
Router(config)# interface f0/0.1
Router(config-if)# encapsulation dot1Q  1 native
Router(config-if)# ip address 10.1.1.1 255.255.255.0
Router(config-if)# interface f0/0.2
Router(config-if)# encapsulation dot1Q 2 
Router(config-if)# ip address 10.2.2.1 255.255.255.0
```

**三层交换机的路由端口配置**

```shell
swith(config-if)# int range f0/1 - 3
swith(config-if-range)# no switchport
swith(config-if-range)# no shut
swith(config-if-range)# inter f0/1
swith(config-if)# ip address 192.168.1.254 255.255.255.0
swith(config-if)# int f0/2
swith(config-if)# ip address 192.168.2.254 255.255.255.0
swith(config-if)# int f0/3
swith(config-if)# ip address 192.168.3.254 255.255.255.0
```

**三层交换机的SVI端口配置**

```shell
swith(config)# ip routing   //启用IP路由
swith(config)# inter vlan 10
swith(config-if)# ip address 192.168.1.254 255.255.255.0
swith(config-if)# int vlan 20
swith(config-if)# ip address 192.168.2.254 255.255.255.0
swith(config-if)# int vlan 30
swith(config-if)# ip address 192.168.3.254 255.255.255.0
```

## 后记

在三种方法中，三层交换机的SVI端口是使用最多的，所以必须很好的掌握。