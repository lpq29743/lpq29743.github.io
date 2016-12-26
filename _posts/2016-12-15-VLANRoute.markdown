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
- 三层交换机
- 在每个VLAN中都设有独立接口的所有外部路由器或路由器组

本文只重点讲解单臂路由器配置，对于每个VLAN，路由器都必须有独立的逻辑连接；为实现这种连接性，可在路由器和交换机之间的物理连接上启用ISL或802.1Q。具体命令如下：

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

## 后记

除了单臂路由器，还可以使用三层交换机的路由端口和SVI接口实现VLAN间的路由，读者可以尝试实现。