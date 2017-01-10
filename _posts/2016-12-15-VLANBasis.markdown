---
layout:     post
title:      "计算机网络实验之VLAN基本配置"
subtitle:   "VLAN配置及VTP配置"
date:       2016-12-15 14:40:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 计算机网络
---

> VLAN配置命令集


## 前言

VLAN是计算机网络中很重要的一个概念，本文我们就来讲讲CLI命令配置VLAN。

---

## 正文

**VLAN配置**

在Cisco里面，所有的端口都默认属于VLAN1，同一个VLAN可以跨越多个交换机。主干功能支持多个VLAN的数据，只有快速以太网端口可以配置为主干端口。具体的配置如下：

```shell
#1.创建VLAN
#全局配置模式下，输入VLAN ID，进入VLAN配置模式：
Switch(config)# vlan {vlan-id}
#注意：vlan-id取值范围为0001~4096；Catalyst交换机默认vlan1为管理vlan，vlan1002-1005保留用于FDDI、令牌环网，不能被修改或删除。 
#为VLAN设置名字（可选）:
Switch(config-vlan)# name {vlan-name}

#2.把交换机端口分配到特定的VLAN
#进入接口配置模式：
Switch(config)# interface {interface}
#定义VLAN端口的成员关系，把它定义为层2接入端口：
Switch(config-if)# switchport mode access
#把端口分配进特定的VLAN里：
Switch(config-if)# switchport access vlan {vlan-id}
#注意：假如把端口分配进了不存在的VLAN里，那么新的VLAN将自动被创建。

#3.设置Trunk端口
#配置中继端口，定义封装方式：
Switch(config-if)# switchport trunk encapsulation {isl|dot1q|negotiate}
#定义端口为层2的中继端口:
Switch(config-if)# switchport mode {dynamic auto|dynamic desirable|trunk}
#dynamic desirable: 主动与对方协商成为Trunk接口的可能性，如果邻居接口模式为trunk/desirable/auto之一，则接口将变成trunk接口工作。如果不能形成trunk模式，则工作在access模式。
#dynamic auto:被动模式，只有邻居交换机主动与自己协商时才会变成Trunk接口。当邻居接口为trunk/desirable之一时，才会成为Trunk。如果不能形成trunk模式，则工作在access模式。
#trunk: 强制接口成为Trunk接口，并且主动诱使对方成为Trunk模式，所以当邻居交换机接口为trunk/desirable/auto时会成为Trunk接口。 
```

**VTP配置**

VTP是一个能够宣告VLAN配置信息的信息系统，通过一个共有的管理域，维持VLAN配置信息一致性。具体配置如下：

```shell
#全局配置模式下，定义VTP模式：
Switch(config)# vtp mode {server|client|transparent}
#定义VTP域名，在同一VLAN管理域的交换机的VTP域名必须相同。该域名长度为1到32字符:
Switch(config)# vtp domain {domain-name}
#设置VTP域的密码，同一VTP域里的交换机的VTP域的密码必须一致，密码长度为8到64字符（可选）：
Switch(config)# vtp password {password}
```

附：

A.在计算机网络中，VLAN和不同网段这两个概念经常会被混淆，那么这两个概念到底有什么区别呢？

- 网段是对IP地址的划分，vlan是对广播域的划分。
- 不同的vlan也可是同一网段，不同的网段也可以同一vlan。

B.对VLAN设置IP地址是必须的吗？为什么要设置IP地址？

VLAN主要工作在OSI参考模型的数据链路层和网络层。如果只工作在第二层，则不需要配置IP，此时VLAN的目的主要是为了抑制广播风暴。如果配置了IP，主要是为了VLAN内部的通信，便于管理。

## 后记

对于计算机网络实验这部分的内容，大部分的文章是以给出命令为主，对于具体的实验请读者利用Packet Tracer进行操作。