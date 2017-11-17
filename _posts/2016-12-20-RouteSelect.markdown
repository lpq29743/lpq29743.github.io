---
layout:     post
title:      "计算机网络实验之路由选择"
subtitle:   "静态路由、RIP及OSPF配置"
date:       2016-12-20 20:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 计算机网络
---

> Boss出场


## 前言

路由选择同样是计算机网络中常见的一个技术点。路由器需要根据路由表进行选路，而路由表的建立主要有直连网络、静态路由、动态路由以及默认路由四种，今天我们就把这几个知识点一一讲一下！

---

## 正文

首先先粗略介绍一下前言提到的四种建立路由表的方法：

- 直连网络：将路由器接口直接连接到网段时即可创建此条目。如果接口出现故障或因管理原因关闭，该条目将从路由表中删除。
- 静态网络：由网络管理员在路由器上手工添加路由信息以实现路由目的。静态路由适用于小型网络，静态路由常用于将分组路由到末节网络（stub network，只能通过一条路由才能到达的网络）
- 动态网络：根据网络结构或流量的变化，路由协议会自动调整路由信息以实现路由。
- 默认路由：如果路由表中没有明确到达目标的路径，则会使用默认路由。默认路由可以手动设置，也可由动态路由协议填写。

**静态路由的配置**

```shell
#静态路由的添加
Router(config)# ip route network  mask {address | interface}
#举例
Router(config)# ip route 172.16.1.0 255.255.255.0 172.16.2.1
#静态路由的删除
Router(config)# no ip route 172.16.1.0 255.255.255.0 172.16.2.1
#默认静态路由的配置
Router(config)# ip route 0.0.0.0 0.0.0.0 s0
#举例
Router(config)# ip route 0.0.0.0 0.0.0.0 172.16.2.2
```

而动态路由中常用到的两种配置分别是距离矢量协议RIP以及链路状态协议OSPF，它们的具体配置如下：

**RIP配置**

```shell
#RIP配置
#激活RIP协议
Router(config)# router rip
#选择需要激活的接口所在的主类网络
Router(config-router)# network network-number
#举例
Router(config)# router rip
Router(config-router)# network 172.16.0.0
Router(config-router)# network 10.0.0.0
#查看RIP信息
Router# show ip protocols
#查看路由表
Router# show ip route

#RIPv2配置
#激活RIP协议（默认为版本1）
Router(config)# router rip
#指定RIPv2
Router(config-router)# version 2 	
#哪些接口参与路由（通告主类网络号码）
Router(config-router)# network network-number

#EIGRP配置
#启动EIGRP
Router(config)# router eigrp {as-number}
#宣告直连主类网络号
Router(config-router)# network {network-number} [wildcard-mask]
#EIGRP 配置的验证
#Displays the neighbors discovered by IP EIGRP
Router# show ip eigrp neighbors
#Displays the IP EIGRP topology table
Router# show ip eigrp topology
#Displays current EIGRP entries in the routing table
Router# show ip route eigrp
#Displays the parameters and current state of the active routing protocol process
Router# show ip protocols
#Displays the number of IP EIGRP packets sent and received
Router# show ip eigrp traffic 
#关闭EIGRP路由的自动汇总特性:
Router(config-router)# no auto-summary
#基于接口的进行EIGRP手动的路由汇总,默认EIGRP汇总路由的管理距离为5:
Router(config-if)# ip summary-address eigrp {AS-number} {ip-address} {mask} [distance]
```

**OSPF配置**

```shell
#启用 OSPF
Router(config)# router ospf process-id
#将网段指派到指定的区域中
Router(config-router)# network address wildcard-mask area area-id
#查看启用的路由协议
Router# show ip protocols
#查看路由表
Router# show ip route
#查看特定区域中的接口
Router# show ip ospf interface
#查看在每一个接口上的邻居信息
Router# show ip ospf neighbor
```

## 后记

本文涉及到的命令较多，朋友们要好好消化，特别是静态路由配置和OSPF配置，两者经常要用到。