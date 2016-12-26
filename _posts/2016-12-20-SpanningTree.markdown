---
layout:     post
title:      "计算机网络实验之生成树协议"
subtitle:   "生成树协议配置"
date:       2016-12-20 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 计算机网络
---

> 今天学学生成树协议~~


## 前言

好几天没写到网络实验课的内容了，今天趁着有空，继续总结学习……

---

## 正文

**配置根桥**

```shell
#降低网桥优先级,手动设置根桥.
Switch(config)# spanning-tree vlan vlan-id priority value

#在配置STP时，建议给根桥指定的根优先级值是4096。例：
Switch(config)# spanning-tree vlan 1 priority 4096
```

**STP的启动和验证**

```shell
#启动STP
Switch(config)# spanning-tree vlan vlan-id

#验证STP
Switch# show spanning-tree vlan vlan-id
```

PortFast启用

```shell
#接口下启用PortFast
Switch(config-if)# spanning-tree portfast
```

## 后记

网络实验这部分主要是命令，学习过程中应注重实践。