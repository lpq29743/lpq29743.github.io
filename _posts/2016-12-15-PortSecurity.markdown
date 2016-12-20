---
layout:     post
title:      "计算机网络实验之端口安全配置"
subtitle:   "MAC地址表管理及端口安全配置"
date:       2016-12-15 14:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - 计算机网络
---

> 数据链路层的端口安全问题


## 前言

上一篇简单地入门了CLI命令，从这一篇我们就要开始做一些实验了，而首先做的就是数据链路层的MAC地址表管理实验和端口安全配置实验。

---

## 正文

**管理MAC地址表**

```shell
#动态MAC地址：由交换机获悉的源MAC地址
#静态MAC地址：由管理员添加到MAC地址表中，永远不会过期。
switch#show mac-address-table   //显示MAC地址表
switch(config)#mac-address-table static mac-address  vlan number  interface type slot/port  //指定静态MAC地址
#举例： 
switch(config)#mac-address-table static 00d0.d3d8.0159 vlan 1 interface f0/1 
```

**端口安全性**

```shell
#将端口配置成安全端口，只允许特定设备与之相连。
#配置端口安全性
switch(config-if)#switchport mode access  //将端口配置为接入端口（连接用户设备的端口）
switch(config-if)#switchport port-security  //启用端口安全性
#注意：不能将端口安全性应用于中继端口。
switch(config-if)#switchport port-security maximum value    //规定最多有多少个地址可以连接到当前接口（value的取值为1~132）
switch(config-if)#switchport port-security mac-address mac-address    //将MAC地址加入到安全端口地址列表中,后面的mac-address替换成相应的MAC地址
switch(config-if)#switchport port-security violation {protect|restrict|shutdown}   //地址违规时，对端口采取的措施（保护、限制、关闭）
地址违规时，对端口采取的措施：
#Shutdown（关闭）：把端口状态置为err-disable，要重新使用这个端口，必须手工激活或者禁止端口安全特性。
#Restrict（限制）：端口仍然处于活跃状态，但对于违规的数据将会被丢弃，并作统计信息，给系统的日志信息发送警报。（如果有黑客不停地伪造mac地址进行攻击，这时会不停地发送报警信息，对设备的性能有很大影响，所以不建议用这种模式。）
#Portect（保护）：端口仍然处于活跃状态，对于违规的数据将会被丢弃，但不发送警报。 
```

## 后记

由于本节比较简单，所以主要是给出命令，具体的还是需要实验，才有不错的效果。