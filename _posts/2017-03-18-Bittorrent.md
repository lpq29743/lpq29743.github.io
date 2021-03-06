---
layout: post
title: 聊聊 BT 那些事
categories: Blog
description: 聊聊BT那些事
keywords: BT, BitTorrent, 磁力链接
---

这篇文章的灵感，来自于网易云音乐的电台节目《软件那些事儿》里的节目[《BT软件(BitTorrent)的前生今世》](http://music.163.com/#/program?id=788827962)。关于这篇文章的主角——BitTorrent，相信大部分朋友都听说过，那么我们对它真的熟悉吗？

### BitTorrent是什么

BitTorrent简称BT，中文全称比特流。HTTP、FTP等传统方式一般是将文件放到服务器上，然后由服务器传送到用户机器上，如果同一时刻下载用户太多，势必影响到下载速度。而BT克服了传统下载方式的局限性，具有下载的人越多，文件下载速度越快的特点。正因如此，BT成为了下载迷的最爱。

**BitTorrent的工作原理是什么**

根据BitTorrent协议，文件发布者根据发布文件生成.torrent文件，即种子文件。种子文件是文本文件，包含Tracker信息和文件信息。Tracker信息是BT下载中用到的Tracker服务器（收集下载者信息的服务器，并提供给其他下载者，使下载者连接起来传输数据）地址和Tracker服务器设置，文件信息根据目标文件生成。BitTorrent的主要原理是把下载文件虚拟分成相等的块，并把每个块的索引信息和Hash验证码写入种子文件中，所以种子文件就是被下载文件的索引。

下载者要下载文件，需先得到相应的种子文件，然后用BT软件下载。下载时，BT客户端首先解析种子得到Tracker地址，然后连接Tracker服务器。Tracker服务器提供其他下载者（包括发布者）的IP。下载者再连接其他下载者，根据种子文件，互相告知已有块，然后交换对方没有的数据。此时不需服务器参与，分散了线路流量，减轻了服务器负担。

为了解决某些用户下完就跑的现象，在非官方BitTorrent协议中还存在一种慢慢开放下载内容的超级种子的算法。

### 磁力链接是什么

讲磁力链接之前，我们先看一下DHT网络技术。DHT全称分布式哈希表。在不需服务器的情况下，每个客户端负责一个小范围路由，并存储小部分数据，从而实现DHT网络的寻址和存储。使用支持该技术的BT下载软件，用户无需连上Tracker即可下载，因为软件会在DHT网络中寻找下载同一文件的其他用户并与之通讯，开始下载任务。这种技术减轻了Tracker负担（甚至不需要），用户之间可以更快速创建通讯（特别是与Tracker连接不上时）。

至于磁力链接，它是通过不同文件内容的Hash结果生成一个纯文本的数字指纹，用来识别文件。从2009年开始，很多BT服务器被关，不仅很多种子文件无法找到，Tracker服务器也断开解析工作，使得BT下载成为很大难题，而磁力链接很好地解决了这个问题，它利用BT种子中记录的数字指纹通过DHT网络进行搜索，获取下载者列表，与其他下载者通讯。

### BitTorrent有哪些缺点

BT下载并不完美，它具有以下缺点：

- 由于无法验证文件发布者，下载内容的安全性难以得到保障
- 对于ADSL用户来说，持续大量上传数据会影响下载速度和其他网络连接的速度
- BT资源存在热度问题，如果发布者停止发布且上传者变少，则下载速度下降甚至无法下载，直至种子失效（最糟糕的情况是小部分失效）
- 搜索和版权问题。用户通过BT网站下载种子从而下载资源，而版权拥有者逐个起诉下载者是不现实的，他们只能把目光集中在BT站点上

### 有哪些知名的BT下载站点呢

世界范围内比较知名的BT下载站点主要有以下几个：

- 海盗湾：瑞典网站，号称世界最大的BT种子服务器，其官方地址为https://thepiratebay.org。该站点资源多，支持中文，查找方便，但速度较慢
- KickassTorrents：同样闻名于世界的BT站点，官方地址是kat.cr，但随着2016年7月20日，年仅30岁的网站所有者Artem Vaulin在波兰被捕，受到侵犯版权，共谋洗钱两项刑事指控后，该网站已经被封
- Torrentz：官方网站也受到了关停，但已有[另一网址](https://torrentz2.eu/)

除了以上三个站点，还有很多BT下载站点（部分站点会长期或不定期失效），具体可以查看这两个链接： [那些神器级别的BT磁力搜索网站](http://askrain.lofter.com/post/47dc29_2487136) 和 [怎么才能搜索到有效的种子或磁力链接](https://www.zhihu.com/question/29682242) 。

### 有哪些知名的BT下载客户端呢

下载到相应的BT种子文件后，我们需要用BT客户端进行下载，那么有哪些知名的BT客户端呢？具体有以下几个：

- 迅雷：国产下载软件，支持BT下载，相当方便好用
- uTorrent：小巧强劲，用C++编写，支持Windows、Mac OS X和GNU/Linux
- Transmission：界面简洁，是一款自由软件，支持跨平台，常在Linux下使用

### BT下载会损坏硬盘吗

这个问题已经在知乎上进行过讨论，这里直接给出[问题链接](https://www.zhihu.com/question/20129670)。