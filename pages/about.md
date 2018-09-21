---
layout: page
title: About
description: 林佩勤的个人介绍
keywords: Peiqin Lin, 林佩勤
comments: true
menu: 关于
permalink: /about/
---

林佩勤，自然语言处理爱好者，电影摇滚爱好者。

热门 Github 开源项目：[IAN](https://github.com/lpq29743/IAN)，[RAM](https://github.com/lpq29743/RAM)。

2014 年至 2018 年，于华南师范大学软件学院就读软件工程专业。

2018 年至 2020 年，于中山大学数据科学与计算机学院就读软件工程专业。

## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}