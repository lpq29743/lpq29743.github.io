---
layout: page
title: About
description: 林佩勤的个人介绍
keywords: Peiqin Lin, 林佩勤
comments: true
menu: 关于
permalink: /about/
---

林佩勤，自然语言处理爱好者，电影摇滚爱好者

## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}