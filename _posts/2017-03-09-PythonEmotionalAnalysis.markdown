---
layout:     post
title:      "Python历险记第四站"
subtitle:   "利用Python实现情感分析"
date:       2017-03-09 19:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 当Python遇到情绪分析~~~


## 前言

继上篇讲述Python实现词云后，这篇文章将围绕Python实现情绪分析进行展开。实验过程会运用到爬虫和情绪分析的知识。话不多说，让我们开始这篇文章！

---

## 正文

**步骤一：选择数据源**

由于之前在[第二站](https://lpq29743.github.io/redant/2016/12/18/PythonQQMusicMS/)的时候由于网站的特殊原因并没有实现scrapy，所以笔者今天打算顺便学习一下scrapy这个爬虫框架。我选择了我最喜欢的电影《发条橙》的[豆瓣短评](https://movie.douban.com/subject/1292233/comments)作为实验对象。

**步骤二：爬取数据**

不得不承认，我在这一个步骤花了太多的时间，遇到了很多问题，先把代码给出如下：

```python
import scrapy

from scrapy.http import Request
from scrapy.selector import Selector
from emotional_analysis.items import DoubanItem

class DoubanSpider(scrapy.Spider):

    name = 'douban'
    allowed_domains = ['douban.com']
    start_urls = [
        'https://movie.douban.com/subject/1292233/comments'
    ]

    def parse(self, response):
        sel = Selector(response)
        item = DoubanItem()
        item['comment'] = sel.xpath('//div[@class = "comment"]/p[@class = ""]/text()[1]').extract()
        yield item

        next_page = '//a[@class="next"]/@href'
        if sel.xpath(next_page):
            url_nextpage = 'https://movie.douban.com/subject/1292233/comments' + sel.xpath(next_page).extract()[0]
            request = scrapy.Request(url_nextpage, callback = self.parse)
            yield request
```

scrapy是现在运用较为广泛的一个爬虫库，笔者使用的版本是1.3.0。这里提供两个相关的教程，第一个是[Scrapy1.3教程](https://oner-wv.gitbooks.io/scrapy_zh/content/)，第二个是官方的[中文文档](http://scrapy-chs.readthedocs.io/zh_CN/1.0/intro/tutorial.html)），不过版本停留在1.0版本，但本人认为差别不大，所以推荐大家优先参考后者。

关于scrapy的安装，我不会进行讲解，需要的朋友自己寻找教程，这里直接记录一下我在整个实验过程中遇到的问题。

首先是robot.txt的问题，想要了解它更多可以参考[什么样的爬虫才是好爬虫：Robots协议探究](https://segmentfault.com/a/1190000006631364)和[robots.txt 能够封禁网络爬虫吗](https://www.zhihu.com/question/19890668)。scrapy默认会使用robot.txt，所以默认情况下无法访问我们爬取的网站。我们需要进行简单的设置，具体参考[这里](http://stackoverflow.com/questions/37274835/getting-forbidden-by-robots-txt-scrapy)。

其次，豆瓣作为一个经常被爬的网站，它的反爬虫自然也做的很不错，这让我遇到了很多的困难。我尝试了多次改进：

- 设置User-Agent：我用F12获取到了爬取网页的Header，并在scrapy的settings.py文件下进行设置，使得访问不会再返回403
- 验证Xpath：成功访问网页后，却发现无法追踪链接，我把问题锁定到了xpath的正确性上，参考了[资料](http://stackoverflow.com/questions/22571267/how-to-verify-an-xpath-expression-in-chrome-developers-tool-or-firefoxs-firebug)解决了这个问题。值得一提的是，scrapy自带的选择器验证经常会因为403等问题而无法使用，所以推荐大家使用我推荐的资料，另外，即使xpath对上了，有时候还是可能出现xpath错误，所以必要的时候要对xpath进行适当删减进行实验。解决了追踪链接的问题，我却发现在爬取到第10页的时候还是出现了403
- 禁用cookies、设置延迟下载：网上提到的解决爬取豆瓣中途失败的方法，具体可以在settings.py进行设置，但问题却依旧没有得到解决
- 使用代理：没有尝试成功，感兴趣的朋友可以自行尝试
- 设置Referer：经过一番资料搜索和独立思考，发现问题很大可能是出现在Referer这里，可是由于Referer的动态变化让我无从下手
- 使用豆瓣API：豆瓣提供了API供用户获取数据，但由于只是小实验，所以本人不计划尝试，需要的朋友请自行查询资料

多次尝试无果后，我决定用前九页的数据先进行实验。我执行了命令`scrapy crawl douban -o items.json`爬取到数据并保存到文件中。在本人的机器上，保存的编码是Unicode，所以我使用了[工具](http://tool.chinaz.com/tools/unicode.aspx)将其转换成了中文。

关于爬取数据的讲述就先到这里了，接下来我们看看怎么情绪分析我们获取到的数据！

**步骤三：情绪分析**

目前常见的情绪分析方法主要是两种：基于情感词典的方法和基于机器学习的方法，我们这里采用的是后者。

首先我们要下载[豆瓣网影评情感测试语料](http://www.datatang.com/data/13539)，然后我们要用我们之前实验用到的工具jieba对数据进行分词处理，最后则是要进行模型的构建，这样子我们就可以得到我们想要的情绪分析结果了！

## 后记

不得不承认，笔者的这篇文章写的相当糟糕，这也与我自己的知识储量有很大的关联。接下来我也会加紧我的学习，慢慢改善这篇文章。
