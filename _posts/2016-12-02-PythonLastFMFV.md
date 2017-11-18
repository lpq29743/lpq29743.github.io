---
layout: post
title: Python 历险记第一站——爬取 last.fm 热门女歌手
categories: Python
description: Python爬取last.fm热门女歌手
keywords: Python, Python爬虫, 爬虫
---

紧接着上一篇文章，这一篇就讲一个Python爬虫实验。博主一直想要在网易云音乐上建个关于热门女歌手的歌单，但是怎样才算热门女歌手呢？最后博主决定以Last.fm的收听量作为标准来进行判定，所以也就有了用爬虫来爬取收听量超过一百万的女歌手的打算。

首先第一步要选取网页，博主仔细甄选，最后选择了http://www.last.fm/zh/tag/female+vocalists/artists?page=1这个网页，其中后面的页数范围从1到50。经过了一个晚上到早上的时间（博主第一次搞，经验不足，所以花的时间比较多），写出了第一版的代码：

```python
import urllib
import urllib.request
import re

num = 0
for x in range(50):
    page = x+1
    url = "http://www.last.fm/zh/tag/female+vocalists/artists?page="+str(page)
    try:
        data = urllib.request.urlopen(url).read()
        data = data.decode('UTF-8')
        reg = (r'class=\"link-block-target\"[\r\n]+.*?>(.*?)</a>[\r\n]+.*?</p>'
        '[\r\n]+.*?<p class=\"grid-items-item-aux-text\">[\r\n]+\s*(.*?)\s位'
        '<span class=\"stat-name\">')
        pattern = re.compile(reg, re.S)
        items = pattern.findall(data)
        for item in items:
            if int(item[1])>1000000:
                print(item[0], item[1])
                num = num + 1
    except urllib.error.URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
print(num)
```

总共爬到了119个歌手，由于用的是“female vocalists”这个标签，所以爬出来的结果可能会有所偏差，这个也希望后面能改进。在Python编程中，面向对象的编程思想也运用的很广泛，所以博主把它稍微修改一下有了以下的代码：

```python
import urllib
import urllib.request
import re

class LastFM:

    #初始化方法
    def __init__(self):
        self.num = 0
        self.pagenum = 50

    #获取数据
    def getItem(self):
        for x in range(self.pagenum):
            page = x+1
            url = "http://www.last.fm/zh/tag/female+vocalists/artists?page="+str(page)
            try:
                data = urllib.request.urlopen(url).read()
                data = data.decode('UTF-8')
                reg = (r'class=\"link-block-target\"[\r\n]+.*?>(.*?)</a>[\r\n]+.*?</p>'
                    '[\r\n]+.*?<p class=\"grid-items-item-aux-text\">[\r\n]+\s*(.*?)\s位'
                    '<span class=\"stat-name\">')
                pattern = re.compile(reg, re.S)
                items = pattern.findall(data)
                for item in items:
                    if int(item[1])>1000000:
                        print(item[0], item[1])
                        self.num = self.num + 1
            except urllib.error.URLError as e:
                print('We failed to reach a server.')
                print('Reason: ', e.reason)
        print(self.num)

spider = LastFM()
spider.getItem()
```

以上程序获得数据之后只是在控制台输出，而一般情况下我们抓取数据都需要保存下来的，这里博主打算用excel来存储数据。在实验过程中，博主尝试了xlwt、openpyxl以及xlsxwriter三种工具，最后选择了最后一种，关于这三种的比较，可以看[这里](http://ju.outofmemory.cn/entry/56671)。花了一个上午，最后的代码是：

```python
import urllib
import urllib.request
import re
import xlsxwriter
import time

class LastFM:

    #初始化方法
    def __init__(self):
        self.num = 0
        self.pagenum = 50
        #获取时间
        date = time.strftime("%Y-%m-%d", time.localtime())
        #建立Excel
        self.workbook = xlsxwriter.Workbook("lastfm_femalesinger-" + date + ".xlsx")
        self.sheet = self.workbook.add_worksheet('lastfm_femalesinger')
        self.item = ['歌手名','收听人数']
        for i in range(2):
            self.sheet.write(0,i,self.item[i])

    #获取数据
    def getItem(self):
        for x in range(self.pagenum):
            page = x+1
            url = "http://www.last.fm/zh/tag/female+vocalists/artists?page="+str(page)
            try:
                data = urllib.request.urlopen(url).read()
                data = data.decode('UTF-8')
                reg = (r'class=\"link-block-target\"[\r\n]+.*?>(.*?)</a>[\r\n]+.*?</p>'
                    '[\r\n]+.*?<p class=\"grid-items-item-aux-text\">[\r\n]+\s*(.*?)\s位'
                    '<span class=\"stat-name\">')
                pattern = re.compile(reg, re.S)
                items = pattern.findall(data)
                for item in items:
                    if int(item[1])>1000000:
                        self.num = self.num + 1
                        self.sheet.write(self.num,0,item[0])
                        self.sheet.write(self.num,1,item[1])
                print(self.num)
            except urllib.error.URLError as e:
                print('We failed to reach a server.')
                print('Reason: ', e.reason)
        self.workbook.close()

spider = LastFM()
spider.getItem()

```

**20170927更新**

由于网站的改版，原先的代码已经无效了，所以我根据网站改版后情况和现有能力修改代码如下：

```python
import requests
from lxml import etree
import re
import sys
import xlsxwriter
import time

class LastFM:

    #初始化方法
    def __init__(self):
        self.num = 0
        self.pagenum = 50
        #获取时间
        date = time.strftime("%Y-%m-%d", time.localtime())
        #建立Excel
        self.workbook = xlsxwriter.Workbook("lastfm_femalesinger-" + date + ".xlsx")
        self.sheet = self.workbook.add_worksheet('lastfm_femalesinger')
        self.item = ['歌手名','收听人数']
        for i in range(2):
            self.sheet.write(0, i, self.item[i])

    #获取数据
    def getItem(self):
        for x in range(self.pagenum):
            page = x+1
            url = "http://www.last.fm/zh/tag/female+vocalists/artists?page="+str(page)
            try:
                response = requests.get(url)
                root = etree.fromstring(response.content, etree.HTMLParser())
                result = root.xpath('//div[contains(@class, "big-artist-list-item")]')
                for i in result:
                    name = i.xpath("./h3/a/text()")[0]
                    fans_num = int(re.sub('\D', '', i.xpath("./p/text()")[0]))
                    if fans_num > 1000000:
                        self.num = self.num + 1
                        self.sheet.write(self.num, 0, name)
                        self.sheet.write(self.num, 1, fans_num)
                print(self.num)
            except:
                print(sys.exc_info()[0])
        self.workbook.close()

spider = LastFM()
spider.getItem()
```