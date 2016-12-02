---
layout:     post
title:      "Python历险记第一章"
subtitle:   "爬取last.fm热门女歌手"
date:       2016-12-02 10:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> last.fm的正确打开方式


## 前言

紧接着上一篇文章，这一篇就讲一个Python爬虫实验。博主一直想要在网易云音乐上建个关于热门女歌手的歌单，但是怎样才算热门女歌手呢？最后博主决定以Last.fm的收听量作为标准来进行判定，所以也就有了用爬虫来爬取收听量超过一百万的女歌手的打算。

---

## 正文

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

class LastFM:

    #初始化方法
    def __init__(self):
        self.num = 0
        self.pagenum = 50
        #建立Excel
        self.workbook = xlsxwriter.Workbook("lastfm_femalesinger.xlsx")
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

## 后记

不实验不知道，原来爬虫入门并没有我们想象的那么复杂，我也从中收获不少乐趣，接下来会再接再厉，献上更好的作品。
