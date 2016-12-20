---
layout:     post
title:      "Python历险记第二站"
subtitle:   "爬取QQ音乐热门华语男歌手"
date:       2016-12-18 16:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 当QQ音乐遇上Python


## 前言

距离上一篇Python主题文章[Python历险记第一站](https://lpq29743.github.io/redant/2016/12/02/PythonLastFMFV/)已经过去将近半个月的时间了，这一篇文章依旧以音乐为主题，我们将爬取的是QQ音乐的热门华语男歌手及其粉丝量。之所以选择QQ音乐，一方面是因为QQ音乐相对国内其他音乐软件，歌手信息比较集中，容易找到爬虫入口，另一方面是因为QQ音乐相比较Last.fm，数据更具有意义，并且难度会适当增大，有利于博主对Python爬虫的学习。

---

## 正文

第一步还是选择爬虫入口，博主选择的是[QQ音乐歌手列表](https://y.qq.com/portal/singerlist.html)，总共爬取10页，共1000个华语男歌手的关注信息，经过一个下午的时间，总算是把程序第一版写出来了：

```python
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class QQMusic:

    #初始化方法
    def __init__(self):
        self.pagenum = 10
        #获取时间
        date = time.strftime("%Y-%m-%d", time.localtime())
        #创建浏览器引擎
        self.driver = webdriver.Firefox()
        #self.driver = webdriver.PhantomJS()

    #点击下一页
    def clickNext(self):
        time.sleep(1)
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "next"))
        )
        element.click()

    #打印数据
    def printItem(self, item):
        self.driver.get(item)
        innerreg = (r'class="data__name_txt js_index" title="(.*?)">.*?[\r\n]+.*?'
                        + 'mod_btn__icon_more.*?</i>关注\s(.*?)</a>')
        innerpattern = re.compile(innerreg, re.S)
        inneritems = innerpattern.findall(self.driver.page_source)
        for inneritem in inneritems:
            print(inneritem[0], inneritem[1])

    #获取数据
    def getItem(self):
        start = time.clock()
        #获取带有图像的歌手信息
        url0 = "https://y.qq.com/portal/singerlist.html#t4=1&t3=all&t2=man&t1=cn&"
        self.driver.get(url0)
        reg = (r'singer_list__item_box">[\r\n]+.*?<a href="(.*?)" class="singer_list__cover')
        pattern = re.compile(reg, re.S)
        items = pattern.findall(self.driver.page_source)
        for item in items:
            self.printItem(item)
            #break
        #获取不带图像的歌手信息
        for x in range(self.pagenum):
            url = "https://y.qq.com/portal/singerlist.html#t4=1&t3=all&t2=man&t1=cn&"
            self.driver.get(url)
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "js_pageindex"))
            )
            if x!=0 and x<4:
                element[x-1].click()
            if x==4:
                element[2].click()
                self.clickNext()
            if x==5:
                element[2].click()
                self.clickNext()
                self.clickNext()
            if x==6:
                element[2].click()
                self.clickNext()
                self.clickNext()
                self.clickNext()
            if x==7:
                element[2].click()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
            if x==8:
                element[2].click()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
            if x==9:
                element[2].click()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
                self.clickNext()
            time.sleep(1)
            reg = (r'singer_list_txt__item"><a href="(.*?)" class="singer_list_txt__link')
            pattern = re.compile(reg, re.S)
            items = pattern.findall(self.driver.page_source)
            for item in items:
                self.printItem(item)
                #break
        end = time.clock()
        print(end-start)
        #self.driver.quit()
        
spider = QQMusic()
spider.getItem()
```

同样是爬取歌手信息，但在QQ音乐的数据爬取却比Last.fm复杂得多，博主在完成这个程序中也遇到了很多问题，现总结如下：

一开始博主选择爬取的url是https://y.qq.com/portal/singerlist.html#t4=1&t3=all&t2=man&t1=cn&，可是用原先的方法发现爬取到的是全部歌手的信息，而不是华语男歌手的信息，但是在浏览器打开此链接却是华语男歌手的信息，这就让我感到纳闷了，难道#号后面的数据没有传过去吗？于是楼主一番谷歌百度，最终终于在阮一峰的博客文章[URL的井号](http://www.ruanyifeng.com/blog/2011/03/url_hash.html)下找到了答案，知道了原因，接下来就是怎么解决了！

博主最终使用了selenium+phantomjs的解决方案，这里的selenium是一个自动化测试工具，可以用pip安装，而phantomjs是一个浏览器引擎，除了phantomjs，也可以选择chrome浏览器或firefox浏览器等等，博主最后因为一些phantomjs上的bug而暂时采用了firefox作为实验的浏览器。无论使用哪一种浏览器，都需要进行环境变量的配置。

关于phantomjs的下载和使用可以参考[这里](http://phantomjs.org/download.html)，对这个工具的学习几乎可以先跳过，重点在于selenium的使用。selenium的用法可以参考[这里](http://cuiqingcai.com/2599.html)，其中页面等待是一个很重要的问题，因为selenium经常用来模拟js操作，如界面渲染就经常要使用到显式等待，而标签点击有时候则要用`time.sleep(secs)`来进行处理。关于selenium文档的查看方法可以参考[这篇文章](http://blog.csdn.net/freesigefei/article/details/50541413)。除此之外，python还有个查看文档的小技巧，直接使用`help()`就能查看某个对象的帮助文档。比如`help(driver)`即可直接查看driver这个对象的文档，包括其内部函数、变量的说明。selenium+phantomjs这个方案总体上还是存在一些问题的，[这篇文章](http://www.jianshu.com/p/9d408e21dc3a)针对这些问题有很不错的讲解。

解决了这些问题之后，程序终于运行出了满意的结果，只不过这运行时间真的有点难以接受，记录了一下运行时间（获取运行时间的方法可以参考[这里](http://www.cnblogs.com/BeginMan/p/3178223.html)，返回的值以秒为单位），总共占用CPU的时间是1952.0409713350145，也就是半个小时左右。虽然说半个小时对于大部分爬虫程序来讲算是正常，但博主觉得问题是出现在自己身上，于是打算在改进改进，最终改进后的程序是

## 后记

