---
layout: post
title: Python 历险记第二站——爬取 QQ 音乐热门华语男歌手
categories: Python
description: Python爬取QQ音乐热门华语男歌手
keywords: Python, Python爬虫, 爬虫
---

距离上一篇Python主题文章[Python历险记第一站](https://lpq29743.github.io/redant/python/2016/12/02/PythonLastFMFV/)已经过去将近半个月的时间了，这一篇文章依旧以音乐为主题，我们将爬取的是QQ音乐的热门华语男歌手及其粉丝量。之所以选择QQ音乐，一方面是因为QQ音乐相对国内其他音乐软件，歌手信息比较集中，容易找到爬虫入口，另一方面是因为QQ音乐相比较Last.fm，数据更具有意义，并且难度会适当增大，有利于博主对Python爬虫的学习。

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

解决了这些问题之后，程序终于运行出了满意的结果，只不过这运行时间真的有点难以接受，记录了一下运行时间（获取运行时间的方法可以参考[这里](http://www.cnblogs.com/BeginMan/p/3178223.html)，返回的值以秒为单位），总共占用CPU的时间是1952.0409713350145，也就是半个小时左右。虽然说半个小时对于大部分爬虫程序来讲算是正常，但博主觉得问题是出现在自己身上，于是打算再改进改进。

博主尝试过采用scrapy+splash，也尝试过采用pyspider，但后面都因为这个链接地址中可恶的#号而受挫了，于是改进后的代码依旧没有采用这两个框架，不过博主也是通过这一次实验对这两个框架有了初步的了解，并且对css选择器以及xpath有了一定的掌握。

通过在知乎看到的问题[《如果网页内容是由javascript生成的，应该怎么实现爬虫呢？》](https://www.zhihu.com/question/27734572)，博主打算模拟js操作来避免使用浏览器带来的开销，可依旧是#号堵住了我的去路，后面如果有新的突破，博主也会回来修改这一部分的内容。

总而言之，博主改进后的代码只实现了mysql的操作，具体代码如下：

```python
#encoding=utf-8
import re
import time
import pymysql
from pymysql import MySQLError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class QQMusic:

    #初始化方法
    def __init__(self):
        self.pagenum = 10
        #打开数据库连接
        self.db = pymysql.connect("localhost","root","","qqmusic_spider",charset="utf8")
        #获取操作游标
        self.cursor = self.db.cursor()
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

    #保存数据
    def printItem(self, item):
        self.driver.get(item)
        innerreg = (r'class="data__name_txt js_index" title="(.*?)">.*?[\r\n]+.*?'
                        + 'mod_btn__icon_more.*?</i>关注\s(.*?)</a>')
        innerpattern = re.compile(innerreg, re.S)
        inneritems = innerpattern.findall(self.driver.page_source)
        for inneritem in inneritems:
            pos = inneritem[1].find("万")
            if pos!=-1:
                try:
                    self.cursor.execute('insert into malesinger(name,fans_num) values ("%s","%f")' % \
                                        (inneritem[0], float(inneritem[1][0:pos])))
                    self.db.commit()
                    print("success")
                except MySQLError as e:
                    print('Got error {!r}, errno is {}'.format(e, e.args[0]))
                    self.db.rollback()

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
        self.db.close()
        
spider = QQMusic()
spider.getItem()
```

数据库相关sql语句如下：

```mysql
create database qqmusic_spider;
use qqmusic_spider;
create table malesinger (
	id int not null auto_increment primary key,
    name varchar(20) not null,
    fans_num float not null
);
```

这个程序总共花了1584.1502326928423，总体开销比之前小。在这一版的代码博主遇到的主要问题是编码的问题，所以博主也提醒大家要注意编码问题，从数据库的编码到表的编码，从字段编码到抓取数据的编码，都要保持统一，最好都是utf-8。此外，mysql数据库操作最好都要附带捕获MySQLError的操作代码，这样可以很清晰地判断自己的sql语句有没有写错。

总而言之，这次爬虫实验的情况还是蛮不错的，上一个实验很适合入门，这一个则很适合进阶。博主从中学到了很多东西，包括这一篇文章中还没有提及的[反爬虫](https://segmentfault.com/a/1190000005840672)、[robots.txt](https://www.zhihu.com/question/19890668)等等。希望接下来继续努力，不断提高自己的水平！