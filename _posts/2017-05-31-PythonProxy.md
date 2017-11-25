---
layout: post
title: Python 历险记第七站——搭建 IP 代理库
categories: Python
description: Python搭建IP代理库
keywords: Python, IP代理库
---

对于豆瓣等有反爬虫机制的网站，使用IP代理爬虫是经常的事，于是我参考了知乎上的[这个讨论](https://www.zhihu.com/question/47464143)，写下来这篇文章。

网上有很多提供IP代理的网站，比较知名的有快代理和西刺代理，所以我们首先要爬取这两个网站上的IP，并将其中有效的IP代理存进我们的数据库中，具体代码如下：

**快代理**

```python
import requests
from lxml import etree
import sys
import time
import threading
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Proxy(Base):
    __tablename__ = 'proxy'
    id = Column(Integer, primary_key=True)
    ip = Column(String(20))
    port = Column(String(20))
    date = Column(String(20))


engine = create_engine('mysql+pymysql://root:@localhost:3306/proxy?charset=utf8')
DBSession = sessionmaker(bind=engine)


class getProxy(threading.Thread):

    def __init__(self, page):
        self.user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        self.headers = {"User-Agent": self.user_agent}
        self.now = time.strftime("%Y-%m-%d")
        threading.Thread.__init__(self)
        self.page = page
        self.thread_stop = False

    def getContent(self, page):
        nn_url = "http://www.kuaidaili.com/free/inha/" + str(page)
        proxy = {'http': "49.140.80.7:8998"}
        try:
            response = requests.get(nn_url, headers=self.headers)
            if response.status_code == 200:
                root = etree.fromstring(response.content, etree.HTMLParser())
                result = root.xpath('//tbody//tr')
                for i in result:
                    t = i.xpath("./td/text()")[:2]
                    if self.isAlive(t[0], t[1]):
                        print("Page:%d   IP:%s   Port:%s" % (self.page, t[0], t[1]))
                        self.insertDB(t[0], t[1])
            else:
                print("Maybe your ip has been blocked")
        except:
            print(sys.exc_info()[0])

    def insertDB(self, ip, port):
        session = DBSession()
        new_proxy = Proxy(ip=ip, port=port, date=self.now)
        session.add(new_proxy)
        session.commit()
        session.close()

    def isAlive(self, ip, port):
        proxy = {'http': ip+':'+port}
        test_url = "http://ip.cip.cc"
        try:
            response = requests.get(test_url,
                                    headers=self.headers,
                                    proxies=proxy,
                                    timeout=10)
            if response.status_code == 200:
                return True
            else:
                print("not work")
                return False
        except:
            print("not work")
            return False

    def run(self):
        while not self.thread_stop:
            self.getContent(self.page)
            self.stop()

    def stop(self):
        self.thread_stop = True


if __name__ == "__main__":
    for i in range(0, 1000):
        try:
            getProxy(i).start()
            time.sleep(5)
        except:
            print(sys.exc_info()[0])
```

**西刺代理**

```python
import requests
from lxml import etree
import sys
import time
import threading
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Proxy(Base):
    __tablename__ = 'proxy'
    id = Column(Integer, primary_key=True)
    ip = Column(String(20))
    port = Column(String(20))
    date = Column(String(20))

engine = create_engine('mysql+pymysql://root:@localhost:3306/proxy?charset=utf8')
DBSession = sessionmaker(bind=engine)


class getProxy(threading.Thread):

    def __init__(self, page):
        self.user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        self.headers = {"User-Agent": self.user_agent}
        self.now = time.strftime("%Y-%m-%d")
        threading.Thread.__init__(self)
        self.page = page
        self.thread_stop = False

    def getContent(self, page):
        nn_url = "http://www.xicidaili.com/nn/" + str(page)
        proxy = {'http': "49.140.80.7:8998"}
        try:
            response = requests.get(nn_url, headers=self.headers)
            if response.status_code == 200:
                root = etree.fromstring(response.content, etree.HTMLParser())
                result_even = root.xpath('//tr[@class=""]')
                result_odd = root.xpath('//tr[@class="odd"]')
                for i in result_even:
                    t1 = i.xpath("./td/text()")[:2]
                    if self.isAlive(t1[0], t1[1]):
                        print("Page:%d   IP:%s   Port:%s" % (self.page, t1[0], t1[1]))
                        self.insertDB(t1[0], t1[1])
                for i in result_odd:
                    t2 = i.xpath("./td/text()")[:2]
                    if self.isAlive(t2[0], t2[1]):
                        print("Page:%d   IP:%s   Port:%s" % (self.page, t2[0], t2[1]))
                        self.insertDB(t2[0], t2[1])
            else:
                print("Maybe your ip has been blocked")
        except:
            print(sys.exc_info()[0])

    def insertDB(self, ip, port):
        session = DBSession()
        new_proxy = Proxy(ip=ip, port=port, date=self.now)
        session.add(new_proxy)
        session.commit()
        session.close()

    def isAlive(self, ip, port):
        proxy = {'http': ip+':'+port}
        test_url = "http://ip.cip.cc"
        try:
            response = requests.get(test_url,
                                    headers=self.headers,
                                    proxies=proxy,
                                    timeout=10)
            if response.status_code == 200:
                return True
            else:
                print("not work")
                return False
        except:
            print("not work")
            return False

    def run(self):
        while not self.thread_stop:
            self.getContent(self.page)
            self.stop()

    def stop(self):
        self.thread_stop = True


if __name__ == "__main__":
    for i in range(0, 100):
        try:
            getProxy(i).start()
            time.sleep(5)
        except:
            print(sys.exc_info()[0])
```

这些网站提供的免费IP代理并不是长期有效，所以我们还需要编写一个程序定期判断代理IP是否有效，具体如下：

```python
import requests
from lxml import etree
import sys
import time
import threading
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Proxy(Base):
    __tablename__ = 'proxy'
    id = Column(Integer, primary_key=True)
    ip = Column(String(20))
    port = Column(String(20))
    date = Column(String(20))

engine = create_engine('mysql+pymysql://root:@localhost:3306/proxy?charset=utf8')
DBSession = sessionmaker(bind=engine)


class testProxy(threading.Thread):

    def __init__(self, ip, port):
        self.user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        self.headers = {"User-Agent": self.user_agent}
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.thread_stop = False

    def test(self, ip, port):
        if self.isAlive(ip, port):
            print("Still work   IP:%s   Port:%s" % (ip, port))
        else:
            session = DBSession()
            affected_rows = session.query(Proxy)
            .filter_by(ip=ip)
            .filter_by(port=port)
            .delete()
            if affected_rows >= 1:
                print("Delete   IP:%s   Port:%s" % (ip, port))
            session.close()

    def isAlive(self, ip, port):
        proxy = {'http': ip+':'+port}
        test_url = "http://ip.cip.cc"
        try:
            response = requests.get(test_url,
                                    headers=self.headers,
                                    proxies=proxy,
                                    timeout=10)
            if response.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def run(self):
        while not self.thread_stop:
            self.test(self.ip, self.port)
            self.stop()

    def stop(self):
        self.thread_stop = True


if __name__ == "__main__":
    session = DBSession()
    proxies = session.query(Proxy)
    for proxy in proxies:
        try:
            testProxy(proxy.ip, proxy.port).start()
            time.sleep(1)
        except:
            print(sys.exc_info()[0])
    session.close()
```

只要把握好上面这几个程序的定期启动时间，我们就可以获得一个属于自己的IP代理库了！