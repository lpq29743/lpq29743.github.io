---
layout:     post
title:      "Python历险记第三站"
subtitle:   "利用Python实现词云"
date:       2017-03-06 21:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Python
---

> 当Python遇到数据可视化~~~


## 前言

词云是数据可视化的一种形式，网上也有一些关于制作词云的网站，具体可以看[这里](https://www.zhihu.com/question/28382979)。实际上，Python也有关于词云实现的相关工具库，今天我们就一起来看看怎么用Python实现词云。

---

## 正文

我们首先先用本地文件做实验。最近两会召开，所以第一个实验对象就是[习总书记的重大讲话](http://pan.baidu.com/share/link?uk=2265408988&shareid=787234409#list/path=%2F%E5%BC%A5%E6%94%BF%E5%8A%9E%E7%BD%91%E6%A0%A1%2F%E7%BD%91%E6%A0%A1%2F%E4%B8%A4%E5%AD%A6%E4%B8%80%E5%81%9A%2F%E4%B9%A0%E8%BF%91%E5%B9%B3%E6%80%BB%E4%B9%A6%E8%AE%B0%E7%B3%BB%E5%88%97%E9%87%8D%E8%A6%81%E8%AE%B2%E8%AF%9D&parentPath=%2F%E5%BC%A5%E6%94%BF%E5%8A%9E%E7%BD%91%E6%A0%A1%2F%E7%BD%91%E6%A0%A1%2F%E4%B8%A4%E5%AD%A6%E4%B8%80%E5%81%9A)。

下载完实验文件后，我们就要下载相应的库。第一个库就是wordcloud（[项目地址](https://github.com/amueller/word_cloud)），它是Python下一个用于制作词云的库；此外，我们还要下载python-docx读写docx，处理word文档的python库主要有：python-docx（只能读写docx）、win32com（只能用于Windows）和antiword（可用于Linux环境读写doc，但是对中文的支持不是特别好），为了尽量方便，我们决定使用第一个库，并把我们的实验文件改为docx形式。下载完相应的库后，我们就可以写出第一版的代码了：

```python
# coding:utf-8
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the whole text.
document = docx.Document('test.docx')
text = ""
for ps in document.paragraphs:
    text += ps.text


# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('test.png')
```

代码终于写好了，在这个过程中，我遇到了很多问题，总结如下：

1. 首先是matplotlib对中文字体的支持问题，对于这个问题，我已经在知乎进行了[总结](https://www.zhihu.com/question/25404709/answer/150519029)。对于matplotlib，还有`plt.show()`没有效果等问题，但这可能归因于CentOS7这个实验环境，这里就不深究了。
2. 其次是wordcloud本身的字体问题，当然这里代码并没有涉及到，只是我在实验中接触到

## 后记


