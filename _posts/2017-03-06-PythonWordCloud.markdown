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

下载完实验文件后，我们就要下载相应的库。第一个库就是wordcloud（[项目地址](https://github.com/amueller/word_cloud)），它是Python下一个用于制作词云的库；此外，我们还要下载python-docx读写docx，处理word文档的python库主要有：python-docx（只能读写docx）、win32com（只能用于Windows）和antiword（可用于Linux环境读写doc，但是对中文的支持不是特别好），为了尽量方便，我们决定使用第一个库，并把我们的实验文件改为docx形式；最后我们还要安装一个中文分词库jieba（[源码地址](https://github.com/fxsjy/jieba)）,当然大家也可以选择THULAC作为中文分词工具，如果想了解更多关于Python下中文分词方案的内容，请点击[这里](https://www.zhihu.com/question/20294818)。下载完相应的库后，我们就可以写出第一版的代码了：

```python
# coding:utf-8
import docx
import jieba.analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the whole text.
document = docx.Document('test.docx')
content = ""
for ps in document.paragraphs:
    content += ps.text

# Use jieba to cut the content.
tags = jieba.cut(content)
text =" ".join(tags)

# Use wordcloud to draw the picture.
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('test0.png')
```

代码终于写好了，在这个过程中，我遇到了很多问题，总结如下：

1. 首先是matplotlib对中文字体的支持问题，对于这个问题，我已经在知乎进行了[总结](https://www.zhihu.com/question/25404709/answer/150519029)。对于matplotlib，还有`plt.show()`没有效果等问题，但这可能归因于CentOS7这个实验环境，这里就不深究了
2. 其次是wordcloud对中文字体的支持问题，具体解决方法可以参考[这里](https://zhuanlan.zhihu.com/p/20436581)

这个简单的版本做好之后，我们要做进一步的改进。在改进之前，我们先提供一个[关于获取谷歌个人搜索数据](https://support.google.com/accounts/answer/3024190?visit_id=1-636246347096333205-2517223644&rd=2)的链接，感兴趣的朋友可以通过上面这个例子分析自己的谷歌搜索数据。至于改进，我们将会在两方面进行改进：

1. 上面的实验生成的图显得有点不美观，我们将会试图定制自己的词云（[参考资料](https://github.com/amueller/word_cloud/blob/master/examples/masked.py)）
2. 将文本替代成url来进行实验，至于实验对象，就用维基百科的[中国](https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9C%8B)词条好了

一番折腾之后，第二版代码就新鲜出炉了：

```python
# coding:utf-8
# coding:utf-8
import re
import requests
from lxml import html
from os import path
from PIL import Image
import numpy as np
import jieba.analyse
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Read the whole text.
url = "https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9C%8B"
response = requests.get(url)

origin_text = response.text
origin_text = re.sub(u'<script.*?>.*?</script>', '', origin_text, flags=re.I|re.M|re.DOTALL)
origin_text = re.sub(u'<style.*?>.*?</style>', '', origin_text, flags=re.I|re.M|re.DOTALL)

doc = html.fromstring(origin_text)
content = doc.xpath('//body//text()')
content = [i.strip() for i in content if i.strip()]
content = u' '.join(content)

# Use jieba to cut the content.
tags = jieba.cut(content)
text =" ".join(tags)

# Use wordcloud to draw the picture.
d = path.dirname(__file__)
alice_mask = np.array(Image.open(path.join(d, "alice_mask.png")))
stopwords = set(STOPWORDS)
stopwords.add("said")
wordcloud = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
               stopwords=stopwords).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('test1.png')
```

## 后记
至此，关于词云的学习就先到这里了。词云能够帮助我们更快地知道一段文字的重点，知道某个网页的热点，提高了数据可读性，感兴趣的朋友都可以学一学。接下来我将尝试做Python的情绪分析，敬请期待！

