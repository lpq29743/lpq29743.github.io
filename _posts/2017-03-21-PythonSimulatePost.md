---
layout: post
title: Python 历险记第五站——利用 Python 模拟 Post 请求
categories: Python
description: Python模拟Post请求
keywords: Python
---

这篇文章的源头来源于一个同学让我帮忙写一个刷参与量的程序，这里需要强调，这个程序所解决的刷参与量只是面对一个娱乐性的活动，并不会造成不良影响。另外，这篇文章也不会明确给出该网址

模拟POST请求主要有两种方法，一种是用模拟器进行POST请求，另一种是用代码进行POST请求，一开始我打算用后者实现，但是遇到了一点小问题，于是先试了一下第一种方法，很快代码就写出来了：

```python
import time
from openpyxl import load_workbook
from selenium import webdriver

# 打开excel
wb = load_workbook(filename=r'test.xlsx')

# 获取所有表格(worksheet)的名字
sheets = wb.get_sheet_names()
# 第一个表格的名称
sheet0 = sheets[0]
# 获取特定的 worksheet
ws = wb.get_sheet_by_name('test')
# 打开模拟器
driver = webdriver.Firefox()
hosturl = "XXX"
# 访问hosturl
driver.get(hosturl)
# 记录开始时间
start = time.clock()

# 遍历excel的行
for i in range(ws.max_row):
    # 获取第i+1行的电话号码
    phone_num = ws.cell(row=i+1,column=1).value
    # 获取号码输入框
    elem = driver.find_element_by_id("telephone")
    # 输入号码
    elem.send_keys(phone_num)
    # 获取参与按钮
    elem0 = driver.find_element_by_id("attend")
    # 点击按钮
    elem0.click()
    # 页面刷新
    driver.refresh()
    # 休眠1s
    time.sleep(1)

# 记录结束时间
end = time.clock()
# 输出程序所用时间
print(end-start)
```

虽然这种方法较为简单，但所用的时间实在太久了，特别是由于该网页的原因需要休眠1s，更是占用了额外的时间，所以没办法，我只能尝试第二种方法了，通过查询一些资料，很快代码也写出来了：

```python
import requests
import time
from openpyxl import load_workbook

# 打开excel
wb = load_workbook(filename=r'test.xlsx')
 
# 获取所有表格(worksheet)的名字
sheets = wb.get_sheet_names()
# 第一个表格的名称
sheet0 = sheets[0]
# 获取特定的 worksheet
ws = wb.get_sheet_by_name('test')
# 记录开始时间
start = time.clock()

# 遍历excel的行
for i in range(ws.max_row):
    # 获取第i+1行的电话号码
    phone_num = ws.cell(row=i+1,column=1).value
    # 设置提交数据，其中articleId是文章id，可动态修改
    postData = {
        'articleId': 'xxx',
        'studentId': 'xxx',
        'telephone': phone_num
    }
    # 模拟post请求
    r = requests.post("postUrl", data = postData)
    # 输出response信息
    print(r.text)

# 记录结束时间
end = time.clock()
# 输出程序所用时间
print(end-start)
```

值得一提的是，这里的postUrl和postData需要借助开发者工具才能得到。

关于这两个程序，还有一个小知识点，那就是openpyxl的使用，之前笔者在另一个程序中用的是xlsxwriter，但该库只支持写不支持读，所以笔者换成了这个，如果想要了解更多它的使用可以查看它的[官方文档](http://openpyxl.readthedocs.io/en/default/)。