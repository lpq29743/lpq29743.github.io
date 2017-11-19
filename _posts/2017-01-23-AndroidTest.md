---
layout: post
title: Android 测试
categories: Android
description: Android测试
keywords: Android, Android测试
---

这是2017年的第一篇文章。之前由于期末考试的原因，抽不出时间写文章，这几天放假了，计划安排好了之后，便又着手写博文。

由于本人不计划从事测试方向，所以本文更多的是了解性的文章，如果大家对某个框架比较感兴趣的话，可以去网上查阅相关资料学习。下面就开始介绍常用的Android测试框架及方法。

### 法宝一：Monkey

在学习Monkey之前，必须对Android ADB有一定的了解和熟悉，具体可以先粗略地阅读一下[这篇文章](www.cnblogs.com/devinzhang/archive/2011/12/17/2291396.html)。

Monkey是Android中的一个命令行工具，可以运行在模拟器里或实际设备中。Monkey测试就是让设备随机的乱点，事件都是随机产生的，不带任何人的主观性。所以Monkey测试主要是测试软件的稳定性、健壮性的方法。此外，Monkey也可以用来做简单的自动化测试工作。

Monkey的基本语法如下：

```shell
#向设备发送500个随机事件
adb shell monkey 500
#列出设备中所有的包
adb shell pm list packages
#限制在某个包中，向设备发送500个随机事件
adb shell monkey –p your-package-name 500
#限制在多个包中，向设备发送500个随机事件
adb shell monkey –p your-package1-name –p your-package2-name 500
```

Monkey经常是用来做压力测试，虽然也可以写简单的脚本来进行测试，但是效果极差，所以这里也不多加说明。

### 法宝二：MonkeyRunner

MonkeyRunner工具提供了一个API，使用此API写出的程序可以在Android代码之外控制Android设备和模拟器。通过MonkeyRunner，我们可以写出一个Python程序去安装一个Android应用程序或测试包，运行它，向它发送模拟击键，截取它的用户界面图片，并将截图存储于工作站上。MonkeyRunner工具的主要设计目的是用于测试功能/框架水平上的应用程序和设备，或用于运行单元测试套件。

MonkeyRunner脚本是用Python写的，可以参考[官方中文文档](http://www.android-doc.com/tools/help/monkeyrunner_concepts.html)。下面用一个例子简单讲解一下：

```python
# import monkeyrunner modules
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice, MonkeyImage
# Parameters
txt1_x = 210
txt1_y = 200
txt2_x = 210
txt2_y = 280
txt3_x = 210
txt3_y = 420
txt4_x = 338
txt4_y = 476
submit_x = 100
submit_y = 540
type = 'DOWN_AND_UP'
seconds = 1
txt1_msg = 'Hello'
txt2_msg = 'MonkeyRunner' 
# package name and activity name
package = 'com.ringo.bugben'
activity = '.MainActivity'
component = package + '/'+activity 
# Connect device
device = MonkeyRunner.waitForConnection() 
# Install bugben
device.installPackage('./bugben.apk')
print 'Install bugben.apk...' 
# Launch bugbendevice.startActivity(component)
print 'Launching bugben...' 
# Wait 1s
MonkeyRunner.sle ep(seconds)
# Input txt1
device.touch(txt1_x, txt1_y, type)device.type(txt1_msg)
print 'Inputing txt1...' 
# Input txt2
device.touch(txt2_x, txt2_y, type)
device.type(txt2_msg)
print 'Inputing txt2...' 
#select bold and size
device.touch(txt3_x, txt3_y, type)
device.touch(txt4_x, txt4_y, type) 
# Wait 1s
MonkeyRunner.slee p(seconds) 
# Submitdevice.touch(submit_x, submit_y, type)
print 'Submiting...' 
# Wait 1s
MonkeyRunner.slee p(seconds) 
# Get the snapshot
picture = device.takeSnapshot()
picture.writeToFile('./HelloMonkeyRunner.png','png')
print 'Complete! See bugben_pic.png in currrent folder!' 
# Back to home
device.press('KEYCODE_HOME', type)
print 'back to home.'
```

将脚本保存为MonRun.py，并和apk一起拷贝到Android SDK的tools目录下，执行`monkeyrunner MonRun.py`。执行完后会在当前目录生成截图，利用MonkeyRecorder提供的控件，还可以对脚本进行录制。

### 法宝三：Instrumentation

Monkey和MonkeyRunner均可通过编写相应的脚本，在不依赖源码的前提下完成部分自动化测试的工作。但它们都是依靠控件坐标进行定位的，在实际项目中，控件坐标往往是最不稳定的，随时都有可能因为程序员对控件位置的调整而导致脚本运行失败。怎样可以不依赖坐标来进行应用的自动化测试呢？答案就是使用Instrumentation框架。Instrumentation框架主要是依靠控件的ID来进行定位的，是Android主推的白盒测试框架。

Android API不提供调用Activity周期函数的方法，但在Instrumentation中则可以这样做。Instrumentation类通过“hooks”控制着Android组件的正常生命周期，同时控制Android系统加载应用程序。Instrumentation和Activity类似，只不过Activity需要界面，而Instrumentation并不用。具体的学习大家可以去网上找相应的博文学习，本人这里不打算深入学习。

Instrumentation框架也有局限性，它不支持多应用的交互，例如测试“通过短信中的号码去拨打电话”这个用例，被测应用将从短信应用界面跳转到拨号应用界面，但Instrumentation没有办法同时控制短信和拨号两个应用，这是因为Android系统自身的安全性限制，禁止多应用的进程间相互访问。

### 法宝四：UIAutomator

鉴于Instrumentation框架需要读懂项目源码、脚本开发难度较高并且不支持多应用交互，Android官网亮出了自动化测试的王牌——UIAutomator，并主推这个自动化测试框架。该框架基于Java，无需项目源码，脚本开发效率高且难度低，并且支持多应用的交互。当UIAutomator面世后，Instrumentation框架回归到了其单元测试框架的本来位置。

但UIAutomator难以捕捉到控件的颜色、字体粗细、字号等信息，要验证该类信息的话，需要通过截图的方式进行半自动验证。同时，UIAutomator的调试相比Instrumentation要困难。所以在平时的测试过程中，经常将两者结合使用。

### 法宝五：其他框架

Robotium是基于Instrumentation的测试框架，网上相关资料丰富，但不能跨app；Appium是近几年很热门的测试框架，功能可以说是最强大的。这两个测试框架，尤其是后者，推荐大家可以尝试学习使用。

### 法宝六：各网站测试平台

腾讯、搜狗等各国内大公司提供了很多软件测试的平台，如果大家感兴趣的话，也可以去了解尝试一下。