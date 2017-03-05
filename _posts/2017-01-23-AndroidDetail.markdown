---
layout:     post
title:      "Android的细枝末节"
subtitle:   "聊聊Android的小知识点"
date:       2017-01-23 19:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> 不定期更新的Android笔记~~


## 前言

本人在大一学年开始接触到Android开发，主要是因为做项目的需要，由于投入了较多的时间和精力，所以对这项开发技术多多少少存在一些情感。虽然现在小程序、WebApp等的出现给Android开发市场带来了不小的挑战，但出于兴趣，本人还是会坚持Android学习，尽管将来很大程度上不以Android为就业方向，但相信在这个学习、研究和实践的过程中，一定能够收获到很多东西。Android这门开发语言设计到的大框架、大技术值得每一位想要深入Android开发的朋友进行学习熟悉，但一些小的细节，一些常用到的东西也同样需要引起我们的关注。如果博主把这些小的知识点放在一起的话，未免显得有些奇怪，所以博主希望通过这篇文章，不定期的记录一些Android的小知识点。

---

## 正文

**获取网络时间**

由于Android获取网络时间需要访问网络，所以要进行异步操作，具体如下：

```java
private class MyTask extends AsyncTask<Void, Void, Void> {

        @Override
        protected void onPreExecute() {
        }

        @Override
        protected void doInBackground(Void... params) {
            String timeUrl = "http://www.beijing-time.org";
            try {
                URL url = new URL(timeUrl);// 取得资源对象
                URLConnection uc = url.openConnection();// 生成连接对象
                uc.connect();// 发出连接
                long ld = uc.getDate();// 读取网站日期时间
                Date date = new Date(ld);
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss EEEE", Locale.CHINA);// 输出北京时间
                Log.e("Time", sdf.format(date));
              	SimpleDateFormat sdf1 = new SimpleDateFormat("EEEE", Locale.CHINA);
              	Log.e("Weekday", sdf1.format(date));// 输出星期几，其中Sunday是星期日，不是星期天
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return;
        }

        @Override
        protected void onPostExecute(Void result) {
        }
}
```

**锁定Activity屏幕方向**

有时应用程序仅能在横屏或竖屏时运行，此时我们需要锁定该Activity运行时的屏幕方向，示例代码如下：

```xml
<activity android:name=".EX01"
	android:label="@string/app_name" 
	android:screenOrientation="portrait">// 竖屏 , 值为 landscape 时为横屏
	…………
</activity>
```

**Activity全屏**

要使Activity全屏，可以在其`onCreate()`方法中添加如下代码实现：

```java
// 设置全屏模式
getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, 
WindowManager.LayoutParams.FLAG_FULLSCREEN); 
// 去除标题栏
requestWindowFeature(Window.FEATURE_NO_TITLE);
```

**EditView设置单行**

`android:singleLine="true"`可以设置单行，但已经过时了，现在的设置方法为：

```java
// 必须加上这一行，否则设置无效
android:inputType="text"
android:maxLines="1"
```

**EditView软键盘弹出**

对于跳转新界面就要弹出软键盘的情况，可能会出现由于界面未加载完全而无法弹出软键盘的情况。此时应该适当的延迟弹出软键盘（保证界面的数据加载完成）。实例代码如下：

```java
Timer timer = new Timer();
timer.schedule(new TimerTask() {

	public void run() {
		InputMethodManager inputManager = (InputMethodManager) mSearchEt.getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
		inputManager.showSoftInput(mSearchEt, 0);
	}

}, 300);
```

## 后记




