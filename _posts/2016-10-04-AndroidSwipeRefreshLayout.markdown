---
layout:     post
title:      "Android控件之SwipeRefreshLayout"
subtitle:   "下拉刷新控件SwipeRefreshLayout"
date:       2016-10-04 15:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 控件
---

> “原生的下拉刷新控件”


## 前言

Google Android自带下拉刷新控件SwipeRefreshLayout，今天我们就来介绍一下这个家伙

---

## 正文

***布局文件***

```java
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:paddingTop="60dp"
    tools:context="com.attendance.activity.MainActivity">

    <android.support.v4.widget.SwipeRefreshLayout
        android:id="@+id/swipeRefreshLayout_listView"
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <ListView
            android:id="@+id/course_lv"
            android:layout_width="match_parent"
            android:layout_height="match_parent" />

    </android.support.v4.widget.SwipeRefreshLayout>

</FrameLayout>
```

***简单使用***

```java
/**
 * Created by peiqin on 7/28/2016.
 */
public class MainActivity extends AppCompatActivity implements SwipeRefreshLayout.OnRefreshListener {
    private SwipeRefreshLayout mSwipeRefreshLayout;

    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //设置下拉刷新界面
        mSwipeRefreshLayout = (SwipeRefreshLayout) findViewById(R.id.swipeRefreshLayout_listView);
        mSwipeRefreshLayout.setOnRefreshListener(this);
        mSwipeRefreshLayout.setColorSchemeResources(R.color.colorPrimary);
        mSwipeRefreshLayout.setDistanceToTriggerSync(200);// 设置手指在屏幕下拉多少距离会触发下拉刷新
        mSwipeRefreshLayout.setSize(SwipeRefreshLayout.DEFAULT); // 设置圆圈的大小
    }


    /*
     * 监听器SwipeRefreshLayout.OnRefreshListener中的方法，当下拉刷新后触发
     */
    public void onRefresh() {
        // 手动刷新操作
     	// ……
    }

}
```

***设置自动刷新和停止刷新***

```java
SwipeRefreshLayout mSwipeRefreshLayout

//自动下拉刷新
mSwipeRefreshLayout.post(new Runnable() {
	@Override
    public void run() {
		mSwipeRefreshLayout.setRefreshing(true);
	}
});
```

## 后记

这篇博客更多的是给出代码，如果要使用到此控件其他方法，可以去参考一下官方文档。
