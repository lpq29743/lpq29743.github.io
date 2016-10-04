---
layout:     post
title:      "Android控件之AlertDialog"
subtitle:   "警示框控件AlertDialog"
date:       2016-10-04 16:10:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 控件
---

> ProgressDialog的父母——AlertDialog


## 前言

ProgressDialog继承于AlertDialog，让我们通过这篇博客来讲一讲它！

## 正文

***普通提示对话框***

```java
new AlertDialog.Builder(DialogDemo.this)  
                .setIcon(R.drawable.gong1)  
                .setTitle("提示文字Dialog")  
                .setPositiveButton("确定",  
                        new DialogInterface.OnClickListener() {  
                            @Override  
                            public void onClick(DialogInterface dialog,  
                                    int which) {  
                                // TODO Auto-generated method stub  
  
                            }  
                        }).setNegativeButton("取消", null).create()  
                .show();  
```

***长文字对话框***

```java
new AlertDialog.Builder(DialogDemo.this)  
                .setIcon(R.drawable.gong2)  
                .setTitle("提示信息Dialog")  
                .setMessage(  
                        "发现新版本（1.0.1.260）"  
                                + "\n"  
                                + "【全新界面】 全新多屏首页，界面更友好、美观、流程\n"  
                                + "【无线文件传输】 点击无线传输按钮，访问提示地址，可在同一无线网内实现手机和电脑文件互传\n"  
                                + "【快捷面板】 快捷面板全新改造，整合多项功能按钮，图标更大更直观，强大易用\n"  
                                + "    是否升级最新版本？")  
                .setPositiveButton("确定",  
                        new DialogInterface.OnClickListener() {  
                            public void onClick(DialogInterface dialog,  
                                    int whichButton) {  
  
                                /* User clicked OK so do some stuff */  
                            }  
                        }).setNeutralButton("中间按钮", null)  
                .setNegativeButton("取消", null).create().show();
```

***单选项选择对话框***

```java
new AlertDialog.Builder(DialogDemo.this)  
                .setTitle("单选Dialog")  
                .setItems(R.array.dialog_arrays,  
                        new DialogInterface.OnClickListener() {  
                            public void onClick(DialogInterface dialog,  
                                    int which) {  
                                String[] items = getResources()  
                                        .getStringArray(  
                                                R.array.dialog_arrays);  
                                Toast.makeText(  
                                        DialogDemo.this,  
                                        "You selected: " + which  
                                                + " , " + items[which],  
                                        1000).show();  
                            }  
                        }).create().show();
```

其中arrays.xml如下

```java
<?xml version="1.0" encoding="utf-8"?>  
<resources>  
    <string-array name="dialog_arrays">  
        <item >浙江</item>  
        <item >山西</item>  
        <item >山东</item>  
        <item >河南</item>  
        <item >河北</item>  
        <item >广东</item>  
    </string-array>  
      
</resources>
```

## 后记

本文也是继承前几篇博客的风格，对于控件简述的比较简单，需要的朋友可以去查看一下官方文档。
