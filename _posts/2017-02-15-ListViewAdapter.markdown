---
layout:     post
title:      "自定义ListView的Adapter"
subtitle:   "谈谈如何优化ListView"
date:       2017-02-15 20:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> 今天讲一讲ListView~~~


## 前言

ListView在Android中的运用相当普遍。学习ListView，自定义适配器是永远无法跳过的知识点，今天就让我们一起来讲一讲ListView。

---

## 正文

我们先创建一个Item类，该类包括img和text两个属性，具体如下：

```java
package com.activitydemo;

/**
 * Created by peiqin on 2/15/2017.
 */
public class Item {

    private int img;
    private String text;

    public Item(int img, String text) {
        this.img = img;
        this.text = text;
    }

    public int getImg() {
        return img;
    }

    public void setImg(int img) {
        this.img = img;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
```

然后再创建item视图：

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical">
    
    <TextView
        android:id="@+id/item_tv"
        android:layout_width="match_parent"
        android:layout_height="wrap_content" />
    
    <ImageView
        android:id="@+id/item_iv"
        android:layout_width="match_parent"
        android:layout_height="wrap_content" />

</LinearLayout>
```

接下来就可以写我们自定义的适配器了：

```java
package com.activitydemo;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.List;

/**
 * Created by peiqin on 2/15/2017.
 */
public class MyAdapter extends BaseAdapter {

    private List<Item> mData;//定义数据。
    private LayoutInflater mInflater;//定义Inflater,加载我们自定义的布局。

    /*
    定义构造器，在Activity创建对象Adapter的时候将数据data和Inflater传入自定义的Adapter中进行处理。
    */
    public MyAdapter(LayoutInflater inflater,List<Item> data){
        mInflater = inflater;
        mData = data;
    }

    @Override
    public int getCount() {
        return mData.size();
    }

    @Override
    public Object getItem(int position) {
        return position;
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup viewGroup) {
        //获得ListView中的view
        View view = mInflater.inflate(R.layout.item_custom,null);
        //获得学生对象
        Item item = mData.get(position);
        //获得自定义布局中每一个控件的对象。
        ImageView imagePhoto = (ImageView) view.findViewById(R.id.item_iv);
        TextView name = (TextView) view.findViewById(R.id.item_tv);
        //将数据一一添加到自定义的布局中。
        imagePhoto.setImageResource(item.getImg());
        name.setText(item.getText());
        return view ;
    }
}
```

最后我们再在我们的Activity中运用这个自定义的适配器就可以了，代码如下：

```java
package com.activitydemo;

import android.app.Activity;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.widget.ListView;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by peiqin on 2/15/2017.
 */
public class ListViewActivity extends Activity {

    //定义数据
    private List<Item> mData;
    //定义ListView对象
    private ListView mListView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //为ListView对象赋值
        mListView = (ListView) findViewById(R.id.custom_lv);
        LayoutInflater inflater = getLayoutInflater();
        //初始化数据
        initData();
        //创建自定义Adapter的对象
        MyAdapter adapter = new MyAdapter(inflater, mData);
        //将布局添加到ListView中
        mListView.setAdapter(adapter);

    }

    /*
    初始化数据
     */
    private void initData() {
        mData = new ArrayList<>();
        mData.add(new Item(android.R.mipmap.sym_def_app_icon, "text1"));
        mData.add(new Item(android.R.mipmap.sym_def_app_icon, "text2"));
        mData.add(new Item(android.R.mipmap.sym_def_app_icon, "text3"));
        mData.add(new Item(android.R.mipmap.sym_def_app_icon, "text4"));
    }

}
```

这样子简单的自定义适配器就完成了，但是这样子ListView的效率极差，所以我们还要对它进行优化。具体的ListView优化方法有：

1. 在Adapter中的getView方法中使用ConvertView，即ConvertView的复用，不需要每次都inflate一个View出来，这样既浪费时间，又浪费内存。

2. 使用ViewHolder，不要在getView方法中写findViewById方法，因为getView方法会执行很多遍，这样也可以节省时间，节约内存。

   结合上面两种优化方法，可以将getView的代码改写成如下代码进行优化：

   ```java
   @Override
   public View getView(int position, View convertView, ViewGroup viewGroup) {
   	ViewHolder vh;
   	if (convertView == null) {
   		convertView = mInflater.inflate(R.layout.item_custom, null);
   		vh = new ViewHolder();
   		vh.text = (TextView) convertView.findViewById(R.id.item_tv);
   		vh.img = (ImageView) convertView.findViewById(R.id.item_iv);
   		convertView.setTag(vh);
   	} else {
   		vh = (ViewHolder) convertView.getTag();
   	}
   	Item item = mData.get(position);
   	vh.text.setText(item.getText());
   	vh.img.setImageResource(item.getImg());
   	return convertView;
   }

   private class ViewHolder {
   	public TextView text;
   	public ImageView img;
   }
   ```

3. 分页加载，实际开发中，ListView的数据肯定不止几百条，成千上万条数据不可能一次性加载出来，所以需要用到分页加载，一次加载几条或者十几条，但如果数据量很大，即使顺利加载到最后面，list中也会有几万甚至几十万的数据，这样可能会导致OOM，所以每加载一页的时候可以覆盖前一页的数据。

4. 如果数据当中有图片的话，使用第三方库来加载(也就是缓存)，如果你的能力强大到能自己维护的话，那也不是不可以。

5. 当你手指在滑动列表的时候，尽可能的不加载图片，这样的话滑动就会更加流畅。

## 后记


