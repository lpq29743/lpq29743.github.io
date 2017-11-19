---
layout: post
title: 自定义 ListView 的 Adapter
categories: Android
description: 自定义ListView的Adapter
keywords: Android, Android ListView, ListView, ListView Adapter
---

ListView在Android中的运用相当普遍。学习ListView，自定义适配器是永远无法跳过的知识点，今天就让我们一起来讲一讲ListView。

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

   要实现分页加载，就必须要用到OnScrollListener，它的主要内容：

   ```java
   // 静态属性
   public static int SCROLL_STATE_IDLE = 0; // 空闲状态
   public static int SCROLL_STATE_TOUCH_SCROLL = 1; // 滚动状态,并且手指在屏幕上
   public static int SCROLL_STATE_FLING = 2;    //    滚动状态,手指已经离开了屏幕

   // 抽象方法
   public void onScrollStateChanged(AbsListView view, int scrollState); 
   // ListView在状态改变的时候调用，用户在正常滑动的时候通常会执行三次(刻意滑动当listView停止的时候才将手离开屏幕执行两次)

   public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount); 
   // ListView在滚动的时候会反复调用该方法,调用次数和listView的子项无关(屏幕只要滑动一点就会调用)
   ```

   所以如果要分页加载的话，实现代码如下：

   ```java
   mNewsListView.setOnScrollListener(new OnScrollListener() {

        @Override
        public void onScrollStateChanged(AbsListView view, int scrollState) {
            if (scrollState == OnScrollListener.SCROLL_STATE_IDLE) {
                if (isBottom) {
                    // 下载更多数据
                    Toast.makeText(MainActivity.this, "正在加载",
                            Toast.LENGTH_SHORT).show();        
                    // 加载数据的方法代码.......
                    // 这里面的代码通常是根据mPageNum加载不同的数据
                    // 对mPageNum处理: mPageNum++

                }
            }
        }

        @Override
        public void onScroll(AbsListView view, int firstVisibleItem,
                int visibleItemCount, int totalItemCount) {
            if (firstVisibleItem + visibleItemCount == totalItemCount) {
                // 说明:
                // fistVisibleItem:表示划出屏幕的ListView子项个数
                // visibleItemCount:表示屏幕中正在显示的ListView子项个数
                // totalItemCount:表示ListView子项的总数    
                // 前两个相加==最后一个说明ListView滑到底部
                isButtom = true;
            }else{
                isButtom = false;
            }
        }
    });
   ```

4. 如果有图片的话，用第三方库来加载（也就是缓存）。另外，滑动列表时尽可能不加载图片。

其实，在这些年的开发中，ListView的使用频率越来越低，取而代之的是RecyclerView。RecyclerView是V7包下新增的控件，用来替代ListView，RecyclerView标准化了ViewHolder（类似于ListView中的convertView）用来做视图缓存。

RecyclerView可通过设置LayoutManager来快速实现Listview、Gridview、瀑布流的效果，还可以设置横向和纵向显示，添加动画也非常简单（自带了ItemAnimation，可以设置加载和移除时的动画，方便做出各种动态浏览的效果）。那么这个如此有用的官方推荐控件，又应该如何使用呢？

首先需要在gradle文件中添加包的引用（配合官方CardView使用）

```
compile 'com.android.support:cardview-v7:21.0.3'
compile 'com.android.support:recyclerview-v7:21.0.3'
```

然后在XML文件中使用它

```
<android.support.v7.widget.RecyclerView
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:id="@+id/recycler_view"
    android:layout_centerVertical="true"
    android:layout_centerHorizontal="true"/>
    
```

接着在Activity中进行设置

```
public class MainActivity extends ActionBarActivity {
    @InjectView(R.id.recycler_view)
    RecyclerView mRecyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.inject(this);

        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));//这里用线性显示 类似于listview
//        mRecyclerView.setLayoutManager(new GridLayoutManager(this, 2));//这里用线性宫格显示 类似于grid view
//        mRecyclerView.setLayoutManager(new StaggeredGridLayoutManager(2, OrientationHelper.VERTICAL));//这里用线性宫格显示 类似于瀑布流
        mRecyclerView.setAdapter(new NormalRecyclerViewAdapter(this));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
```

最后是自定义RecyclerView的适配器

```
public class NormalRecyclerViewAdapter extends RecyclerView.Adapter<NormalRecyclerViewAdapter.NormalTextViewHolder> {
    private final LayoutInflater mLayoutInflater;
    private final Context mContext;
    private String[] mTitles;

    public NormalRecyclerViewAdapter(Context context) {
        mTitles = context.getResources().getStringArray(R.array.titles);
        mContext = context;
        mLayoutInflater = LayoutInflater.from(context);
    }

    @Override
    public NormalTextViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        return new NormalTextViewHolder(mLayoutInflater.inflate(R.layout.item_text, parent, false));
    }

    @Override
    public void onBindViewHolder(NormalTextViewHolder holder, int position) {
        holder.mTextView.setText(mTitles[position]);
    }

    @Override
    public int getItemCount() {
        return mTitles == null ? 0 : mTitles.length;
    }

    public static class NormalTextViewHolder extends RecyclerView.ViewHolder {
        @InjectView(R.id.text_view)
        TextView mTextView;

        NormalTextViewHolder(View view) {
            super(view);
            ButterKnife.inject(this, view);
            view.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Log.d("NormalTextViewHolder", "onClick--> position = " + getPosition());
                }
            });
        }
    }
}
```

这样子，我们就实现了RecyclerView，那么对于已经使用了ListView的情况，我们有没有必要把它替换成RecyclerView呢？这个问题应该分情况进行讨论：

1. 如果需要支持动画、频繁更新（如弹幕）或者局部刷新，建议使用RecyclerView，更加强大完善。
2. 其它情况两者都OK，RecyclerView从性能上并没有带来显著的提升，但ListView在使用上会更加方便，快捷。