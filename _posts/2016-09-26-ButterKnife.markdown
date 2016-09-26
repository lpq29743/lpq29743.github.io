---
layout:     post
title:      "Android依赖注入框架ButterKnife"
subtitle:   "帮你偷懒的ButterKnife"
date:       2016-09-26 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
    - 开源框架
---

> “不会偷懒的程序员不是好的程序员！”
>


## 前言

依赖注入框架在Android项目开发过程中极为方便，本位主要围绕这ButterKnife进行讲述。

---

## 正文

本文所介绍的版本是ButterKnife8.0.1（2016/04/29更新的版本），ButterKnife项目由Jake Wharton编写，项目源地址为：[https://github.com/JakeWharton/butterknife](https://github.com/JakeWharton/butterknife)，相关博客文章地址为：[http://jakewharton.github.io/butterknife/](http://jakewharton.github.io/butterknife/)。ButterKnife的原理只是把我们原先写的代码封装起来，所以在性能方面不会受到任何影响。

###配置###

***步骤一***

Project 的 build.gradle 添加：

```java
dependencies {
  classpath 'com.neenbedankt.gradle.plugins:android-apt:1.8'
}
```

***步骤二***

App 的 build.gradle 添加：

```java
apply plugin: 'com.neenbedankt.android-apt'

dependencies {
  compile 'com.jakewharton:butterknife:8.0.1'
  apt 'com.jakewharton:butterknife-compiler:8.0.1'
}
```

（这里要加入apt插件的原因是：apt是用于编译时扫描和解析 Java 注解的工具，通过它我们可以自己定义注解，并定义解析器来处理它们。它的原理是读入 Java 源代码，解析注解，然后生成新的 Java 代码，新生成的代码最后被编译成 Java 字节码。）

***步骤三***

安装Android Butterknife Zelezny（一般步骤一二执行完后会添加，没添加的话File-->Settings-->Plugins-->Browse repositories添加），这个插件可以让你在添加Butterkinfe注解时偷偷懒，直接点击几下鼠标既可以完成注解的增加，同时还是图形化的操作。

### 使用###

1. @BindView 来消除 findViewById

   使用前：

   ```java
   mTextView = (TextView) findViewById(R.id.text);
   ```

   使用后：

   ```java
   /**单个View控件的绑定*/
   @BindView(R.id.btn_login)
   /**多个控件的绑定可以写在List或者Array中*/
   @BindViews({ R.id.first_name, R.id.middle_name, R.id.last_name })
   List<EditText> nameViews;
   ```

   再具体点：

   ```java
   class ExampleActivity extends Activity {
     // 声明注解
     @BindView(R.id.title) TextView title;
     @BindView(R.id.subtitle) TextView subtitle;
     @BindView(R.id.footer) TextView footer;

     @Override public void onCreate(Bundle savedInstanceState) {
       super.onCreate(savedInstanceState);
       setContentView(R.layout.simple_activity);
       // 进行绑定
       ButterKnife.bind(this);
       // TODO Use fields...
     }
   }
   ```

   在Fragment中有所区别，但不是很大：

   ```java
   @BindView(R.id.me_about_us) LinearLayout meAboutUs;

   @Override
   public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
   	View view = inflater.inflate(R.layout.nav_me,container, false);
   	ButterKnife.bind(this, view);
   	return view;
   }

   // onDestroyView()中解绑
   private Unbinder unbinder = ButterKnife.bind(this, view);

   @Override
   public void onDestroyView() {
   	unbinder.unbind();
   	super.onDestroyView();
   }
   ```

   而在ViewHolder中是这样的：

   ```java
   @Override
   public View getView(int position, View convertView, ViewGroup parent) {
   	ViewHolder viewHolder = null;
   	if (convertView == null) {
   		convertView = LayoutInflater.from(mContext).inflate(R.layout.common_laguage_child_item, null);
   		viewHolder = new ViewHolder(convertView);
   		convertView.setTag(viewHolder);
   	} else {
   		viewHolder=(ViewHolder)convertView.getTag();
   	}
   	JSONObject child =getItem(position);
   	viewHolder.tvChildCommon.setText(child.optString("content",""));
   	return convertView;
   }

   class ViewHolder {
   	@BindView(R.id.tv_child_common)
   	TextView tvChildCommon;
   	ViewHolder(View view) {
   		ButterKnife.bind(this, view);
   	}
   }
   ```

2. @OnClick消除setOnClickListener

   使用前：

   ```java
   button.setOnClickListener(this);
   ```

   使用后：

   ```java
   @OnClick({R.id.bt_1, R.id.bt_2, R.id.bt_3})
   public void buttonClick(View v) {
       Toast.makeText(this, "view:" + v, Toast.LENGTH_SHORT).show();
   }
   ```

   监听器的参数是可选的：

   ```
   @OnClick(R.id.submit)
   public void sayHi(Button button) { 
       button.setText("Hello!");
   }
   ```

3. 资源绑定

   ```java
   @BindString(R.string.title) String title; 
   @BindDrawable(R.drawable.graphic) Drawable graphic; 
   @BindColor(R.color.red) int red; // int or ColorStateList field 
   @BindDimen(R.dimen.spacer) Float spacer; // int (for pixel size) or float (for exact value) field
   ```

4. action功能，批量操作view

   ```java
   @BindViews({R.id.bt_1, R.id.bt_2, R.id.bt_3})
   List<Button> buttons;

   static final ButterKnife.Action<Button> disable=new ButterKnife.Action<Button>() {
       @Override
       public void apply(@NonNull Button view, int index) {
           view.setEnabled(false);
       }
   };
   static final ButterKnife.Action<Button> enable=new ButterKnife.Action<Button>() {
       @Override
       public void apply(@NonNull Button view, int index) {
           view.setEnabled(true);
       }
   };

   // 使用
   ButterKnife.apply(buttons,disable);
   // ButterKnife.apply(buttons,enable);
   ```

### 注意###

1. Activity ButterKnife.bind(this);必须在setContentView();之后，且父类bind绑定后，子类不需要再bind
2. ButterKnife不能再你的library module中使用哦!!这是因为你的library中的R字段的id值不是final类型的，但是你自己的应用module中确是final类型的。


## 后记

做第一个Android项目的时候就会经常听到依赖注入这个名词，后面接触Java Web也常有耳闻。于是，后面做Android项目的时候就会经常考虑使用依赖注入框架。而ButterKnife作为Android依赖注入框架的代表，是每一个想要“偷懒”的程序员必须掌握的。
