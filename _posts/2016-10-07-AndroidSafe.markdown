---
layout:     post
title:      "Android防破解总结"
subtitle:   "誓死保卫app的安全"
date:       2016-10-07 13:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - Android
---

> “别想打我开发的app的主意”


## 前言

上一篇文章我们讲解了如何通过apk文件查看Android应用的源码。通过该文章，我们也认识到了Android开发安全的重要性。那么，这篇文章来讲一讲如何防止Android应用被破解的方法。

---

## 正文

防破解技术主要有四种实现方式：

1. 代码混淆（ProGuard）技术
2. 签名比对技术
3. NDK  .so 动态库技术
4. 动态加载技术

#### 代码混淆技术

该技术主要是进行代码混淆，降低代码逆向编译后的可读性，但该技术无法防止加壳技术进行加壳（加入吸费、广告、病毒等代码），而且只要是细心的人，依然可以对代码依然可以对代码进行逆向分析，所以该技术并没有从根本解决破解问题，只是增加了破解难度。此文所提供的教程代码混淆只针对Android Studio。

**第一步：配置build.gradle**

```shell
buildTypes {
    debug {
        minifyEnabled false
    }
    release {
        signingConfig signingConfigs.release
        minifyEnabled true
        proguardFiles 'proguard-rules.pro'
    }
}
```

**第二步：混淆注意事项**

一般情况，我们只需要修改proguard-rules.pro文件，把不需要混淆的部分在该文件中声明。不需混淆的部分如下：

- 在AndroidManifest中配置的类，比如四大组件
- JNI调用的方法
- 反射用到的类，如Gson，Fastjson时，Bean类不需要混淆
- WebView中JavaScript调用的方法
- Layout文件引用到的自定义View
- 一些引入的第三方库，如百度地图等
- 枚举不需混淆

**第三步：根据ProGuard语法修改proguard-rules.pro文件**

***保留***

- -keep {Modifier} {class_specification} 保护指定的类文件和类的成员
- -keepclassmembers {modifier} {class_specification} 保护指定类的成员，如果此类受到保护他们会保护的更好
- -keepclasseswithmembers {class_specification} 保护指定的类和类的成员，但条件是所有指定的类和类成员是要存在。
- -keepnames {class_specification} 保护指定的类和类的成员的名称（如果他们不会压缩步骤中删除）
- -keepclassmembernames {class_specification} 保护指定的类的成员的名称（如果他们不会压缩步骤中删除）
- -keepclasseswithmembernames {class_specification} 保护指定的类和类的成员的名称，如果所有指定的类成员出席（在压缩步骤之后）
- -printseeds {filename} 列出类和类的成员-keep选项的清单，标准输出到给定的文件

***压缩***

- -dontshrink 不压缩输入的类文件
- -printusage {filename}
- -whyareyoukeeping {class_specification}

***优化***

- -dontoptimize 不优化输入的类文件
- -assumenosideeffects {class_specification} 优化时假设指定的方法，没有任何副作用
- -allowaccessmodification 优化时允许访问并修改有修饰符的类和类的成员

***混淆***

- -dontobfuscate 不混淆输入的类文件
- -obfuscationdictionary {filename} 使用给定文件中的关键字作为要混淆方法的名称
- -overloadaggressively 混淆时应用侵入式重载
- -useuniqueclassmembernames 确定统一的混淆类的成员名称来增加混淆
- -flattenpackagehierarchy {package_name} 重新包装所有重命名的包并放在给定的单一包中
- -repackageclass {package_name} 重新包装所有重命名的类文件中放在给定的单一包中
- -dontusemixedcaseclassnames 混淆时不会产生形形色色的类名
- -keepattributes {attribute_name,…} 保护给定的可选属性，例如LineNumberTable, LocalVariableTable, SourceFile, Deprecated, Synthetic, Signature, and InnerClasses.
- -renamesourcefileattribute {string} 设置源文件中给定的字符串常量

***通配符匹配规则***

- ?：匹配单个字符
- *：匹配类名中的任何部分，但不包含额外的包名
- **：匹配类名中的任何部分，并且可以包含额外的包名
- %：匹配任何基础类型的类型名
- ***：匹配任意类型名 ,包含基础类型/非基础类型
- ...：匹配任意数量、任意类型的参数
- <init>：匹配任何构造器
- <ifield>：匹配任何字段名
- <imethod>：匹配任何方法
- *(当用在类内部时)：匹配任何字段和方法
- $：指内部类

> 更详细的语法请戳:[http://proguard.sourceforge.net/manual/usage.html#classspecification](http://proguard.sourceforge.net/manual/usage.html#classspecification)

**第四步：根据例子进行修改proguard-rules.pro文件**

 查看语法可能会让基础不好的朋友一头雾水，接下来给出例子，大家可根据例子进行修改：

```shell
################common###############
-keep class com.jph.android.entity.** { *; } #实体类不参与混淆
-keep class com.jph.android.view.** { *; } #自定义控件不参与混淆

################baidu map###############
-libraryjars libs/baidumapapi_v3_2_0.jar
-libraryjars libs/locSDK_5.0.jar
-keep class com.baidu.** { *; }
-keep class vi.com.gdi.bgl.android.**{*;}
-dontwarn com.baidu.**

################afinal##################
#-libraryjars libs/afinal_0.5_bin.jar
#-keep class net.tsz.afinal.** { *; } 
#-keep public class * extends net.tsz.afinal.**  
#-keep public interface net.tsz.afinal.** {*;}
#-dontwarn net.tsz.afinal.**

################xutils##################
-libraryjars libs/xUtils-2.6.14.jar
-keep class com.lidroid.xutils.** { *; } 
-keep public class * extends com.lidroid.xutils.**  
-keepattributes Signature
-keepattributes *Annotation*
-keep public interface com.lidroid.xutils.** {*;}
-dontwarn com.lidroid.xutils.**
-keepclasseswithmembers class com.jph.android.entity.** {
	<fields>;
	<methods>;
}

################支付宝##################
-libraryjars libs/alipaysecsdk.jar
-libraryjars libs/alipayutdid.jar
-libraryjars libs/alipaysdk.jar
-keep class com.alipay.android.app.IAliPay{*;}
-keep class com.alipay.android.app.IAlixPay{*;}
-keep class com.alipay.android.app.IRemoteServiceCallback{*;}
-keep class com.alipay.android.app.lib.ResourceMap{*;}

################gson##################
-libraryjars libs/gson-2.2.4.jar
-keep class com.google.gson.** {*;}
#-keep class com.google.**{*;}
-keep class sun.misc.Unsafe { *; }
-keep class com.google.gson.stream.** { *; }
-keep class com.google.gson.examples.android.model.** { *; } 
-keep class com.google.** {
    <fields>;
    <methods>;
}
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}
-dontwarn com.google.gson.**

################httpmime/httpcore##########
-libraryjars libs/httpcore-4.3.2.jar
-libraryjars libs/httpmime-4.3.5.jar
-keep class org.apache.http.** {*;}
-dontwarn org.apache.http.**

####################jpush##################
-libraryjars libs/jpush-sdk-release1.7.1.jar
-keep class cn.jpush.** { *; }
-keep public class com.umeng.fb.ui.ThreadView { } #双向反馈功能代码不混淆
-dontwarn cn.jpush.**
-keepclassmembers class * {
    public <init>(org.json.JSONObject);
}
 #不混淆R类
-keep public class com.jph.android.R$*{ 
    public static final int *;
}
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

####################umeng##################
-libraryjars libs/umeng-analytics-v5.2.4.jar
-keep class com.umeng.analytics.** {*;}
-dontwarn com.umeng.analytics.**

#-keep public class * extends com.umeng.**  
#-keep public class * extends com.umeng.analytics.**  
#-keep public class * extends com.umeng.common.**  
#-keep public class * extends com.umeng.newxp.** 
-keep class com.umeng.** { *; }  
-keep class com.umeng.analytics.** { *; }  
-keep class com.umeng.common.** { *; }  
-keep class com.umeng.newxp.** { *; } 

-keepclassmembers class * {
   public <init>(org.json.JSONObject);
}
-keep class com.umeng.**

-keep public class com.idea.fifaalarmclock.app.R$*{
    public static final int *;
}

-keep public class com.umeng.fb.ui.ThreadView {
}

-dontwarn com.umeng.**

-dontwarn org.apache.commons.**

-keep public class * extends com.umeng.**

-keep class com.umeng.** {*; }

####################universal-image-loader########
-libraryjars libs/universal-image-loader-1.9.3.jar
-keep class com.nostra13.universalimageloader.** {*;}
-dontwarn com.nostra13.universalimageloader.**

####################zxing#####################
-libraryjars libs/zxing.jar
-libraryjars libs/zxing_apply.jar
-keep class com.google.zxing.** {*;}
-dontwarn com.google.zxing.**

####################BASE64Decoder##################
-libraryjars libs/sun.misc.BASE64Decoder.jar

####################support.v4#####################
-libraryjars libs/android-support-v4.jar
-keep class android.support.v4.** { *; }
-dontwarn android.support.v4.**

###################other####################
# slidingmenu 的混淆
-dontwarn com.jeremyfeinstein.slidingmenu.lib.**
-keep class com.jeremyfeinstein.slidingmenu.lib.** { *; }
# ActionBarSherlock混淆
-dontwarn com.actionbarsherlock.**
-keep class com.actionbarsherlock.** { *; }
-keep interface com.actionbarsherlock.** { *; }
-keep class * extends java.lang.annotation.Annotation { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

-keep class com.jph.android.entity.** {
    <fields>;
    <methods>;
}

-dontwarn android.support.**
-dontwarn com.slidingmenu.lib.app.SlidingMapActivity
-keep class android.support.** { *; }
-keep class com.actionbarsherlock.** { *; }
-keep interface com.actionbarsherlock.** { *; }
-keep class com.slidingmenu.** { *; }
-keep interface com.slidingmenu.** { *; }
```

**第五步：输出文件说明**

混淆之后，会给我们输出一些文件，android studio 在目录/build/proguard/下有以下文件：

- dump.txt 描述apk文件中所有类文件间的内部结构

- mapping.txt 列出了原始的类，方法，和字段名与混淆后代码之间的映射

- seeds.txt 列出了未被混淆的类和成员

- usage.txt 列出了从apk中删除的代码


当我们发布的release版本的程序出现bug时，可以通过以上文件（特别是mapping.txt）找到错误原始的位置，进行bug修改。同时，可能一开始的proguard配置有错误，也可以通过错误日志，根据这些文件，找到哪些文件不应该混淆，从而修改proguard的配置。

sdk\tools\proguard\bin 目录下有个retrace工具可以将混淆后的报错堆栈解码成正常的类名window下为retrace.bat，linux和mac为retrace.sh，使用方法如下：

1. 将crash log保存为yourfilename.txt
2. 拿到版本发布时生成的mapping.txt
3. 执行命令retrace.bat -verbose mapping.txt yourfilename.txt

值得注意的是，重新release编译后，这些文件会被覆盖，所以每发布一次程序，都要保存一份配置文件。不过，可以通过配置gradle进行自动保存，具体方法如下：

```shell
android {
applicationVariants.all { variant ->
        variant.outputs.each { output ->
            if (variant.getBuildType().isMinifyEnabled()) {
                variant.assemble.doLast{
                        copy {
                            from variant.mappingFile
                            into "${projectDir}/mappings"
                            rename { String fileName ->
                                "mapping-${variant.name}.txt"
                            }
                        }
                }
            }
        }
        ......
    }
}
```

#### 签名比对技术

#### NDK  .so 动态库技术

#### 动态加载技术

## 后记

安全是计算机领域一个重要的版块，也希望通过这篇文章，让大家的开发安全意识有所提高。