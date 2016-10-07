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

接下来我们就来一一讲解这几种技术！

#### 代码混淆技术

该技术主要是进行代码混淆，降低代码逆向编译后的可读性，但该技术无法防止加壳技术进行加壳（加入吸费、广告、病毒等代码），而且只要是细心的人，依然可以对代码依然可以对代码进行逆向分析，所以该技术并没有从根本解决破解问题，只是增加了破解难度。此文所提供的教程代码混淆只针对Android Studio

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

**第二步：根据ProGuard语法修改proguard-rules.pro文件**

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

**通配符匹配规则**

| 通配符        | 规则                     |
| ---------- | ---------------------- |
| ？          | 匹配单个字符                 |
| *          | 匹配类名中的任何部分，但不包含额外的包名   |
| **         | 匹配类名中的任何部分，并且可以包含额外的包名 |
| %          | 匹配任何基础类型的类型名           |
| ***        | 匹配任意类型名 ,包含基础类型/非基础类型  |
| ...        | 匹配任意数量、任意类型的参数         |
| <init>     | 匹配任何构造器                |
| <ifield>   | 匹配任何字段名                |
| <imethod>  | 匹配任何方法                 |
| *(当用在类内部时) | 匹配任何字段和方法              |
| $          | 指内部类                   |

> 更详细的语法请戳:[http://proguard.sourceforge.net/manual/usage.html#classspecification](http://proguard.sourceforge.net/manual/usage.html#classspecification)



 



一般情况，我们只需要修改proguard-rules.pro文件，把不需要混淆的部分在该文件中声明，因为有些类已经混淆过，。比如使用百度地图安卓sdk需要在proguard-rules.pro中加入下面代码：

| 123  | -keep class com.baidu.** {*;}-keep class vi.com.** {*;}-dontwarn com.baidu.** |
| ---- | ---------------------------------------- |
|      |                                          |

 

 

使用gson，fastjsoon时，bean类不需要混淆，否则会出错，为此还特意请教了大神，哈哈。
因为他们利用发射来解析json，混淆后找不到对应的变量导致空指针。

 

 

在android Manifest文件中的activity，service，provider， receiver，等都不能进行混淆。一些在xml中配置的view也不能进行混淆，当然，这些在sdk里默认配置文件proguard-android.txt中都有，就不用我们配置咯。

 

1 混淆之后，会给我们输出一些文件，android studio 在目录app/build/outputs/mapping下有以下文件：

dump.txt 描述apk文件中所有类文件间的内部结构。

mapping.txt 列出了原始的类，方法，和字段名与混淆后代码之间的映射，常用。

seeds.txt 列出了未被混淆的类和成员

usage.txt 列出了从apk中删除的代码

当我们发布的release版本的程序出现bug时，可以通过以上文件（特别是mapping.txt）找到错误原始的位置，进行bug修改。同时，可能一开始的proguard配置有错误，也可以通过错误日志，根据这些文件，找到哪些文件不应该混淆，从而修改proguard的配置。
注意：重新release编译后，这些文件会被覆盖，所以每发布一次程序，最好都保存一份配置文件。

2 通过mapping.txt,通过映射关系找到对应的类，方法，字段来修复bug。
这里需要利用sdk给我们提供的retrace脚本，可以将一个被混淆过的堆栈跟踪信息还原成一个可读的信息，window下时retrace.bat，linux和mac是retrace.sh，该脚本的位置在*/sdk/tools/proguard/bin/下。语法为：

| 1    | retrace.bat\|retrace.sh [-verbose] mapping.txt [<stacktrace_file>] |
| ---- | ---------------------------------------- |
|      |                                          |

例如：

| 1    | ./retrace.sh -verbose mapping.txt a.txt |
| ---- | --------------------------------------- |
|      |                                         |

其中的a.txt文件可以新建，然后把logcat的错误信息复制后粘贴到a.txt即可。

如果你没有指定，retrace工具会从标准输入(一般是键盘)读取。

这里有个小技巧：不用mapping也可以显示行号，避免Unknown Source

很简单，在混淆里加这么一句就可以：

| 1    | -keepattributes SourceFile,LineNumberTable |
| ---- | ---------------------------------------- |
|      |                                          |

这样，apk包会增大一些，我5.8M的包增加254K大小，还是可以接受的。

 

它主要保留了继承自Activity、Application、Service、BroadcastReceiver、ContentProvider、BackupAgentHelper、Preference和ILicensingService的子类。因为这些子类，都是可能被外部调用的。
另外，它还保留了含有native方法的类、构造函数从xml构造的类（一般为View的子类）、枚举类型中的values和valueOf静态方法、继承Parcelable的跨进程数据类。

## 后记

安全是计算机领域一个重要的版块，也希望通过这篇文章，让大家的开发安全意思有所提高。