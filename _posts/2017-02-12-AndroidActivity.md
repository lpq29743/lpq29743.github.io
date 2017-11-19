---
layout: post
title: Android 四大组件之 Activity
categories: Android
description: Android四大组件之Activity
keywords: Android, Android Activity
---

Activity用于显示控件以及监听并处理用户事件，在Android App中扮演着举重若轻的角色，今天将让我们一起来认识一下Android四大组件之一的Activity。

### Activity生命周期

Activity生命周期是每一个讲Activity的人都不可以跳过的内容，可以说是Activity这个知识点中及其重要的一部分。在做具体的实验之前，我们先将Android API上面的这张图弄上来：

![Activity生命周期](/redant/images/posts/android/activity-lifecycle.gif)

然后我们开始做实验。创建一个小项目，写上实验代码，就可以进行实验了！

```java
package com.activitydemo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private Button mDialogBtn, mActivityBtn;

    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mDialogBtn = (Button) findViewById(R.id.dialog_btn);
        mActivityBtn = (Button) findViewById(R.id.activity_btn);

        mDialogBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("test")
                        .setMessage("test dialog")
                        .setPositiveButton("confirm", null)
                        .show();
            }
        });
        mActivityBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, AssistActivity.class);
                startActivity(intent);
            }
        });

        Log.e(TAG, "Activity onCreate");
    }

    @Override
    protected void onStart() {
        super.onStart();
        Log.e(TAG, "Activity onStart");
    }

    @Override
    protected void onRestart() {
        super.onRestart();
        Log.e(TAG, "Activity onRestart");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.e(TAG, "Activity onResume");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.e(TAG, "Activity onPause");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.e(TAG, "Activity onStop");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.e(TAG, "Activity onDestroy");
    }

}
```

在有些实验中，还添加了onSaveInstanceState和onRestoreInstanceState方法：

- onSaveInstanceState

  在Activity被覆盖或退居后台之后，系统资源不足将其杀死，此方法会被调用；我们无法保证什么时候发生，系统根据资源紧张程度去调度。

  在用户改变屏幕方向时，此方法会被调用；屏幕翻转方向时，系统先销毁当前的Activity，然后再重建一个新的，调用此方法时，我们可以保存一些临时数据；

  在当前Activity跳转到其他Activity或者按Home键回到主屏，自身退居后台时，此方法会被调用。系统调用此方法是为了保存当前窗口各个View组件的状态。

- onRestoreInstanceState：

  在Activity被覆盖或退居后台之后，系统资源不足将其杀死，然后用户又回到了此Activity，此方法会被调用；

  在用户改变屏幕方向时，重建的过程中，此方法会被调用。我们可以重写此方法，以便可以恢复一些临时数据。

运行程序，打印日志为：

```
02-12 15:14:30.371 32163-32163/com.activitydemo E/MainActivity: Activity onCreate
02-12 15:14:30.371 32163-32163/com.activitydemo E/MainActivity: Activity onStart
02-12 15:14:30.371 32163-32163/com.activitydemo E/MainActivity: Activity onResume
```

按下Back键，即结束程序，打印日志为：

```
02-12 15:22:44.121 6598-6598/com.activitydemo E/MainActivity: Activity onPause
02-12 15:22:44.381 6598-6598/com.activitydemo E/MainActivity: Activity onStop
02-12 15:22:44.381 6598-6598/com.activitydemo E/MainActivity: Activity onDestroy
```

打开程序，按Home键，打印日志为：

```
02-12 15:24:18.911 6598-6598/com.activitydemo E/MainActivity: Activity onPause
02-12 15:24:19.141 6598-6598/com.activitydemo E/MainActivity: Activity onStop
```

再次启动程序，打印日志为：

```
02-12 15:25:21.261 6598-6598/com.activitydemo E/MainActivity: Activity onRestart
02-12 15:25:21.271 6598-6598/com.activitydemo E/MainActivity: Activity onStart
02-12 15:25:21.271 6598-6598/com.activitydemo E/MainActivity: Activity onResume
```

锁屏，打印日志为：

```
02-12 15:24:18.911 6598-6598/com.activitydemo E/MainActivity: Activity onPause
02-12 15:24:19.141 6598-6598/com.activitydemo E/MainActivity: Activity onStop
```

解锁，打印日志为：

```
02-12 15:25:21.261 6598-6598/com.activitydemo E/MainActivity: Activity onRestart
02-12 15:25:21.271 6598-6598/com.activitydemo E/MainActivity: Activity onStart
02-12 15:25:21.271 6598-6598/com.activitydemo E/MainActivity: Activity onResume
```

通过以上四次打印，可以得出，锁屏和回到桌面是相对应的，而进一步的实验，我们还可以得知菜单键也是同样的效果。

按第一个按钮弹出对话框，无打印日志。事实上，弹出对话框的时候，不会引起任何Activity的任何生命周期的变化，除了启动Theme为Dialog的Activity的这种情况，我们会稍后讨论。

按第二个按钮启动Activity，打印日志为：

```
02-12 17:57:52.974 24866-24866/com.activitydemo E/MainActivity: Activity onPause
02-12 17:57:53.004 24866-24866/com.activitydemo E/AssistActivity: Activity onCreate
02-12 17:57:53.004 24866-24866/com.activitydemo E/AssistActivity: Activity onStart
02-12 17:57:53.004 24866-24866/com.activitydemo E/AssistActivity: Activity onResume
02-12 17:57:53.294 24866-24866/com.activitydemo E/MainActivity: Activity onStop
```

返回原页面，打印日志为：

```
02-12 18:20:56.804 20891-20891/com.activitydemo E/AssistActivity: Activity onPause
02-12 18:20:56.824 20891-20891/com.activitydemo E/MainActivity: Activity onRestart
02-12 18:20:56.824 20891-20891/com.activitydemo E/MainActivity: Activity onStart
02-12 18:20:56.824 20891-20891/com.activitydemo E/MainActivity: Activity onResume
02-12 18:20:57.114 20891-20891/com.activitydemo E/AssistActivity: Activity onStop
02-12 18:20:57.114 20891-20891/com.activitydemo E/AssistActivity: Activity onDestroy
```

接下来我们在配置文件中对AssistActivity多配置一句`android:theme="@android:style/Theme.Dialog"`，即以Dialog形式启动AssistActivity，再启动App，然后点击启动AssistActivity，打印日志为：

```
02-12 18:11:59.934 11152-11152/com.activitydemo E/MainActivity: Activity onPause
02-12 18:11:59.964 11152-11152/com.activitydemo E/AssistActivity: Activity onCreate
02-12 18:11:59.964 11152-11152/com.activitydemo E/AssistActivity: Activity onStart
02-12 18:11:59.964 11152-11152/com.activitydemo E/AssistActivity: Activity onResume
```

这个时候，我们进行锁屏，打印日志为：

```
02-12 18:16:59.924 11152-11152/com.activitydemo E/AssistActivity: Activity onPause
02-12 18:16:59.934 11152-11152/com.activitydemo E/AssistActivity: Activity onStop
02-12 18:16:59.934 11152-11152/com.activitydemo E/MainActivity: Activity onStop
```

解锁，打印日志为：

```
02-12 18:18:29.614 11152-11152/com.activitydemo E/AssistActivity: Activity onRestart
02-12 18:18:29.644 11152-11152/com.activitydemo E/AssistActivity: Activity onStart
02-12 18:18:29.654 11152-11152/com.activitydemo E/MainActivity: Activity onRestart
02-12 18:18:29.664 11152-11152/com.activitydemo E/MainActivity: Activity onStart
02-12 18:18:29.664 11152-11152/com.activitydemo E/AssistActivity: Activity onResume
```

返回原Activity，打印日志为：

```
02-12 18:19:00.754 11152-11152/com.activitydemo E/AssistActivity: Activity onPause
02-12 18:19:00.784 11152-11152/com.activitydemo E/MainActivity: Activity onResume
02-12 18:19:00.814 11152-11152/com.activitydemo E/AssistActivity: Activity onStop
02-12 18:19:00.814 11152-11152/com.activitydemo E/AssistActivity: Activity onDestroy
```

通过以上的实验，我们也可以做一个简短的总结

> 在Activity创建的时候，会执行onCreate->onStart->onResume
>
> 在我们进入Activity之后按返回键，会执行onPause->onStop->onDestory
>
> 在我们进入Activity之后按Home键，会执行onPause->onStop
>
> 这个时候当我们又回到Activity，会执行onRestart->onStart->onResume
>
> 当从A1界面跳到A2界面，生命周期变化为：onCreate(A1)->onStart(A1)->onResume(A1)->onPause(A1)->onCreate(A2)->onStart(A2)->onResume(A2)->onStop(A1)
>
> 此时如果在A2界面按下返回键，生命周期会的变化如下：onPause(A2)->onRestart(A1)->onStart(A1)->onResume(A1)->onStop(A2)->onDestory(A2)
>
> 注：如果A2界面是一个透明主题（如之前提到的Dialog，如果要自己创建一个透明主题的Activity，参考[这里](http://stackoverflow.com/questions/2176922/how-to-create-transparent-activity-in-android)）的话，那么A1不会调用onStop方法

### Activity堆栈管理和控制

在Android系统中，即使有多个Activity分别来自不同应用程序，Android系统仍然可以将它们无缝结合到一起。之所以能实现这一点，是因为这些Activity都是存在于一个相同的任务(Task)当中的。任务是一个Activity的集合，它使用栈的方式来管理其中的Activity，这个栈又被称为返回栈(back stack)，栈中Activity的顺序就是按照它们被打开的顺序依次存放的。

当用户在Home界面上点击了一个应用，这个应用的任务就会被转移到前台。如果这个应用目前没有任何一个任务的话，系统就创建新的任务，并且将该应用的主Activity放到返回栈当中。当一个Activity启动了另外一个Activity时，新的Activity就会被放置到栈顶并将获得焦点。前一个Activity仍然保留在返回栈当中，但会处于停止状态。当用户按下Back键的时候，栈中最顶端的Activity会被移除掉，然后前一个Activity则会得重新回到最顶端的位置。如果用户一直地按Back键，这样返回栈中的Activity会一个个地被移除，直到最终返回到主屏幕。当返回栈中所有的Activity都被移除掉的时候，对应的任务也就不存在了。

任务除了可以被转移到前台，也是可以被转移到后台的。当用户开启了新的任务，或者点击Home键回到主屏幕时，之前任务就会被转移到后台了。举个例子，当前任务A的栈中有三个Activity，现在用户按下Home键，然后启动另外一个应用程序。当系统回到桌面的时候，其实任务A就已经进入后台了，然后当另外一个应用程序启动的时候，系统会为这个程序开启一个新的任务(任务B)。当用户再次按下Home键回到桌面，这时任务B也进入了后台。然后用户重新打开第一次使用的程序，这个时候任务A又会回到前台，A任务栈中的三个Activity仍然会保留着刚才的顺序，最顶端的Activity将重新变为运行状态。这就是Android中多任务切换的例子。

这个时候，用户还可以将任意后台的任务切换到前台，这样用户应该就会看到之前离开这个任务时处于最顶端的那个Activity。举个例子来说，当前任务A的栈中有三个Activity，现在用户按下Home键，然后点击桌面上的图标启动了另外一个应用程序。当系统回到桌面的时候，其实任务A就已经进入后台了，然后当另外一个应用程序启动的时候，系统会为这个程序开启一个新的任务(任务B)。当用户使用完这个程序之后，再次按下Home键回到桌面，这个时候任务B也进入了后台。然后用户又重新打开了第一次使用的程序，这个时候任务A又会回到前台，A任务栈中的三个Activity仍然会保留着刚才的顺序，最顶端的Activity将重新变为运行状态。之后用户仍然可以通过Home键或者多任务键来切换回任务B，或者启动更多的任务，这就是Android中多任务切换的例子。

由于返回栈中的Activity的顺序永远都不会发生改变，所以如果你的应用程序中允许有多个入口都可以启动同一个Activity，那么每次启动的时候就都会创建该Activity的一个新的实例，而不是将下面的Activity的移动到栈顶。这样的话就容易导致一个问题的产生，即同一个Activity可能被实例化很多次。如果我们不希望同一个Activity被多次实例化，那么我们应该怎么做呢？

Android系统管理任务和返回栈的方式，正如上面所描述的一样。如果想打破这种默认的行为，比如说当启动新的Activity时，希望它可以存在一个独立的任务当中。或者当启动一个Activity时，如果这个Activity已经存在于返回栈中，我们能把它直接移到栈顶，而不是创建新的实例。再比如清除掉返回栈中除了最底层的Activity之外的其它所有Activity。这些都是可以通过配置manifest文件中的activity元素的属性，或者是在启动Activity时配置Intent的flag来实现的。

**定义启动模式**

启动模式定义如何将Activity实例和当前任务进行关联，可通过manifest文件和在Intent中加入flag两种方式来定义。也就是说，如果Activity A启动Activity B，Activity B可以定义自己如何与当前任务进行关联，而Activity A也可以要求Activity B该如何与当前任务进行关联。如果Activity B在manifest中已经定义了如何与任务进行关联，而Activity A同时也在Intent中要求了Activity B如何与当前任务进行关联，那么此时Intent中的定义将覆盖manifest中的定义。需要注意的是，有些启动模式在manifest中可以指定，但在Intent中不行，反之也一样。

当在manifest文件中定义Activity时，可通过activity元素的launchMode属性指定如何与任务进行关联。launchMode属性一共有以下四种可选参数：

- "standard"(默认启动模式)：默认的启动模式。每次启动该Activity时都会创建新的实例并放入当前任务。声明成这种启动模式的Activity可以被实例化多次，一个任务当中也可以包含多个这种Activity的实例。
- "singleTop"：如果Activity已经存在且处于栈顶位置，那么系统不会再创建实例，而是调用栈顶Activity的onNewIntent()方法。声明成这种启动模式的Activity也可以被实例化多次，一个任务当中也可以包含多个这种Activity的实例。
- "singleTask"：系统会创建新的任务，并将启动的Activity放入新任务栈底。但如果现有任务已经存在该Activity的实例，那么系统不会再创建一次实例，而是调用它的onNewIntent()方法。声明成这种启动模式的Activity，在同一个任务当中只会存在一个实例。注意这里的启动Activity，都指的是启动其它应用程序中的Activity，因为"singleTask"模式在默认情况下只有启动其它程序的Activity才会创建新的任务，启动自己程序中的Activity还是会使用相同的任务。
- "singleInstance"：这种启动模式和"singleTask"有点相似，只不过系统不会向声明成"singleInstance"的Activity所在的任务当中再添加其它Activity。通过这个Activity再打开的其它Activity会被放到别的任务中。Android内置浏览器就是一个例子。当程序准备打开Android内置浏览器的时候，新打开的Activity并不会放入当前任务中，而是启动一个新的任务。而如果浏览器程序在后台已经存在一个任务了，则会把这个任务切换到前台。

不管Activity在新任务钟启动，还是在当前任务中启动，返回键都是返回到上一个Activity。但有种情况比较特殊，就是如果Activity的启动模式是"singleTask"，并且启动的是另外一个应用程序中的Activity，这个时候当发现该Activity正好处于一个后台任务中的话，就会直接将这整个后台任务切换到前台。此时按下返回键会优先将目前最前台的任务进行回退。

除了使用manifest文件之外，还可以在调用startActivity()方法的时候，为Intent加入flag来改变关联方式，具体如下：

- FLAG_ACTIVITY_NEW_TASK：Activity被放置到新任务中（与"singleTask"类似，但不完全一样），这里还是启动其它程序中的Activity。这个flag通常是模拟Launcher，列出可启动的东西，但启动的Activity都是运行在独立任务中。
- FLAG_ACTIVITY_SINGLE_TOP：如果启动Activity在当前任务中已存在且还处于栈顶，那么就不会再次创建这个实例，而是调用它的onNewIntent()方法。这种flag和在launchMode中指定"singleTop"模式所实现的效果是一样的。
- FLAG_ACTIVITY_CLEAR_TOP：如果启动Activity在当前任务中已存在，就不会创建实例，而是把该Activity上的所有Activity全关掉。如任务中有ABCD，D调用了startActivity()来启动B，flag指定成FLAG_ACTIVITY_CLEAR_TOP，此时C和D就会被关闭掉，返回栈中只剩下A和B。

那么此时Activity B会接收到这个启动它的Intent，你可以决定是让Activity B调用onNewIntent()方法(不会创建新的实例)，还是将Activity B销毁掉并重新创建实例。如果Activity B没有在manifest中指定任何启动模式(也就是"standard"模式)，并且Intent中也没有加入一个FLAG_ACTIVITY_SINGLE_TOP flag，那么此时Activity B就会销毁掉，然后重新创建实例。而如果Activity B在manifest中指定了任何一种启动模式，或者是在Intent中加入了一个FLAG_ACTIVITY_SINGLE_TOP flag，那么就会调用Activity B的onNewIntent()方法。

FLAG_ACTIVITY_CLEAR_TOP和FLAG_ACTIVITY_NEW_TASK结合在一起使用也会有比较好的效果，比如可以将一个后台运行的任务切换到前台，并把目标Activity之上的其它Activity全部关闭掉。这个功能在某些情况下非常有用，比如说从通知栏启动Activity的时候。

**处理affinity**

affinity指定Activity依附于哪个任务，同一应用程序中所有Activity默认具有相同的affinity，但修改activity元素的taskAffinity属性可以改变Activity的affinity值。taskAffinity属性接收一个字符串参数，可以指定成任意的值（字符串中至少要包含一个.），但不能和应用程序包名相同，因为系统会使用包名来作为默认的affinity值。affinity主要应用于以下两种应用场景：

- 如果在Intent中加入了FLAG_ACTIVITY_NEW_TASK flag（或声明启动模式是"singleTask"），系统就会尝试为启动Activity创建新的任务。但规则并不是这么简单，系统会检测这个Activity的affinity和当前任务的affinity是否相同，如果相同就会把它放入到现有任务，如果不同则会去创建新的任务。而同一个程序中所有Activity的affinity默认都是相同的，这也是为什么说，同一个应用程序中即使声明成"singleTask"，也不会为这个Activity再创建新的任务。
- 当Activity的allowTaskReparenting属性为true时，Activity就有转移所在任务的能力。具体说就是一个Activity现在是处于某个任务当中的，但是它与另外一个任务具有相同的affinity值，当另外这个任务切换到前台时，该Activity就可以转移到现在的这个任务当中。比如有个天气预报程序，它有个Activity是用于显示天气信息的，这个Activity和天气预报程序的所有Activity有相同的affinity值，并且将allowTaskReparenting属性设置成true了。这时你的应用程序启动了这个Activity，那么这个Activity是和你的应用程序是在同一个任务中的。但当把天气预报程序切换到前台时，这个Activity又会被转移到天气预报程序的任务中并显示出来。

**清空返回栈**

如果任务切换到后台太久，系统会将这个任务中除了最底层的Activity之外的其它Activity全清除掉。当用户重新回到任务时，最底层的Activity将恢复。这是系统默认的行为，但我们可以设置activity元素中的几种属性来改变这一默认行为：

- alwaysRetainTaskState：如果将最底层的Activity的这个属性设置为true，那么默认行为就不会发生。
- clearTaskOnLaunch：如果将最底层的Activity的这个属性设置为true，那么只要用户离开了当前任务，再次返回时就会将最底层Activity上的其它Activity全清除掉。
- finishOnTaskLaunch：如果Activity这个属性为true，那么用户一旦离开当前任务，再次返回时此Activity就会被清除。

### Fragment生命周期

Fragment常用于平板开发和Tab切换，掌握它的生命周期也是相当重要的，首先我们先看两个图（前者是Fragment生命周期图，后者是Fragment与Activity生命周期对比图）：

![Fragment生命周期](/redant/images/posts/android/fragment-lifecycle.png)

![Fragment与Activity生命周期对比图](/redant/images/posts/android/lifecycle-comparison.png)

这里我们就不继续做实验了，从上面两个图我们可以得到：

- fragment创建时，会执行onAttach()->onCreate()->onCreateView()->onActivityCreated()
- fragment对用户可见时，会执行onStart()->onResume()
- fragment进入“后台模式”时，会执行onPause()->onStop()
- fragment被销毁，会执行onPause()->onStop()->onDestroyView()->onDestroy()->onDetach()
- fragment与activity相比，有一些新的状态：onAttached()（当fragment加入到activity时调用，在这个方法中可获得所在的activity）、onCreateView() （当activity要得到fragment的layout时，调用此方法，fragment在其中创建自己的layout）、onActivityCreated()（当activity的onCreated()方法返回后调用此方法）、onDestroyView() （当fragment中的视图被移除时，调用这个方法）和onDetach() （当fragment和activity分离的时候，调用这个方法）