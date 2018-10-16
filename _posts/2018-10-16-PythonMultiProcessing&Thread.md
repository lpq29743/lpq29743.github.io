---
layout: post
title: Python 多进程和多线程
categories: Python
description: Python多进程和多线程
keywords: Python, 进程, 线程, 多进程, 多线程
---

### Python 实现多进程的方式

在 Unix/Linux 系统用中，调用`fork()`就可以创建一个子进程，其向子进程返回 0，向父进程返回子进程的 ID。如果子进程要获取父进程的 ID，只需要调用`os.getppid()`。

```python
import os

print('Process %s start' % os.getpid())
pid = os.fork()
if pid == 0:
    print('I am child process %s and my parent is %s' % (os.getpid(), os.getppid()))
else:
    print('I am parent process %s and my kid is %s' % (os.getpid(), pid))
```

由于 Windows 不存在 `fork()`，而且多进程管理十分繁琐，所以我们可以考虑用 python 的`multiprocessing`包来处理多进程问题。在`multiprocessing`中，每一个进程都用一个`Process`类来表示，其 API 如下：

```python
# group 分组，实际上不使用
# target 表示调用对象，你可以传入方法的名字
# name 是别名，相当于给这个进程取一个名字
# args 表示被调用对象的位置参数元组，如 target 是函数 a，他有两个参数 m，n，那么 args 就传入(m, n)
# kwargs 表示调用对象的字典
# daemon 为 True 时，则主线程不必等待子进程，主线程结束则所有结束
multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
```

接下来我们先来看一个简单的例子：

```python
from multiprocessing import Process
import os


def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child Process will start.')
    # 启动进程
    p.start()
    # 进程同步，所以要等子进程结束后才打印结束语句。如果进行注释，则为异步操作，打印语句和函数调用会同时执行
    p.join()
    print('Child Process end.')
```

接下来可以尝试把同步过程去掉，创建多个进程，并尝试用`cpu_count()`和`active_children()`分别查看当前机器的 CPU 核心数量以及目前正在运行的进程。具体如下：

```python
import multiprocessing
from multiprocessing import Process
import os
import time


def run_proc(name, num):
    time.sleep(num)
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    for i in range(5):
        p = Process(target=run_proc, args=('test' + str(i), i))
        print('Child Process will start.')
        p.start()
    print('CPU number: %s' % multiprocessing.cpu_count())
    for p in multiprocessing.active_children():
        print('Child process name:' + p.name + ' id:' + str(p.pid))
```

类`Process`还具有以下方法：

```python
# 判断进程是否活跃
is_alive()
# 结束进程
terminate()
```

还可以以类的形式来创建进程：

```python
import multiprocessing


class MyProcess(multiprocessing.Process):
    def __init__(self, i):
        multiprocessing.Process.__init__(self)
        self.i = i

    def run(self):
        # Do something
        print(self.i)

if __name__ == '__main__':
    for i in range(10):
        p = MyProcess(i)
        p.start()
```

也可以进程池的方式批量创建子进程：

```python
from multiprocessing import Pool


def run_proc(name):
    print(name)

if __name__ == '__main__':
    # 最多同时执行四个进程。若不指定参数，则默认为 CPU 核心数量。此参数没有限制
    p = Pool(4)
    for i in range(5):
        p.apply_async(run_proc, args=(i,))
    print('Waiting for all subprocesses done...')
    # 关闭进程池，从而无法继续添加新的进程
    p.close()
    # 同步执行进程
    p.join()
    print('All subprocesses done.')
```

当使用多进程时，经常会出现在某一段时间时，资源只能由一个进程访问，其他进程只能等待，这种情况叫做“互斥”。为了实现这种需求，我们使用了锁机制。

```python
from multiprocessing import Process, Lock


def printer(item, lock):
    lock.acquire()
    try:
        print(item)
    finally:
        lock.release()

if __name__ == '__main__':
    lock = Lock()
    items = ['chinese', 'english', 'spanish', 'japanese']
    for item in items:
        p = Process(target=printer, args=(item, lock))
        p.start()
```

### Python 实现多线程的方式

Python 中存在全局解释器锁（Global Interpreter Lock, GIL），所以同一时刻只能有一个线程获取锁并执行，遇到 IO 操作才会释放切换。因此在 Python 中，多线程使用的是 CPU 的一个核，适合 IO 密集型；多进程使用的是 CPU 的多个核，适合运算密集型。

在 Python 中，`threading`用来实现多线程，先看一个创建线程的实例：

```python
import threading


def run_thread(name):
    print(name)

if __name__ == '__main__':
    t = threading.Thread(target=run_thread, args=('Test', ))
    t.start()
```

与进程相似，线程也可以用集成父类的形式定义一个新的进程。