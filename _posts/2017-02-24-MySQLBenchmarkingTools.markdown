---
layout:     post
title:      "MySQL基准测试工具"
subtitle:   "MySQL的压力测试"
date:       2017-02-24 21:30:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - MySQL
---

> 开始寻找MySQL的瓶颈……


## 前言

MySQL基准测试（benchmark），是新手和专家都必须掌握的一项基本技能，今天就让我们一起来看看有哪些基准测试的工具！

---

## 正文

#### 集成式测试工具

**ab**

Apache测试工具，可测试每秒处理请求数。对于Windows，ab默认在apache的bin目录下，而Linux可以用`yum install httpd-tools`进行安装。使用简单，用途有限，具体使用方法如下：

```shell
// -c 10 代表并发数是10
// -n 10 总共进行100次访问
D:\phpStudy\Apache\bin>ab -c 10 -n 1000 http://example.com/phpinfo.php
```

**http_load**

http_load以并行复用的方式运行，用于测试web服务器的吞吐量与负载。http_load基于Linux，windows下也可用（[下载地址](http://download.csdn.net/detail/pcvic/8138847)）。它可以以单一进程运行，一般不会搞死客户机，还可测试HTTPS请求。我们可通过文件提供多个url，http_load随机选择进行测试，也以订制http_load，使其按照时间比率测试，而不仅仅测试最大请求处理能力。具体使用方法如下：

1. 创建urls.txt文件

   ```
   http://www.mysqlperformanceblog.com/
   http://www.mysqlperformanceblog.com/page/2/
   http://www.mysqlperformanceblog.com/mysql-patches/
   http://www.mysqlperformanceblog.com/mysql-performance-presentations/
   http://www.mysqlperformanceblog.com/2006/09/06/slow-query-log-analyzes-tools/
   ```

2. 使用http_load

   ```shell
   // 命令格式：http_load -p 并发访问进程数 -s 访问时间 需要访问的URL文件
   // 同时使用50个进程，随机访问urls.txt中的网址列表，总共访问10秒
   http_load -p 50 -s 10 urls.txt
   // 命令格式：http_load -r 每秒的访问频率 -f 总计的访问次数 需要访问的URL文件
   // 每秒请求50次，总共请求5000次停止
   http_load -r 50 -f 5000 urls.txt
   ```

3. 分析结果

   ```
   49 fetches, 2 max parallel, 289884 bytes, in 10.0148 seconds
   // 测试运行了49个请求，最大并发进程数是2，总计传输数据是289884bytes，运行时间是10.0148秒
   5916 mean bytes/connection
   // 说明每一连接平均传输的数据量289884/49=5916
   4.89274 fetches/sec, 28945.5 bytes/sec
   // 说明每秒的响应请求为4.89274，每秒传递的数据为28945.5bytes/sec
   msecs/connect: 28.8932 mean, 44.243 max, 24.488 min
   // 说明每连接平均响应时间是28.8932msecs，最大响应时间为44.243msecs，最小响应时间为24.488msecs
   msecs/first-response: 63.5362 mean, 81.624 max, 57.803 min
   HTTP response codes: code 200 — 49
   // 说明打开响应页面的类型，如果403类型过多，那可能是系统遇到了瓶颈
   ```

**JMeter**

JMeter用Java实现，查看源代码及文档请点击[这里](http://jakarta.apache.org/jmeter/index.html)。JMeter可用于测试静态或动态资源的性能（文件、Servlet、Perl脚本、Java对象、数据库和查询、FTP服务器等）。JMeter可以模拟在服务器、网络或其他对象上附加高负载以测试受压能力，或分析不同负载下的总性能情况。使用JMeter需先配置好Java环境，然后下载安装JMeter（[下载地址](http://jmeter.apache.org/download_jmeter.cgi)），进入bin目录（Windows用户）执行jmeter.bat就可运行软件了。相比前面的工具，JMeter较为复杂，所以本文不会详解。

#### 单组件测试工具

**mysqlslap**

MySQL从5.1.4开始提供的压力测试工具，通过模拟多个并发客户端访问MySQL来执行压力测试，并提供详细的性能报告，且能对比多个存储引擎在相同环境下的并发压力性能。具体使用方法如下：

```
// 查看帮助
mysqlslap –help
// 以自动生成测试表和数据的形式，分别模拟50和100个客户端并发连接处理1000个query的情况
mysqlslap -a --concurrency=50,100 --number-of-queries=1000 
// 增加--debug-info选项，可以输出内存和CPU信息
mysqlslap -a --concurrency=50,100 --number-of-queries=1000 --debug-info
// 增加--iterations选项，可以重复执行5次 
mysqlslap -a --concurrency=50,100 --number-of-queries=1000 --iterations=5 --debug-info
// 可以针对远程主机上的mysql进行测试
mysqlslap -a --concurrency=50,100 --number-of-queries=1000 -h 172.16.81.99 -P 3306 -p
// 使用--only-print选项查看测试过程中如何执行sql语句。这种方式仅会对数据库进行模拟操作 
mysqlslap -a --only-print
// 使用--defaults-file选项，指定从配置文件中读取选项配置
// 使用--number-int-cols选项，指定表中会包含4个int型的列
// 使用--number-char-cols选项，指定表中会包含 35 个char型的列
// 使用--engine选项，指定针对何种存储引擎进行测试
mysqlslap --defaults-file=/etc/my.cnf --concurrency=50,100,200 --iterations=1 --number-int-cols=4 --number-char-cols=35 --auto-generate-sql --auto-generate-sql-add-autoincrement --auto-generate-sql-load-type=mixed --engine=myisam,innodb --number-of-queries=200 --debug-info -S /tmp/mysql.sock
// 除了以上的使用方式，还可以使用存储过程进行测试，这里由于篇幅原因并不给出
// 有时候命令需要加入账号和密码，如-uroot -p123456
```

**sysbench**

多线程系统的基准测试工具，可测量文件I/O性能、调度性能、内存分配和交换速度、POSIX线程以及数据库本身。它的安装稍微有点复杂，这里不给出，我们直接讲解它的使用：

1. 通用选项

   ```
   --num-threads=N		// 创建测试线程的数目。默认为1
   --max-requests=N	// 请求的最大数目。默认为10000，0代表不限制
   --max-time=N		// 最大执行时间，单位是s。默认是0,不限制
   --forced-shutdown=STRING	// 超过max-time强制中断。默认是off
   --thread-stack-size=SIZE	// 在测试开始时是否初始化随机数发生器。默认是off
   --test=STRING		// 指定测试项目名称
   --debug=[on|off]	// 是否显示更多的调试信息。默认是off
   --validate=[on|off]	// 在可能情况下执行验证检查。默认是off
   --help=[on|off]		// 帮助信息
   --version=[on|off]	// 版本信息
   ```

2. 简单测试

   ```
   // 测试CPU
   sysbench --test=cpu --cpu-max-prime=2000 run
   // 测试线程
   sysbench --test=threads --num-threads=500 --thread-yields=100 --thread-locks=4 run
   // 测试IO：sysbench --num-threads 开启的线程 --file-total-size 总的文件大小
   // prepare阶段，生成需要的测试文件，完成后会在当前目录下生成很多小文件。
   sysbench --test=fileio --num-threads=16 --file-total-size=2G --file-test-mode=rndrw prepare     
   // run阶段
   sysbench --test=fileio --num-threads=20 --file-total-size=2G --file-test-mode=rndrw run
   // 清理测试时生成的文件
   sysbench --test=fileio --num-threads=20 --file-total-size=2G --file-test-mode=rndrw cleanup
   // 测试内存
   sysbench --test=memory --memory-block-size=8k --memory-total-size=1G run
   // 测试mutex
   sysbench –test=mutex –num-threads=100 –mutex-num=1000 –mutex-locks=100000 –mutex-loops=10000 run
   // OLTP测试较为复杂，这里由于篇幅原因并不给出
   ```

**Database Test Suite**

Database Test Suite由Open-Source Development Labs设计开发，实现了TPC标准，只支持InnoDB和Falcon，网上资料相对较少，这里简单带过。

**MySQL Benchmark Suite (sql-bench)**

MySQL发布的工具，由一系列Perl模块和脚本构成。它可以用来测试其他数据库系统的性能，可以在MySQL、Oracle和Microsoft SQL Server上运行同样测试，准确知道MySQL比目前使用的数据库系统好多少。如果想在其他数据库系统使用它，可以用`--server='server'`，这个参数的可取值包括MySQL、Oracle、Informai和MS-SQL。其常用参数如下：

```
--log			// 把测试结果保存到指定文件
--dir			// 指定存放测试结果的子目录
--user			// 用来登录服务器的用户名
--password		// 用来登录服务器的口令字
--host			// 服务器的主机名
--small-test	// 小测试模式，运行最少的性能测试。省略将执行全部测试。小测试一般已足以确定常用性能指标
```

**Super Smack**

针对MySQL和PostgreSQL的测试工具，只在少数Unix和Linux上运行。它可模拟更多用户，读取测试数据到数据库中，并能在表中随机产生数据。为了让效果更好，测试前应禁用掉MySQL的查询缓存功能，执行`SET GLOBALS query_cache_size = 0;`。如果小测试模式已满足需要，可运行`perl run-all-tests-small -test`生成报告。运行所有测试可做出全面评估，但花费时间长。如果只想了解某个特定部分的性能，可进行单项测试。常用的单项测试如下：

```
test-ATIS.sh		// 创建29个数据表并对它们进行一些查询
test-connect.sh		// 测试服务器的连接速度
test-create.sh		// 测试数据表的创建速度
test-insert.sh		// 测试数据表的创建和填充操作
test-wisconsin.sh	// 运行这个性能测试工具的PostgreSQL版本      
```

## 后记

由于本文篇幅的原因，并不能对每一个工具都一一详解，但其中几个工具的使用还是相当重要的，需要的朋友可以找相关文档或教程深入学习。