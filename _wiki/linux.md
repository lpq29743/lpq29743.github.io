---
layout: wiki
title: Linux
categories: Linux
description: Linux
keywords: Linux, Linux命令
---

### ls

### cd

### pwd

### mkdir

### rm

### rmdir

### mv

### cp

### touch

### cat

```bash
# 显示文件内容
cat test.txt
# 显示多个文件内容
cat test1.txt test2.txt
# 倒序显示文件内容
tac text.txt
# 无论是否有空行，都输出行号，相当于 nl -b a test.txt
cat -n test.txt
# 对空行不编号
cat -b test.txt
# 出现多个空白行时，替代为一行的空白行
cat -s test.txt
# 将 test1.txt 文件覆盖到 test2.txt 文件中，相当于复制
cat test1.txt > test2.txt
# 合并文件
cat test1.txt test2.txt > test.txt
# 将 test1.txt 文件加上行号追加到 test2.txt 文件末尾
cat -n test1.txt >> test2.txt
# 创建文件 test.txt，并输入文件内容
cat > test.txt
```

### nl

```bash
# 输出带有行号的文本内容
nl test.txt
```

### more

```bash
# 逐页显示文本内容，空格键显示下一页，回车键显示下一行，q 退出
more test.txt
# 逐页显示文本内容，多个空白行用一个空白行显示
more -s test.txt
# 从第 20 行开始显示内容
more +20 test.txt
```

### less

```bash
# 逐页显示文本内容，空格键显示下一页，b 展示上一页，回车键显示下一行，q 退出
less test.txt
# 查看进程信息并用 less 分页显示
ps -ef | less
```

### head

```bash
# 显示文本开头内容，默认为 10 行
head test.txt
# 显示文本的前 20 行
head -n 20 test.txt
# 显示文本的前 20 个字节
head -c 20 test.txt
# 显示除了后 20 行的文本内容
head -n -20 test.txt
# 显示除了后 20 个字节的文本内容
head -c -20 test.txt
```

### tail

```bash
# 显示文本结尾内容，默认为 10 行
tail test.txt
# 持续监测文本结尾内容
tail -f test.txt
# 显示文本的后 20 行
tail -n 20 test.txt
# 从第 20 行开始显示文本内容
tail -n +20 test.txt
# 显示前 20 行，但从第 11 行开始
head -n 20 test.txt | tail -n +11
```

### which

### whereis

### locate

### find

### chmod

### tar

### chgrp

### chown

### gzip

### df

### du

### ln

### diff

### date

### cal

### grep

### wc

### ps

### kill

### killall

### top

### free

### vmstat

### iostat

### watch

### at

### crontab

### lsof

### ifconfig

### route

### ping

### traceroute

### netstat

### ss

### telnet

### rcp

### scp

### wget