---
layout: wiki
title: Linux
categories: Linux
description: Linux
keywords: Linux, Linux命令
---

### ls

```bash
# 列出当前目录下的所有目录和文件
ls
# 列出 /bin 下的所有目录和文件
ls /bin
# 列出当前目录下的所有目录和文件（包含详细信息）
ls -l
# 列出当前目录下的所有目录和文件（包含隐藏文件）
ls -a
# 列出当前目录下的所有目录和文件（易读形式）
ls -lh
# 列出当前目录下的所有目录和文件（只列目录）
ls -d
# 列出当前目录下的所有目录和文件（按时间序输出）
ls -t
# 列出当前目录下的所有目录和文件（按大小序输出）
ls -S
# 列出当前目录下的所有目录和文件（按字母序反向输出）
ls -r
# 列出当前目录下的所有目录和文件（按时间反序输出）
ls -rt
# 以递归的方式列出所有子目录和文件
ls -R
```

### cd

### pwd

### mkdir

### rm

### rmdir

### mv

### cp

### scp

### rsync

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

```bash
# 列出当前目录及其子目录下的所有文件和文件夹
find .
# 删除当前目录及其子目录下的所有文件和文件夹
find . -delete
# 在当前目录及其子目录下查找（按文件名）
find . -name test.txt
# 在 /home 目录及其子目录下查找（按文件名）
find /home -name test.txt
# 在当前目录及其子目录下查找（按文件名后缀）
find . -name *.txt
# 在当前目录及其子目录下查找（按文件名，忽略大小写）
find . -iname test.txt
# 在当前目录及其子目录下查找（按正则文件名）
find . -regex .*\(\.txt\|\.pdf\)$
# 在当前目录及其子目录下查找（按正则文件名，忽略大小写）
find . -iregex .*\(\.txt\|\.pdf\)$
# 列出当前目录及其子目录下的所有文件
find . -type f
# 列出当前目录及其子目录下的所有文件夹
find . -type d
# 在当前目录及其子目录下查找（按权限）
find . -perm 777
# 在当前目录及其子目录下查找（按文件所属的用户）
find . -user xiaoming
# 在当前目录及其子目录下查找（按文件所属的组）
find . -group china
# 在当前目录及其子目录下查找（最近 20 天内更新）
find . -mtime -20
# 在当前目录及其子目录下查找（20 天以前更新）
find . -mtime +20
# 在当前目录及其子目录下查找（按大小，刚好 50M）
find . -size 50M
# 在当前目录及其子目录下查找（按大小，从 50M 到 100M）
find . -size +50M -size -100M
# 在当前目录及其子目录下查找（按目录深度）
find . -maxdepth 3
```

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

### awk

### sed

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