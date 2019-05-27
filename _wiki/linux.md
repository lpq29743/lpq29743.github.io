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
# 查看进程运行目录
ls /proc/PID
```

### cd

```bash
# 切换到某个路径，可为绝对路径，也可为相对路径
cd some_dir
# 相当于 cd ~，切换到 /home 目录
cd
```

### pwd

```bash
# 打印当前工作目录
pwd
```

### mkdir

```bash
# 创建一个新目录
mkdir new_dir
# 创建多个新目录
mkdir new_dir1 new_dir2
# 递归创建新目录
mkdir -p new_dir/sub_new_dir
# 创建新目录并设置权限（方式一）
mkdir -m=r-- new_dir
# 创建新目录并设置权限（方式二）
mkdir -m 711 new_dir
# 创建目录并打印信息
mkdir -v new_dir
```

### rm

```bash
# 删除文件
rm file
# 删除前询问
rm -i file
# 强制删除，无需确认
rm -f file
# 将目录及其下文件删除
rm -r dir
# 将当前目录下的所有内容删除
rm -r /path/to/directory/*
# 删除并打印信息
rm -v file
```

### rmdir

```bash
# 删除目录
mkdir dir
# 递归删除目录
mkdir -p new_dir/sub_new_dir
```

### mv

```bash
# 移动一个文件到目录
mv file dir
# 移动多个文件到目录
mv file1 file2 dir
# 文件改名
mv file1 file2
# 文件改名（询问）
mv -i file1 file2
# 文件改名（不询问）
mv -f file1 file2
# 文件覆盖前做备份
mv -b file1 file2
# 目录移动
mv dir1 dir2
```

### cp

```bash
# 复制文件
cp file1 file2
# 复制文件（询问）
cp -i file1 file2
# 复制文件（备份）
cp -b file1 file2
# 创建硬链接
cp -l file1 file2
# 创建软链接
cp -s file1 file2
# 递归复制目录
cp -r dir1 dir2
```

### scp

```bash
# 从本地复制到远程
scp /path/file1 myuser@192.168.0.1:/path/file2
# 从远程复制到本地
scp myuser@192.168.0.1:/path/file2 /path/file1
# 限制速度为 400 Kbit/s
scp -l 400 /path/file1 myuser@192.168.0.1:/path/file2
# 使用 2222 端口
scp -P 2222 /path/file1 myuser@192.168.0.1:/path/file2
# 使用 IPv4
scp -4 /path/file1 myuser@192.168.0.1:/path/file2
# 使用 IPv6
scp -6 /path/file1 myuser@192.168.0.1:/path/file2
```

### touch

```bash
# 创建一个空文件
touch new_file
# 修改文件的时间属性为当前系统时间
touch old_file
# 只更新访问时间
touch -a old_file
# 只更新修改时间
touch -m old_file
# 将一个文件的时间属性复制到另一个文件上
touch -r source target
# 指定时间属性（2 月 5 号 13 点 51 分）
touch -t 02051351 old_file
```

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

### umask

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
# 把搜索到的文件删除掉（方式一）
find ./foo -type f -name "*.txt" -exec rm {} \; 
# 把搜索到的文件删除掉（方式二）
find ./foo -type f -name "*.txt" | xargs rm
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

```bash
# 在 test.txt 中寻找 str 字符串
grep str test.txt
# 在 test1.txt，test2.txt 中寻找 str 字符串
grep str test1.txt test2.txt
# 在当前目录下的所有文件中寻找 str 字符串
grep -r str .
# 在 test.txt 中寻找 str 字符串，并输出匹配行的行号
grep -n str test.txt
# 在 test.txt 中寻找不含 str 字符串的行
grep -v str test.txt
# 在 test.txt 中寻找 str 字符串，并忽略字母大小写
grep -i str test.txt
# 计算在 test.txt 中中出现 str 字符串的行数
grep -c str test.txt
# 在 test.txt 中全字寻找 str 字符串
grep -w str test.txt
# 在 test.txt 中寻找 str 字符串，并使查询结果高亮
grep --color str test.txt
# 使用正则表达式查找，相当于 egrep "[1-9]+"
grep -E "[1-9]+"
```

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

```bash
# 检查端口
lsof -i :80
```

### ifconfig

### route

### ping

### traceroute

### netstat

```bash
# 检查端口
netstat -tplugn | grep :22
```

### ss

```bash
# 检查端口
ss -lntu | grep ':25'
```

### telnet

### rcp

### scp

### wget