---
layout:     post
title:      "一起来学GCC与Makefile"
subtitle:   "专属于Linux的GCC和Makefile"
date:       2017-03-14 17:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - C
---

> 玩玩Linux下的C


## 前言

在Linux下的C编程中，不可避免地会接触到GCC和Makefile，今天就让我们一起来学一下这些东西！

---

## 正文

#### GCC

GCC是GNU推出的基于C/C++的编译命令，其编译有以下四个步骤：

1. 预处理：将包含的头文件编译进来，文件形式一般为hello.i，对应选项为-E
2. 编译：检查代码规范性及语法错误等，文件形式一般为hello.s，对应选项为-S
3. 汇编：把编译生成的”.s”文件转成二进制目标代码，文件形式一般为hello.o，对应选项为-c
4. 链接：成功编译后就进入了链接阶段，文件形式一般为hello，对应选项为-o

```shell
# 无选项编译链接。预处理、汇编、编译并链接形成可执行文件，默认输出为a.out
gcc hello.c
# -o选项。指定输出文件的文件名
gcc hello.c -o hello
# -E选项。预处理生成test.i
gcc -E hello.c -o hello.i
# -S选项。将预处理输出文件hello.i汇编成hello.s文件 
gcc –S hello.i –o hello.s
# -c选项。将汇编输出文件hello.s编译输出hello.o文件
gcc –c hello.s –o hello.o
# 无选项链接。将编译输出文件hello.o链接成最终可执行文件hello
gcc hello.o -o hello
# -O选项。用编译优化级别1编译程序。级别为1~3，级别越大效果越好，但时间越长
gcc -O1 hello.c -o hello
# -wall选项。显示警告信息
gcc -Wall hello.c -o hello
# 多文件编译。将test1.c和test2.c分别编译链接成test可执行文件，需要所有文件重新编译
gcc test1.c test2.c -o test
# 多文件编译。将test1.c编译成test1.o，再将test2.c编译成test2.o，最后将test1.o和test2.o链接成test，只重新编译修改文件，未修改文件不用重新编译
gcc -c test1.c
gcc -c test2.c
gcc -o test1.o test2.o -o test
```

Makefile就像一个Shell脚本一样，它带来的好处就是自动化编译。只需要一个`make`命令，整个工程完全自动编译，极大提高了软件开发效率。

Makefile的形式如下：

```c
target: prerequisites
	command
```

target是可以是Object File，也可以是执行文件，还可以是标；prerequisites是要生成target所需要的文件或目标；command是make需要执行的命令。

假设工程有8个C文件和3个头文件，要写Makefile来告诉make命令如何编译和链接这些文件，规则是：

1. 如果工程没有编译过，则所有C文件都要编译并被链接
2. 如果工程的某几个C文件被修改，则只编译被修改的C文件，并链接目标程序
3. 如果工程的头文件改变了，则编译引用了这些头文件的C文件，并链接目标程序

为了完成以上规则，Makefile大概如下：

```c
edit : main.o kbd.o command.o display.o /
	insert.o search.o files.o utils.o
	gcc -o edit main.o kbd.o command.o display.o /
	insert.o search.o files.o utils.o
main.o : main.c defs.h
	gcc -c main.c
kbd.o : kbd.c defs.h command.h
	gcc -c kbd.c
command.o : command.c defs.h command.h
	gcc -c command.c
display.o : display.c defs.h buffer.h
	gcc -c display.c
insert.o : insert.c defs.h buffer.h
	gcc -c insert.c
search.o : search.c defs.h buffer.h
	gcc -c search.c
files.o : files.c defs.h buffer.h command.h
	gcc -c files.c
utils.o : utils.c defs.h
	gcc -c utils.c
clean :
	rm edit main.o kbd.o command.o display.o /
	insert.o search.o files.o utils.o
```

我们把以上内容保存为文件名为Makefile或makefile的文件中，然后用命令`make`就可以生成执行文件edit。若要删除执行文件和所有中间目标文件，执行`make clean`就可以了。

在这个makefile中，目标文件（target）包含：执行文件edit和中间目标文件（*.o），依赖文件（prerequisites）就是冒号后面的那些 .c 文件和 .h文件。每一个 .o 文件都有一组依赖文件，而这些 .o 文件又是执行文件 edit 的依赖文件。依赖关系的实质上就是说明了目标文件是由哪些文件生成的，换言之，目标文件是哪些文件更新的。

在定义好依赖关系后，后续的那一行定义了如何生成目标文件的操作系统命令，一定要以一个Tab键作为开头。记住，make并不管命令是怎么工作的，他只管执行所定义的命令。make会比较targets文件和prerequisites文件的修改日期，如果prerequisites文件的日期要比targets文件的日期要新，或者target不存在的话，那么，make就会执行后续定义的命令。

这里要说明一点的是，clean不是一个文件，它只不过是一个动作名字，有点像[C语言](http://lib.csdn.net/base/c)中的lable一样，其冒号后什么也没有，那么，make就不会自动去找文件的依赖性，也就不会自动执行其后所定义的命令。要执行其后的命令，就要在make命令后明显得指出这个lable的名字。这样的方法非常有用，我们可以在一个makefile中定义不用的编译或是和编译无关的命令，比如程序的打包，程序的备份，等等。

## 后记




