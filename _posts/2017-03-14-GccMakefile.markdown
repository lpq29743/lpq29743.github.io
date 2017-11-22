---
layout: post
title: 一起来学 GCC 与 Makefile
categories: C
description: 一起来学GCC与Makefile
keywords: C, GCC, Makefile, Linux C
---

在Linux下的C编程中，不可避免地会接触到GCC和Makefile，今天就让我们一起来学一下这些东西！

### GCC

GCC是GNU推出的基于C/C++的编译命令，其编译包括以下四个步骤：

1. 预处理：将包含的头文件编译进来，文件形式一般为hello.i，对应选项为-E
2. 编译：检查代码规范性及语法错误等，文件形式一般为hello.s，对应选项为-S
3. 汇编：把编译生成的”.s”文件转成二进制目标代码，文件形式一般为hello.o，对应选项为-c
4. 链接：成功编译后就进入了链接阶段，文件形式一般为hello，对应选项为-o

接下来我们再介绍一下GCC的命令：

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

### Makefile

#### 什么是Makefile

Makefile就像Shell脚本，它带来的好处是自动化编译。只需一个make命令，工程就自动编译，其形式具体如下：

```shell
target: prerequisites
	command
```

target是可以是Object File，也可以是执行文件，还可以是标签；prerequisites是生成target所需的文件或目标；command是make需要执行的命令。

#### 一个例子

假设工程有8个C文件和3个头文件，我们要写一个Makefile并且满足以下规则：

1. 如果工程没有编译过，则所有C文件都要编译并被链接
2. 如果工程的某几个C文件被修改，则只编译被修改的C文件，并链接目标程序
3. 如果工程的头文件改变了，则编译引用了这些头文件的C文件，并链接目标程序

为了完成以上规则，Makefile大概如下：

```shell
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

我们把以上内容保存为文件名为Makefile或makefile的文件中，然后用命令make就可以生成执行文件edit。若要删除执行文件和所有中间目标文件，执行make clean就可以了。使用clean不仅便于重编译，也很利于保持文件清洁。

在这个makefile中，target包含执行文件edit和中间目标文件，prerequisites就是冒号后的.c文件和.h文件。每个.o文件都有一组依赖文件，而这些.o文件又是edit的依赖文件。依赖关系说明了目标文件是由哪些文件生成的，或者说是更新的。

定义好依赖关系后，后续那行定义了生成目标文件的操作系统命令。make会比较targets文件和prerequisites文件的修改日期，如果prerequisites文件比targets文件新，或target不存在的话，则make就会执行后续定义的命令。

在整个工程中，make的具体工作如下：

1. 在当前目录下找叫Makefile或makefile的文件
2. 如果找到，它会找文件中的第一个target，在上面例子中是edit这个文件，并把这个文件作为最终的目标文件
3. 如果edit文件不存在，或依赖文件比edit新，那么，它会执行后面所定义的命令生成edit
4. 如果edit依赖的文件存在，那么make会在当前文件中找目标为.o文件的依赖性，若找到则再根据规则3生成.o文件
5. 根据C文件和H文件，make会生成.o文件，再用.o文件声明make的终极任务，也就是执行文件edit了

找寻过程中如果出现错误，如依赖文件找不到，那么make会退出并报错，而对于命令错误或编译不成功，make不理采。像clean这种没有被关联到的，它定义的命令不会自动执行，只能用命令make clean显式执行。

在上面makefile例子中的第一段，.o文件的字符串重复了两次，如果要加入新的.o文件，则需要多次修改。对于这个问题，我们可以使用变量进行解决，具体改良如下：

```shell
objects = main.o kbd.o command.o display.o \
	insert.osearch.o files.o utils.o 
edit : $(objects)
	gcc -o edit $(objects)
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
	rm edit $(objects)
```

make可推导文件及依赖关系后面的命令，一看到.o文件，就会把.c文件加到依赖关系，故可改进代码如下（其实.PHONY表示clean是伪目标文件）：

```shell
objects = main.o kbd.o command.o display.o \
	insert.osearch.o files.o utils.o 
edit : $(objects)
	gcc -o edit $(objects)
main.o : defs.h
kbd.o : defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h
.PHONY : clean
clean :
	rm edit $(objects)
```

即然make可以自动推导命令，那其实makefile可以继续改进：

```shell
objects = main.o kbd.o command.o display.o \
	insert.osearch.o files.o utils.o 
edit : $(objects)
	gcc -o edit $(objects)
$(objects) : defs.h
kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h
.PHONY : clean
clean :
	rm edit $(objects)
```

这让makefile变得简单，但依赖关系显得凌乱，具体选择哪一种写法，就看大家喜好了！

#### 引用其它的Makefile

Makefile使用include可以把别的Makefile包含进来，include的语法是：

```shell
# filename可以是当前操作系统Shell的文件模式
include<filename>
# 有几个Makefile：a.mk、b.mk、c.mk，还有foo.make，以及变量$(bar)，其包含了e.mk和f.mk，则下面语句：
include foo.make *.mk $(bar)
# 等价于：
include foo.make a.mk b.mk c.mk e.mk f.mk
```

如果文件没指定路径，make会在当前目录下寻找，如果没找到，则还会在以下目录查找：

1. 如果make执行时，有“-I”或“--include-dir”参数，那么make就会这个参数所指定的目录寻找
2. 如果目录/include（一般是/usr/local/bin或/usr/include）存在，make也会在该目录寻找

如果没有找到，make会生成警告信息，但不会出现错误，并继续载入其它文件。一旦读完makefile，make会重试没有找到或不能读取的文件，如果不行，make会出现一条致命信息。如果想让make不理会无法读取的文件而继续执行，可以在include前加减号“-”。如-include<filename>。

#### 在规则中使用通配符

make支持以下通配符：

- **\~**：“\~/test”表示当前用户$HOME目录下的test目录，而“~hchen/test”表示用户hchen宿主目录下的test目录。
- **\\**：如果我们的文件名中有通配符，如“\*”，那么可以用转义字符“\”表示真实的“*”字符