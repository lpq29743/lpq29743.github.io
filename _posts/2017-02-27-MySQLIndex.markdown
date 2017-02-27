---
layout:     post
title:      "MySQL索引类型"
subtitle:   "从外到内了解MySQL索引"
date:       2017-02-27 19:00:00
author:     "林佩勤"
header-img: "img/post-bg.jpg"
tags:
    - MySQL
---

> 最“简单”的东西往往没那么简单


## 前言

索引是学习数据库中无法躲避的一个重要知识点，今天我们把讨论范围缩小到MySQL上，具体讲解一下MySQL的索引。

---

## 正文

#### 从逻辑的角度看

索引的分类方式有很多种，首先我们从逻辑的角度看，它主要分为以下几种：

**普通索引**

最基本的索引，没有任何限制，具体操作如下：

```sql
// 直接创建索引
CREATE INDEX indexName ON table(column(length))
// 修改表结构的方式添加索引
ALTER tableADD INDEX indexName ON (column(length))
// 创建表的时候同时创建索引
CREATE TABLE table (
	id int(11) NOT NULL AUTO_INCREMENT ,
	title char(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
	content text CHARACTER SET utf8 COLLATE utf8_general_ci NULL ,
	time int(10) NULL DEFAULT NULL ,
	PRIMARY KEY (id),
	INDEX indexName (title(length))
)
// 删除索引
DROP INDEX indexName ON table
```

**唯一索引**

与普通索引类似，不同的是索引列的值须唯一，但允许空值。如果是组合索引，则列值组合必须唯一，操作如下：

```sql
// 创建唯一索引
CREATE UNIQUE INDEX indexName ON table(column(length))
// 修改表结构
ALTER table ADD UNIQUE indexName ON (column(length))
// 创建表的时候直接指定
CREATE TABLE table (
	id int(11) NOT NULL AUTO_INCREMENT ,
	title char(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
	content text CHARACTER SET utf8 COLLATE utf8_general_ci NULL ,
	time int(10) NULL DEFAULT NULL ,
	PRIMARY KEY (id),
	UNIQUE indexName (title(length))
);
```

**全文索引**

全文索引仅可用于MyISAM，具体操作如下：

```sql
// 创建表的适合添加全文索引
CREATE TABLE table (
	id int(11) NOT NULL AUTO_INCREMENT ,
	title char(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
	content text CHARACTER SET utf8 COLLATE utf8_general_ci NULL ,
	time int(10) NULL DEFAULT NULL ,
	PRIMARY KEY (id),
	FULLTEXT (content)
);
// 修改表结构添加全文索引
ALTER TABLE article ADD FULLTEXT index_content(content)
// 直接创建索引
CREATE FULLTEXT INDEX index_content ON article(content)
```

**单列索引、多列索引**

索引可以是单列索引也可以是多列索引（也叫复合索引）。按照上面形式创建出来的索引是单列索引，现在先看看创建多列索引： 

```sql
create table test3 ( 
	id int not null primary key auto_increment, 
	uname char(8) not null default '', 
	password char(12) not null, 
	INDEX(uname,password) 
) type=MyISAM; 
```

**主键索引**

主键是一种唯一性索引，不能用`CREATE INDEX`创建PRIMARY KEY索引，具体操作如下：

```sql
// 主键一般在创建表的时候指定
CREATE TABLE 表名( [...], PRIMARY KEY (列的列表) );
// 通过修改表的方式加入主键
ALTER TABLE 表名 ADD PRIMARY KEY (列的列表);
```

#### 从存储的角度看

MYSQL存储引擎的索引不一定完全相同，也不是所有存储引擎都支持所有索引类型。MYSQL目前提供了4种索引：

- B-Tree索引：最常见类型，大部分引擎支持，接下来内容也默认以B树索引为基础展开
- HASH索引：只有Memory引擎支持，使用场景简单
- R-Tree索引（空间索引）：MyISAM的索引类型，用于地理空间数据类型
- Full-text（全文索引）：MyISAM的索引类型，用于全文索引，InnoDB从5.6版本提供全文索引支持

#### 索引使用策略及优化

**问题一：怎么分析查询语句**

MySQL执行查询前会分析SQL，如果发送`select * from blog where false`，MySQL是不会执行查询的，因为经过SQL分析器分析后MySQL已经清楚不会有语句符合操作。利用`EXPLAIN`可以分析SQL问题，具体如下：

```
mysql> EXPLAIN SELECT `birday` FROM `user` WHERE `birthday` < "1990/2/2";
-- 结果：
id: 1

select_type: SIMPLE -- 查询类型（简单查询,联合查询,子查询）

table: user -- 显示这一行的数据是关于哪张表的

type: range -- 区间索引（在小于1990/2/2区间的数据),这是重要的列,显示连接使用了何种类型。从最好到最差的连接类型为system > const > eq_ref > ref > fulltext > ref_or_null > index_merge > unique_subquery > index_subquery > range > index > ALL,const代表一次就命中,ALL代表扫描了全表才确定结果。一般来说,得保证查询至少达到range级别,最好能达到ref。

possible_keys: birthday  -- 指出MySQL能使用哪个索引在该表中找到行。如果是空的,没有相关的索引。这时要提高性能,可通过检验WHERE子句,看是否引用某些字段,或者检查字段不是适合索引。 

key: birthday -- 实际使用到的索引。如果为NULL,则没有使用索引。如果为primary的话,表示使用了主键。

key_len: 4 -- 最长的索引宽度。如果键是NULL,长度就是NULL。在不损失精确性的情况下,长度越短越好

ref: const -- 显示哪个字段或常数与key一起被使用。 

rows: 1 -- 这个数表示mysql要遍历多少数据才能找到,在innodb上是不准确的。

Extra: Using where; Using index -- 执行状态说明,这里可以看到的坏的例子是Using temporary和Using 
```

上面都在说使用索引的好处，但过多的使用索引将会造成滥用。因此索引也会有它的缺点：虽然索引大大提高了查询速度，同时却会降低更新表的速度，如对表进行INSERT、UPDATE和DELETE。因为更新表时，MySQL不仅要保存数据，还要保存一下索引文件。建立索引会占用磁盘空间的索引文件。一般情况这个问题不太严重，但如果你在一个大表上创建了多种组合索引，索引文件的会膨胀很快。索引只是提高效率的一个因素，如果你的MySQL有大数据量的表，就需要花时间研究建立最优秀的索引，或优化查询语句。下面是一些总结以及收藏的MySQL索引的注意事项和优化方法。

## 后记


