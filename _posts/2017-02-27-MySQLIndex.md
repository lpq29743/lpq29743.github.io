---
layout: post
title: MySQL 索引类型
categories: MySQL
description: MySQL索引类型
keywords: MySQL, MySQL索引, MySQL索引类型
---

索引是学习数据库中无法躲避的一个重要知识点，今天我们把讨论范围缩小到MySQL上，具体讲解一下MySQL的索引。

### 从逻辑的角度看

索引的分类方式有很多种，首先我们从逻辑的角度看，它主要分为以下几种：

#### 普通索引

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

#### 唯一索引

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

#### 全文索引

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

#### 单列索引、多列索引

索引可以是单列索引也可以是多列索引（也叫复合索引）。按照上面形式创建出来的索引是单列索引，现在先看看创建多列索引： 

```sql
create table test3 ( 
	id int not null primary key auto_increment, 
	uname char(8) not null default '', 
	password char(12) not null, 
	INDEX(uname,password) 
) type=MyISAM; 
```

#### 主键索引

主键是一种唯一性索引，不能用`CREATE INDEX`创建PRIMARY KEY索引，具体操作如下：

```sql
// 主键一般在创建表的时候指定
CREATE TABLE 表名( [...], PRIMARY KEY (列的列表) );
// 通过修改表的方式加入主键
ALTER TABLE 表名 ADD PRIMARY KEY (列的列表);
```

### 从存储的角度看

MYSQL存储引擎的索引不一定完全相同，也不是所有存储引擎都支持所有索引类型。MYSQL目前提供了4种索引：

- B-Tree索引：最常见类型，大部分引擎支持，接下来内容也默认以B树索引为基础展开
- HASH索引：只有Memory引擎支持，使用场景简单
- R-Tree索引（空间索引）：MyISAM的索引类型，用于地理空间数据类型
- Full-text（全文索引）：MyISAM的索引类型，用于全文索引，InnoDB从5.6版本提供全文索引支持

### 索引使用策略及优化

#### 问题一：怎么分析查询语句

MySQL执行查询前会分析SQL，如果发送`select * from blog where false`，MySQL是不会执行查询的，因为经过SQL分析器分析后MySQL已经清楚不会有语句符合操作。利用`EXPLAIN`可以分析SQL问题，具体如下：

```
mysql> EXPLAIN SELECT `birday` FROM `user` WHERE `birthday` < "1990/2/2";
-- 结果：
id: 1
select_type: SIMPLE		// 查询类型（简单查询,联合查询,子查询）
table: user		// 显示这一行的数据是关于哪张表的
type: range		// 区间索引，显示使用何种类型。一般要保证至少到range，最好到ref
possible_keys: birthday		// 可用索引。若没有，则要提高性能，检验WHERE子句，看是否引用某些字段或某些字段不适合索引
key: birthday	// 实际使用索引。如果为NULL，则没有使用索引。如果为primary的话，表示使用了主键
key_len: 4		// 最长索引宽度。如果键是NULL，长度就是NULL。在不损失精确性的情况下，长度越短越好
ref: const		// 显示哪个字段或常数与key一起被使用
rows: 1			// 表示mysql要遍历多少数据才能找到，在innodb上是不准确的
Extra: Using where; Using index		// 执行状态说明，这里可以看到的坏的例子是Using temporary和Using
```

另外对其中几个输出再作详细的介绍：

**select_type**

- simple：简单select（不使用union或子查询）
- primary：最外面的select
- union：union中的第二个或后面的select语句
- dependent union：union中的第二个或后面的select语句，取决于外面的查询
- union result：union的结果
- subquery：子查询中的第一个select
- dependent subquery：子查询中的第一个select，取决于外面的查询
- derived：导出表的select（from子句的子查询）

**type**

- system：表只有一行（system表）。这是const连接类型的特殊情况
- const：表中一个记录的最大值能匹配查询。因为仅一行，故MYSQL先读值再把它当常数
- eq_ref：表连接时，MYSQL从前面的表中，对每个记录的联合都从表中读取一个记录
- ref：查询使用不是唯一或主键的键或这些类型的部分时发生。对于之前表每个行联合，全部记录都从表读出
- range：使用索引返回一个范围中的行，使用>或<时发生
- index：对前面表中每个记录联合进行完全扫描
- ALL：对前面每个记录联合进行完全扫描，应尽量避免

**Extra**

- Distinct：一旦找到与行相联合匹配的行，就不再搜索
- Not exists：MYSQL优化了LEFT JOIN，一旦找到匹配LEFT JOIN的行，就不再搜索
- Range checked for each Record：没找到理想索引。对前面表中的每个行组合，检查使用哪个索引，并用它返回行
- Using filesort：查询需优化。需根据连接类型及存储排序键值和匹配条件的全部行的行指针来排序
- Using index：列数据从仅使用索引信息而无读取行动的表返回，发生在对表的全部请求列都是同一索引的部分时
- Using temporary：查询需优化。需创建临时表存储结果，发生在对不同列集ORDER BY上，而不是GROUP BY上
- Where used：使用了WHERE从句

#### 问题二：使用索引会带来什么影响

索引可以大大提高查询效率，但同时会降低更新表的速度，如对表进行`INSERT`、`UPDATE`和`DELETE`。因为更新表时，不仅要保存数据，还要保存索引文件。

#### 问题三：怎么判断是否应该创建索引

- 较频繁的作为查询条件的字段应该创建索引
- 唯一性太差的字段不适合单独创建索引，即使频繁作为查询条件
- 更新非常频繁的字段不适合创建索引
- 不会出现在WHERE子句中的字段不该创建索引
- 表记录较少，如果小于2000，一般没必要建索引
- 索引列基数越大，索引效果越好。如性别索引没有多大用，不管怎样都会得出约一半的行

#### 问题四：什么样的sql语句会使用到索引

MySQL只对以下操作符使用索引：<、<=、=、>、>=、between、in和某时的like（不以%或_开头），具体如下：

````sql
SELECT sname FROM stu WHERE age+10=30;		// 不会使用索引,因为所有索引列参与了计算
SELECT sname FROM stu WHERE LEFT(date,4) <1990;		// 不会使用索引,因为使用了函数运算,原理与上面相同
SELECT * FROM houdunwang WHERE uname LIKE'后盾%'		// 走索引
SELECT * FROM houdunwang WHERE uname LIKE "%后盾%"	// 不走索引
// 正则表达式不使用索引，字符串与数字比较也不使用索引，如下
CREATE TABLE a (a char(10));
EXPLAIN SELECT * FROM a WHERE a="1"		// 走索引
EXPLAIN SELECT * FROM a WHERE a=1		// 不走索引
// 如果sql有or，即使有条件带索引也不会用。or要求所有字段都必须建立索引, 故应尽量避免使用or关键字
select * from dept where dname='xxx' or loc='xx' or deptno=45	
````

#### 问题五：什么是最左前缀原则

最左前缀原则指的是在sql where子句中列的顺序要和多索引一致，只要非顺序出现、断层都无法用到多列索引。