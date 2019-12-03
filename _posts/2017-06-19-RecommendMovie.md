---
layout: post
title: UserCF 和 ItemCF 在 MovieLens 上的实现
categories: ArtificialIntelligence
description: UserCF和ItemCF在MovieLens上的实现
keywords: 人工智能, 推荐系统, 推荐系统, 协同过滤算法, UserCF, ItemCF
---

最近刚好在阅读项亮的《推荐系统实践》，又恰巧遇上了课程上数据挖掘大作业的要求，于是一想，便利用了这个机会写下了下面这篇文章。

### 背景介绍

本文采用的数据集是GroupLens提供的MovieLens数据集（附[下载地址](https://grouplens.org/datasets/movielens/)），MovieLens数据集包含6000多用户对4000多部电影的100万条评分。该数据集是一个评分数据集，用户可以给电影评5个不同等级的分数（1-5分）。

### 问题定义

基于多名用户对多部电影的评分数据，实现向用户推荐N部电影的功能。此处推荐的任务是预测用户会不会对某部电影评分，而不是预测用户在准备对某部电影评分的前提下会给电影评多少分。

### 解决方案

#### 什么是协同过滤算法

协同过滤算法是推荐系统中最基本的算法，分为两大类，一类是基于用户的协同过滤算法，另一类是基于物品的协同过滤算法。

#### 什么是基于用户的协同过滤算法

基于用户的协同过滤算法是推荐系统中最古老的算法。一定程度上讲，该算法的诞生标志了推荐系统的诞生。基于用户的协同过滤算法于1992年被提出，并应用于邮件过滤算法，1994年被GroupLens用于新闻过滤。在此之后直到2000年，该算法都是推荐系统领域最著名的算法。

#### 怎样实现基于用户的协同过滤算法

基于用户的协同过滤算法主要包括两个步骤，分别是：

1. 找到和目标用户兴趣相似的用户集合
2. 找到这个集合中的用户喜欢的，且目标用户没有听说过的物品推荐给目标用户

下面我们根据这两个步骤来实现一下电影的TopN推荐。对于步骤一，最关键的是如何计算两个用户的兴趣相似度，这里我们可以用余弦相似度来计算，即`相似度 = 用户u和用户v共同评价过的电影数 / √(用户u评价的电影数 * 用户v评价的电影数)`。由于对两两用户计算余弦相似度非常耗时，所以我们可以先计算这个公式的分子，如果分子为0，即用户u和用户v没有共同评价过的电影，则无需计算余弦相似度。为此我们可以建立电影到用户的倒排表，对于每部电影都保存对评价过该电影的用户列表，然后用稀疏矩阵usersim_mat表示用户u和用户v共同评价过的电影，这样子，扫描一遍倒排表并将同一物品下的两两用户对应的矩阵值加1，就可以通过稀疏矩阵中值为0的点知道哪些用户没有共同评价的电影了。具体实现的代码如下（Python3版本，以下代码都是Python3版本）：

```python
def calc_user_sim(self):
	# 构建物品-用户倒排表
	movie2users = dict()
	for user, movies in self.trainset.items():
		for movie in movies:
			if movie not in movie2users:
				movie2users[movie] = set()
			movie2users[movie].add(user)
			if movie not in self.movie_popular:
				self.movie_popular[movie] = 0
			self.movie_popular[movie] += 1

	# 计算两两用户之前的共同评价电影数
	usersim_mat = self.user_sim_mat
	for movie, users in movie2users.items():
		for u in users:
			for v in users:
				if u == v:
					continue
				usersim_mat.setdefault(u, {})
				usersim_mat[u].setdefault(v, 0)
				usersim_mat[u][v] += 1

	# 计算用户兴趣相似度
	for u, related_users in usersim_mat.items():
		for v, count in related_users.items():
			usersim_mat[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))
```

得到用户之间的兴趣相似度后，我们就可以用UserCF算法给用户推荐和他兴趣最相似的K个用户的电影了，这里我们直接用K个用户中看过某电影的用户群的兴趣相似度之和来表示被推荐用户对某部电影的感兴趣程度，具体的推荐函数实现如下：

```python
def recommend(self, user):
	K = self.n_sim_user
	N = self.n_rec_movie
	rank = dict()
	watched_movies = self.trainset[user]

	for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                  key=itemgetter(1), reverse=True)[0:K]:
		for movie in self.trainset[similar_user]:
			if movie in watched_movies:
				continue
			# 预测该用户对每部电影的兴趣
			rank.setdefault(movie, 0)
			rank[movie] += similarity_factor
	# 返回评分最高的N部电影
	return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
```

实现两个核心的步骤之后，我们也可以得到一个完整的程序了。我们可以按一定的比例将数据集分为训练集和测试集，训练集训练模型，而测试集测试预测的好坏，具体的代码可参考[这里](https://github.com/Lockvictor/MovieLens-RecSys/blob/master/usercf.py)。

#### 什么是基于物品的协同过滤算法

基于物品的协同过滤算法是目前业界应用最多的算法。无论是Amazon，还是Netflix、Hulu、YouTube，其推荐算法的基础都是该算法。

#### 怎样实现基于物品的协同过滤算法

基于物品的协同过滤算法主要分为两步，分别是：

1. 计算物品之间的相似度
2. 根据物品的相似度和用户的历史行为给用户生成推荐列表

计算物品之间相似度的方法与上面计算用户之间相似度的方法类似，即`相似度 = 评价过电影i和电影j的用户数 / √(评价过电影i的用户数 * 评价过电影j的用户数)`，同样可以通过建立用户-物品倒排表来减少计算量，具体的实现代码如下：

```python
def calc_movie_sim(self):
	for user, movies in self.trainset.items():
		for movie in movies:
			# 计算每部电影评价的用户数
			if movie not in self.movie_popular:
				self.movie_popular[movie] = 0
			self.movie_popular[movie] += 1

	# 计算两两电影的共同评价用户数
	itemsim_mat = self.movie_sim_mat
	for user, movies in self.trainset.items():
		for m1 in movies:
			for m2 in movies:
				if m1 == m2:
					continue
				itemsim_mat.setdefault(m1, {})
				itemsim_mat[m1].setdefault(m2, 0)
				itemsim_mat[m1][m2] += 1

	# 计算相似矩阵
    for m1, related_movies in itemsim_mat.items():
        for m2, count in related_movies.items():
            itemsim_mat[m1][m2] = count / math.sqrt(
                self.movie_popular[m1] * self.movie_popular[m2])
```

计算完两两电影之间的相似度之后，我们便可以根据用户的历史评分记录给出N部电影的推荐了，具体实现的思路如下：

1. 找出与某用户看过的某部电影 i 相似度最大的K部电影
2. 遍历这K部电影，如果该用户看过则跳过，否则则尝试将其加入候选推荐电影列表，如果已在列表中，则在原来的基础上将推荐指数加上相似度与电影 i 评分的乘积，否则则加入列表并将初始推荐指数设为相似度与电影 i 评分的乘积
3. 遍历该用户看过的所有电影，最后可以得到一个推荐列表，返回此列表中推荐指数最高的N部电影

有了清晰的思路之后，我们就可以用代码将其实现了，具体如下：

```python
def recommend(self, user):
	K = self.n_sim_movie
	N = self.n_rec_movie
	rank = {}
	watched_movies = self.trainset[user]
	for movie, rating in watched_movies.items():
	for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                   key=itemgetter(1), reverse=True)[:K]:
		if related_movie in watched_movies:
			continue
		rank.setdefault(related_movie, 0)
		rank[related_movie] += similarity_factor * rating
	# 返回N部推荐的电影
	return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
```

把数据集分为训练集和测试集后，我们就可以训练并测试我们的推荐模型了，完整的代码请点击[这里](https://github.com/Lockvictor/MovieLens-RecSys/blob/master/itemcf.py)。

#### UserCF和ItemCF的综合比较

UserCF是推荐系统领域较为古老的算法，而ItemCF算法则相对较新，那么哪一种算法更适用于本文的场景呢？首先我们对比一下两种算法的区别，如下图：

![UserCF和ItemCF对比图](/images/posts/artificialintelligence/UserCFOrItemCF.png)

通过上面的对比中，我们很容易地得到结论：ItemCF算法更适用于电影推荐情景，理由如下：

1. 相对于新闻等推荐对象来说，电影更新的速度不会特别快，维护物品相似度矩阵的技术代价可以接受
2. 用户个性化需求强烈，需要发挥长尾理论的作用，即通过用户的历史记录向用户推荐一些并不热门的电影
3. ItemCF算法可以向用户提供推荐某部电影的理由，如“由于你喜欢A电影所以向你推荐B电影”

那么是不是UserCF算法就被淘汰了呢？实际上不是的，比如在新闻类的推荐中，UserCF算法就起到了很重要的作用。一是由于新闻网站中用户的兴趣不是特别细化，所以可以使用基于用户的协同过滤算法，二是从技术角度看，新闻每时每刻都在更新，维护一张如此庞大的物品相关度表在技术上很难实现。

#### 使用MapReduce实现ItemCF算法

根据上面的分析，我们得到了ItemCF算法更适用于电影推荐情景的结论，但是在实际运用中，我们可以发现，ItemCF算法的运行需要很长的时间。本文数据集的大小是23.4MB，在单台机器上运行需要几分钟的时间，这在可接受范围之内，可当我们的数据集变得庞大，比如Netflix的数据集就达2GB左右，我们在单台机器上运行ItemCF算法就变得不切实际了，这个时候我们就要用到MapReduce了！

使用MapReduce实现ItemCF算法的基本思路如下：

1. 建立物品的同现矩阵，即统计两两物品同时出现的次数
2. 建立用户对物品的评分矩阵，即每一个用户对某一物品的评分
3. 计算推荐矩阵，推荐矩阵等于同现矩阵与评分矩阵的乘积
4. 过滤用户已评分的物品项
5. 对推荐结果按推荐分值从高到低排序

首先我们需要建立同现矩阵，它需要使用两次MapReduce，分别如下：

**第一次MapReduce**

Map输入：

```
user_id | item_id | rating
```

Map的输出：

```
key: user_id
value: item_id
```

Reduce的输入：

```
key: user_id
values: item_id1, item_id2, ....
```

Reduce的输出：

```
key: item_id1, item_id1   value: 1
key: item_id1, item_id2   value: 1
key: item_id2, item_id1   value: 1
key: item_id2, item_id2   value: 1
......
```

**第二次MapReduce**

Map的输入：

```
item_id_x, item_id_y    1
```

Map的输出：

```
key: item_id_x, item_id_y
value: 1
```

Reduce的输入：

```
key: item_id_x, item_id_y
values: 1, 1, 1, ......
```

Reduce的输出：

```
key: item_id_x, item_id_y
value: sum(values)
```

这样子，我们就可以得到同现矩阵了，由于评分矩阵已有，所以将同现矩阵与评分矩阵相乘就可以得到推荐矩阵了。得到推荐矩阵后，我们需要过滤掉被推荐用户已看过的电影，具体的思路是：将评分矩阵中等于0的值改成1，大于0的改成0，得到过滤矩阵，将推荐矩阵与过滤矩阵点乘，去掉结果中值为0的元素。最后再对推荐结果按推荐分值从高到低进行排序就可以了。

### 参考资料

1. [《推荐系统实践》2.4节内容](https://book.douban.com/subject/10769749/)
2. [基于MovieLens-1M数据集实现的协同过滤算法](https://github.com/Lockvictor/MovieLens-RecSys)
3. [如何使用MapReduce实现基于物品的协同过滤](http://www.letiantian.me/2014-11-20-map-reduce-item-cf-1/)
4. [基于物品的协同过滤ItemCF的mapreduce实现](http://www.cnblogs.com/anny-1980/articles/3519555.html)