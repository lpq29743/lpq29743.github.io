jieba 源码阅读笔记（一）

### jieba 初始化

jieba 自 0.28 版后就开始采用了延迟加载机制，jieba 在初始化的时候会创建一个`Tokenizer`实例`dt`，此实例在创建时不会构建词频字典，它定义了一个描述当前初始化情况的变量`self.initialized`，并在初始化时，设置其为`False`。因此当我们导入 jieba 时，词典并不会被加载，只有当我们使用 jieba 的具体功能的，`dt`实例才会调用`check_initialized(self)`函数，判断`initialized`变量是否为`True`，如果不是，则会调用`initialize(self, dictionary=None)`函数，实行初始化。

初始化过程主要加载词频文件，返回词频字典 FREQ（key 为词，value 为词频，可用`jieba.get_FREQ(k)`搜索对应词的词频）和词的总数目。

如果我们要手动初始化 jieba，可以手动调用`jieba.initialize()`进行初始化。

### Trie 树

根据词频字典 FREQ，jieba 可以将一个句子转换为一个有向无环图（DAG），而这其中利用到的一个重要数据结构就是 Trie 树（原始版本直接实现 Trie 树，现有版本借鉴思想，将前缀树存储在文件中）。

Trie 树，又叫字典树或前缀树，是一棵多叉树。Trie 树一般用来统计、排序和保存字符串，具体的应用包括词频统计、前缀匹配和自动补全。当字符串量级大、字符串过长的时候不适合用 Trie 树。Trie 树的常用操作是查找和插入，时间复杂度为 $$O(m)$$，其中 $$m$$ 是待插入/查询的字符串，删除操作很少用。

Trie 树有 3 个基本性质：

1. 根节点不包含字符，除根节点外每一个节点都只包含一个字符
2. 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串
3. 每个节点的所有子节点包含的字符都不相同

词频字典 FREQ 实际上就是以 Trie 树的思想存储的，每个词的前缀都保存在文本中，对于没有在分析文本中独立出现的前缀词，词频记为 0。

### 获取 DAG

jieba 通过调用`get_DAG(self, sentence)`来获取每个字符基于 Trie 树的有向无环图（DAG），我们以官方样例“我来到北京清华大学”来进行研讨。

`get_DAG(self, sentence)`返回一个 DAG 字典（key 为每个字符，value 为字符对应的 DAG，{0:[0, 1]} 就是一个简单的例子），对于上面的这个句子，得到的字典为`{0: [0], 1: [1, 2], 2: [2], 3: [3, 4], 4: [4], 5: [5, 6, 8], 6: [6, 7], 7: [7, 8], 8: [8]}`，其中位置 5（即“清”）的 DAG 是`[5, 6, 8]`，之所以跳过 7（即“大”），是因为“清华大”出现在字典中（表明拥有其作为前缀的词），但词频为 0（表示前缀没有单独出现在统计文本中）。

### 全模式

使用 jieba 的全模式（把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；）的案例代码如下：

```python
import jieba
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式
```

在 jieba 中，全模式实现的函数为`__cut_all`，其具体实现的思路是：

1. 遍历每一个字符`i`
2. 如果当前字符`i`的 DAG 中只有自己一个字符，判断其是否已被前面 DAG 包括，如果是跳过，如果不是，则输出
3. 如果当前字符`i`的 DAG 包括多个字符，则遍历其后面的每个字符`j`，将区间`[i, j]`内的字符合并成词输出

### 精确模式



https://blog.csdn.net/daniel_ustc/article/details/48195287

https://zhuanlan.zhihu.com/p/25303529

https://zhuanlan.zhihu.com/p/35846232

http://midday.me/article/003023dc3f814bc493b37c50b2a9ee71

http://www.voidcn.com/article/p-bgvtrzjf-rp.html

https://github.com/fxsjy/jieba

