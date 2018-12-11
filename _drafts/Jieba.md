jieba 源码阅读笔记（一）

### jieba 初始化

jieba 自 0.28 版后就开始采用了延迟加载机制，jieba 在初始化的时候会创建一个`Tokenizer`实例`dt`，此实例在创建时不会构建词频字典，它定义了一个描述当前初始化情况的变量`self.initialized`，并在初始化时，设置其为`False`。因此当我们导入 jieba 时，词典并不会被加载，只有当我们使用 jieba 的具体功能的，`dt`实例才会调用`check_initialized(self)`函数，判断`initialized`变量是否为`True`，如果不是，则会调用`initialize(self, dictionary=None)`函数，实行初始化。

初始化过程主要加载词频文件，返回词频字典 FREQ（key 为词，value 为词频，可用`jieba.get_FREQ(k)`搜索对应词的词频）和词的总数目。

如果我们要手动初始化 jieba，可以手动调用`jieba.initialize()`进行初始化。

### Trie 树

根据词频字典 FREQ，jieba 可以将一个句子转换为一个有向无环图（DAG），而这其中利用到的一个重要数据结构就是 Trie 树。

get_DAG(self, sentence) 基于 Trie 树获取有向无环图
原句：我	来		到	北	京	清	华	大	学
DAG：我  来到	到	 北京 京 清华学 华大 大学 学
一个 词语的前缀一定出现在词频字典中

__cut_all 全模式
我 来到 北京 清华 清华大学 华大 大学

https://blog.csdn.net/daniel_ustc/article/details/48195287

https://zhuanlan.zhihu.com/p/25303529

https://zhuanlan.zhihu.com/p/35846232

http://midday.me/article/003023dc3f814bc493b37c50b2a9ee71

http://www.voidcn.com/article/p-bgvtrzjf-rp.html

https://github.com/fxsjy/jieba

