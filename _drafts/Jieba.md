import jieba
jieba 初始化
创建了 Tokenizer 实例
Tokenizer 初始化时设置变量 initialized 为 False
等到使用的时候再调用 check_initialized() 判断是否已初始化，如果没有初始化，进行初始化
初始化过程主要加载词频文件，返回词频字典FREQ（key为词，value为词频，可用jieba.get_FREQ(k)搜索对应词的词频）和词的总数目。
可以手动调用jieba.initialize()进行初始化

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

