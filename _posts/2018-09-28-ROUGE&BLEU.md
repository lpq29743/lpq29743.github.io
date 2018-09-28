---
layout: post
title: 摘要生成评估指标：ROUGE 和 BLEU
categories: ArtificialIntelligence
description: 摘要生成评估指标：ROUGE 和 BLEU
keywords: 机器学习, 深度学习, 自然语言处理, 机器翻译, 阅读理解, 摘要生成, 评估指标, ROUGE, BLEU
---

ROUGE 和 BLEU 是机器翻译、摘要生成以及阅读理解等领域经常用到的评估指标，这篇文章我们就来认识一下这两个指标：

#### ROUGH

相关文章：ROUGE: A Package for Automatic Evaluation of Summaries (Chin-Yew Lin, 2004)

摘要：文章提出了一种面向召回率的评价方法 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）。该方法可细分为 ROUGE-N, ROUGE-L, ROUGE-W 以及 ROUGE-S 四种评价指标。

##### ROUGE-N

ROUGE-N 指标计算生成的摘要与相应的参考摘要的 n-gram 召回率，具体的公式为：

$$
ROUGE-N=\frac{\sum_{S \in {ReferenceSummaries}}\sum_{gram_n \in S}Count_{match}(gram_n)}{\sum_{S \in {ReferenceSummaries}}\sum_{gram_n \in S}Count_(gram_n)}
$$

其中分母部分计算参考摘要中 n-gram 的个数，分子部分计算参考摘要和自动摘要共有的 n-gram 的个数。

实例：

> 自动摘要：the cat was found under the bed
>
> 参考摘要：the cat was under the bed
>
> ROUGE-1 = 6/6 = 1
>
> ROUGE-2 = 4/5 = 0.8

##### ROUGE-L

ROUGE-L 指标基于两个文本单元的最长公共序列，计算 F-measure，具体公式如下：

$$
R_{LCS}=\frac{LCS(X, Y)}{m}
$$

$$
P_{LCS}=\frac{LCS(X, Y)}{n}
$$

$$
F_{LCS}=\frac{(1+\beta^2)R_{LCS}P_{LCS}}{R_{LCS}+\beta^2P_{LCS}}
$$

其中 X 为参考摘要，长度为 m；Y 为生成摘要，长度为 n；β 为精确率和召回率的比值。

实例：

> 自动摘要：police kill the gunman
>
> 参考摘要：police killed the gunman
>
> R = 3/4
>
> P = 3/4
>
> ROUGH-L = F = 3/4 = 0.75

##### ROUGE-W

ROUGE-W 指标在 ROUGE-L 的基础上进行加权计算。

> X: [A B C D E F G]
>
> Y1: [A B C D H I K]
>
> Y2: [A H B K C I D]

Y1 和 Y2 的 ROUGH-L 值都为 4/7，但明显 Y1 与参考摘要更加接近，所以作者提出了一个基于最长公共子序列的加权算法。 

##### ROUGH-S

ROUGH-S 使用了 skip-grams，在参考摘要和生成摘要进行进行匹配时，不要求 gram 之间是连续的，可跳过几个单词，如 skip-bigram，在产生 grams 时，允许最多跳过两个词。

实例：

> 例子：cat in the hat
>
> skip-bigrams：cat in, cat the, cat hat, in the, in hat, the hat

#### BLEU

相关文章：BLEU: a Method for Automatic Evaluation of Machine Translation (Kishore Papineni, 2002)

摘要：文章提出了一种评估机器翻译结果质量的方法 BLEU（Bilingual Evaluation Understudy）。

BLEU 指标先计算生成翻译与相应的参考翻译的 n-gram 精确率，具体的公式为：

$$
p_n=\frac{\sum_{C \in {Candidate}}\sum_{gram_n \in C}Count_{clip}(gram_n)}{\sum_{C' \in {Candidate}}\sum_{gram_n \in C'}Count_(gram_n)}
$$

其中分母部分计算生成翻译中 n-gram 的个数，分子部分计算参考摘要和自动摘要共有的 n-gram 的个数。

然后再计算具体的 BLEU 值：

BLEU=BP*exp(\sum_{n=1}^N w_nlogp_n)

其中 w 表示 n 元词的权重，BP 是对生成翻译文本过短的一个惩罚因子。具体计算公式如下：

$$
BP=
\left\{  
\begin{array}{**lr**}  
1 & c > r \\  
exp(1-r/c) & c\leq r \\  
\end{array}  
\right.
$$

其中 c 为生成翻译长度，r 为参考翻译长度。