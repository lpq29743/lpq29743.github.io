---
layout: post
title: 计算机科学学术研究方法
categories: Blog
description: 计算机科学学术研究方法
keywords: 计算机科学, 学术研究
---

### 调研

1. 调研参考材料有综述论文（最方便的方法是在 Google Scholar 搜索“领域名称 + survey / review / tutorial / 综述”）、近一两年顶会论文的 Introduction、Related Work、引用和被引用以及知乎
2. 调研工作不仅要在课题初始时展开，也要在课题进行过程中展开，要时刻关注各大顶会相关领域的成果
3. 调研内容包括 Introduction、Datasets、Metrics 和 Related Work
4. 调研的结果应用 PPT 进行记录，尤其是 Paper 的引用
5. 主要工作的代码可以通过 Paper 中的链接、Github 以及作者的个人主页尝试查找

### 灵感

1. 引入更多信息
2. 进行更多交互

### 实验

1. 模型跑通不等于模型是对的，要检查中间变量的维度和输出
2. 做实验不要贪多，每次实验只验证一个想法。每次实验之后，必须保存代码、参数、模型、日志、实验结果（要带有模型参数、模块信息的标签）以及结果分析，要进行错误分析，要进行版本控制
3. 要确保进行多次实验得到有说服力的准确率结果，追求代码的一致性和可复现性（https://2020.emnlp.org/call-for-papers）
4. 调参方式：对于大数据集可以先对数据集进行分层采样之后进行调参尝试，对于小数据集可以网格搜索。

### 代码撰写

1. [Writing Code for NLP Research](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018)
2. 好的编码风格：写模型的时候，要给变量、常量等赋以有意义的名称，要对张量的形状做一定注释，要通过注释描述不明显的逻辑。

### 论文撰写

1. 不要等到全部做完才开始写论文，开题不久后就可以开始撰写，这是将 idea 公式化、明确化的好习惯
2. 要对论文进行版本控制，可用 overleaf 工具
3. 要围绕主要创新撰写，相关的工作和理论简写
4. 摘要部分的撰写目标就是让读者能够用一句话来总结你的论文
5. Motivation 要符合人的思考，多问自己 what 和 why
6. 常见的书写错误包括：a/an，模型名称首字母大写，缩写，权值和向量加粗，Related Work 过去式
7. 实验部分定性和定量去验证结论
8. 代码公开（https://github.com/tdurieux/anonymous_github）

### 工作推广

1. 代码、笔记公开
2. 可提供demo