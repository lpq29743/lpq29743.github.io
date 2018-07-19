---
layout: wiki
title: Computer Science Papers
categories: Blog
description: 计算机科学论文
keywords: 计算机科学论文, 深度学习论文, 自然语言处理论文
---

### 论文集

[ACL, EMNLP, COLING, SemEval 历年论文集](https://aclanthology.coli.uni-saarland.de/)

[JMLR 历年论文集](http://www.jmlr.org/papers/)

[ICCV CVPR 历年论文集](http://openaccess.thecvf.com/menu.py)

[AAAI 历年论文集](http://www.aaai.org/Library/AAAI/aaai-library.php)

[IJCAI 历年论文集](https://www.ijcai.org/proceedings/2017/)

### 自然语言处理

#### 词嵌入

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) (Tomas Mikolov, 2013, [code](https://github.com/danielfrg/word2vec), [note](https://zhuanlan.zhihu.com/p/34718114))

[GloVe: Global Vectors for Word Representation](http://www.anthology.aclweb.org/D/D14/D14-1162.pdf) (Jeffrey Pennington, EMNLP 2014, [code](https://github.com/stanfordnlp/GloVe), [note](https://zhuanlan.zhihu.com/p/34959040))

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf) (Armand Joulin, 2016, [code](https://github.com/facebookresearch/fastText))

[Enriching Word Vectors with Subword Information](https://pdfs.semanticscholar.org/e2db/a792360873aef125572812f3673b1a85d850.pdf) (Piotr Bojanowski, 2016)

#### 情感分析

##### 概述

[Opinion Mining and Sentiment Analysis](https://www.cse.iitb.ac.in/~pb/cs626-449-2009/prev-years-other-things-nlp/sentiment-analysis-opinion-mining-pang-lee-omsa-published.pdf) (Bo Pang, 2008)

[Sentiment Analysis and Opinion Mining](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.244.9480&rep=rep1&type=pdf) (Bing Liu, 2012)

##### 情感词典

[Building Large-Scale Twitter-Specific Sentiment Lexicon: A Representation Learning Approach](http://www.aclweb.org/anthology/C14-1018) (Duyu Tang, COLING 2014)

[Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora](http://www.aclweb.org/anthology/D16-1057) (William L. Hamilton, EMNLP 2016)

##### 情感词嵌入

[Learning sentiment-specific word embedding for twitter sentiment classification](http://www.anthology.aclweb.org/P/P14/P14-1146.pdf) (Duyu Tang, ACL 2014)

[SenticNet 5: Discovering Conceptual Primitives for Sentiment Analysis by Means of Context Embeddings](http://sentic.net/senticnet-5.pdf) (Erik Cambria, AAAI 2018, [code1](http://sentic.net/downloads/), [code2](https://github.com/yurimalheiros/senticnetapi))

##### 对象抽取

[Recursive neural conditional random fields for aspect-based sentiment analysis](http://www.aclweb.org/anthology/D16-1059) (Wenya Wang, EMNLP 2016, [code](https://github.com/happywwy/Recursive-Neural-Conditional-Random-Field))

[Improving Opinion Aspect Extraction Using Semantic Similarity and Aspect Associations](https://aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11973/12051) (Qian Liu, AAAI 2016)

[Unsupervised word and dependency path embeddings for aspect term extraction](https://arxiv.org/pdf/1605.07843.pdf) (Yichun Yin, IJCAI 2016)

[Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14441/14256) (Wenya Wang, AAAI 2017, [code](https://github.com/happywwy/Coupled-Multi-layer-Attentions))

[An Unsupervised Neural Attention Model for Aspect Extraction](http://www.aclweb.org/anthology/P17-1036) (Ruidan He, ACL 2017)

[Deep Multi-Task Learning for Aspect Term Extraction with Memory Interaction](http://www.aclweb.org/anthology/D17-1310) (Xin Li, EMNLP 2017)

##### 对象级情感分析

[SemEval-2014 Task 4: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S14-2004) (Maria Pontiki, SemEval 2014)

[SemEval-2015 Task 12: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S15-2082) (Maria Pontiki, SemEval 2015)

[SemEval-2016 Task 5: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S16-1002) (Maria Pontiki, SemEval 2016)

[Target-dependent twitter sentiment classification](http://www.anthology.aclweb.org/P/P11/P11-1016.pdf) (Long Jiang, ACL 2011)

[Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](http://www.aclweb.org/anthology/P14-2009) (Li Dong, ACL 2014)

[Target-dependent twitter sentiment classification with rich automatic features](http://www.ijcai.org/Proceedings/15/Papers/194.pdf) (Duy-Tin Vo, IJCAI 2015, [code](https://github.com/duytinvo/ijcai2015))

[Effective LSTMs for Target-Dependent Sentiment Classification](http://www.aclweb.org/anthology/C16-1311) (Duyu Tang, COLING 2016, [code](https://github.com/scaufengyang/TD-LSTM), [note](https://zhuanlan.zhihu.com/p/33986102))

[Attention-based LSTM for Aspect-level Sentiment Classification](http://www.aclweb.org/anthology/D16-1058) (Yequan Wang, EMNLP 2016, [code](https://github.com/scaufengyang/TD-LSTM), [note](https://zhuanlan.zhihu.com/p/34005136))

[Aspect Level Sentiment Classification with Deep Memory Network](http://www.aclweb.org/anthology/D16-1021) (Duyu Tang, EMNLP 2016, [code](https://github.com/pcgreat/mem_absa), [note](https://zhuanlan.zhihu.com/p/34033477))

[A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis](http://www.aclweb.org/anthology/D16-1103) (Sebastian Ruder, EMNLP 2016)

[Interactive Attention Networks for Aspect-Level Sentiment Classification](http://static.ijcai.org/proceedings-2017/0568.pdf) (Dehong Ma, IJCAI 2017, [code](https://github.com/lpq29743/IAN), [note](https://zhuanlan.zhihu.com/p/34041012))

[Recurrent Attention Network on Memory for Aspect Sentiment Analysis](http://www.aclweb.org/anthology/D17-1047) (Peng Chen, EMNLP 2017, [code](https://github.com/lpq29743/RAM), [note](https://zhuanlan.zhihu.com/p/34043504))

[Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16541/16152) (Yukun Ma, AAAI 2018)

[Transformation Networks for Target-Oriented Sentiment Classification](https://arxiv.org/pdf/1805.01086.pdf) (Xin Li, ACL 2018, [code](https://github.com/lixin4ever/TNet))

#### 问答系统与机器阅读理解

[DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications](https://arxiv.org/pdf/1711.05073.pdf) (Wei He, 2017, [code](https://github.com/baidu/DuReader))

[TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://aclweb.org/anthology/P17-1147) (Mandar Joshi, ACL 2017)

[ROUGE: A Package for Automatic Evaluation of Summaries](http://www.aclweb.org/anthology/W/W04/W04-1013.pdf) (Chin-Yew Lin, 2004)

[BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P/P02/P02-1040.pdf) (Kishore Papineni, 2002)

[Machine comprehension using match-lstm and answer pointer](https://arxiv.org/pdf/1608.07905.pdf) (Shuohang Wang, ICLR 2017, [code](https://github.com/MurtyShikhar/Question-Answering))

[Bidirectional attention flow for machine comprehension](https://arxiv.org/pdf/1611.01603.pdf) (Minjoon Seo, ICLR 2017, [code](https://github.com/allenai/bi-att-flow))

[Attention-over-Attention Neural Networks for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1055.pdf) (Yiming Cui, ACL 2017, [code](https://github.com/OlavHN/attention-over-attention))

[Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/pdf/1710.10723.pdf) (Christopher Clark, 2017)

[Gated Self-Matching Networks for Reading Comprehension and Question Answering](http://aclweb.org/anthology/P17-1018) (Wenhui Wang, ACL 2017, [code](https://github.com/HKUST-KnowComp/R-Net))

[R-Net: machine reading comprehension with self-matching networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) (Natural Language Computing Group, 2017, [code](https://github.com/HKUST-KnowComp/R-Net))

[QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://openreview.net/pdf?id=B14TlG-RW) (Adams Wei Yu, ICLR 2018, [code](https://github.com/minsangkim142/Fast-Reading-Comprehension))

#### 知识图谱

##### 知识图谱表示和补全

[Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) (Antoine Bordes, NIPS 2013, [code](https://github.com/thunlp/KB2E))

[Knowledge Graph Embedding by Translating on Hyperplanes](https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf) (Zhen Wang, AAAI 2014, [code](https://github.com/thunlp/KB2E))

[Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.698.8922&rep=rep1&type=pdf) (Yankai Lin, AAAI 2015, [code](https://github.com/thunlp/KB2E))

[Knowledge Graph Embedding via Dynamic Mapping Matrix](http://anthology.aclweb.org/P/P15/P15-1067.pdf) (Guoliang Ji, ACL 2015, [code](https://github.com/thunlp/OpenKE))

[TransA: An Adaptive Approach for Knowledge Graph Embedding](https://arxiv.org/pdf/1509.05490.pdf) (Han Xiao)

[Modeling Relation Paths for Representation Learning of Knowledge Bases](https://arxiv.org/pdf/1506.00379.pdf) (Yankai Lin, EMNLP 2015, [code](https://github.com/thunlp/KB2E))

[TransG : A Generative Model for Knowledge Graph Embedding](http://www.aclweb.org/anthology/P16-1219) (Han Xiao, ACL 2016, [code](https://github.com/BookmanHan/Embedding))

[Knowledge Graph Completion with Adaptive Sparse Transfer Matrix](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693) (Guoliang Ji, AAAI 2016, [code](https://github.com/thunlp/Fast-TransX))

[Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/pdf/1707.01476.pdf) (Pasquale Minervini, AAAI 2018, [code](https://github.com/TimDettmers/ConvE))

[Open-World Knowledge Graph Completion](https://arxiv.org/pdf/1711.03438.pdf) (Baoxu Shi, AAAI 2018, [code](https://github.com/bxshi/ConMask))

##### 知识图谱应用

[Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN](https://arxiv.org/ftp/arxiv/papers/1804/1804.03317.pdf) (Yingqqi Qu, ISMC 2018, [code](https://github.com/quyingqi/kbqa-ar-smcnn))

[Learning beyond datasets: Knowledge Graph Augmented Neural Networks for Natural language Processing](https://arxiv.org/pdf/1802.05930.pdf) (Annervaz K M, HAACL HLT 2018)

[Variational Reasoning for Question Answering with Knowledge Graph](https://arxiv.org/pdf/1709.04071.pdf) (Yuyu Zhang, AAAI 2018)

### 机器学习

#### 模型

[An Introduction to Conditional Random Fields](https://arxiv.org/pdf/1011.4088.pdf) (Charles Sutton, 2010)

### 深度学习

#### 模型

[A Critical Review of Recurrent Neural Networks for Sequence Learning](http://pdfs.semanticscholar.org/0651/b333c2669227b0cc42de403268a4546ece70.pdf) (Zachary C. Lipton, 2015)

[On the properties of neural machine translation: Encoder–Decoder approaches](https://arxiv.org/pdf/1409.1259.pdf) (Kyunghyun Cho, 2014)

[Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/pdf/1406.1078.pdf) (Kyunghyun Cho, EMNLP 2014)

[Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf) (Alex Graves, 2014, [code](https://github.com/carpedm20/NTM-tensorflow))

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (Dzmitry Bahdanau, ICLR 2015)

[Memory Networks](https://arxiv.org/pdf/1410.3916v11.pdf) (Jason Weston, ICLR 2015)

[End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf) (Sainbayar Sukhbaatar, 2015)

[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://www.thespermwhale.com/jaseweston/ram/papers/paper_21.pdf) (Ankit Kumar, 2015)

[Gated End-to-End Memory Networks](http://www.aclweb.org/anthology/E/E17/E17-1001.pdf) (Fei Liu, 2016)

[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) (Ashish Vaswani, 2017, [code1](https://github.com/jadore801120/attention-is-all-you-need-pytorch), [code2](https://github.com/Kyubyong/transformer), [code3](https://github.com/bojone/attention))

#### 优化器

[On the Momentum Term in Gradient Descent Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf) (Ning Qian, 1999)

[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (John Duchi, JMLR 2011)

[ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701.pdf) (Matthew D. Zeiler, 2012)

[ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf) (Diederik P. Kingma, 2015)

#### 参数初始化

[Understanding the difficulty of training deep feedforward neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf) (Xavier Glorot, JMLR 2010)

[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) (Kaiming He, ICCV 2015)

#### 损失函数

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) (Florian Schroff, CVPR 2015, [code](https://github.com/davidsandberg/facenet))

[A Discriminative Feature Learning Approach for Deep Face Recognition](http://www.eccv2016.org/files/posters/P-3B-20.pdf) (Yandong Wen, ECCV 2016, [code](https://github.com/pangyupo/mxnet_center_loss))

[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) (Weiyang Liu, ICML 2016, [code](https://github.com/wy1iu/LargeMargin_Softmax_Loss))

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf) (Weiyang Liu, ICML 2017, [code](https://github.com/wy1iu/sphereface))

[Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) (Feng Wang, 2018, [code](https://github.com/Joker316701882/Additive-Margin-Softmax))

#### 激活函数

[Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](https://arxiv.org/pdf/1603.05201v2.pdf) (Wenling Shang, ICML 2016)

### 推荐系统

#### 新闻推荐

[Google News Personalization: Scalable Online Collaborative Filtering](http://wwwconference.org/www2007/papers/paper570.pdf) (Abhinandan Das, WWW 2007)

[Personalized News Recommendation Based on Click Behavior](http://www.cs.northwestern.edu/~jli156/IUI224-liu.pdf) (Jiahui Liu, 2010)

[Personalized Recommendation on Dynamic Content Using Predictive Bilinear Models](http://wwwconference.org/www2009/proceedings/pdf/p691.pdf) (Wei Chu, WWW 2009)

[A Contextual-Bandit Approach to Personalized News Article Recommendation](http://wwwconference.org/proceedings/www2010/www/p661.pdf) (Lihong Li, WWW 2010)

[A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf) (Ali Elkahky, WWW 2015)

### 工具

[HemI: A Toolkit for Illustrating Heatmaps](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0111988&type=printable) (Wankun Deng, 2014, [code](http://hemi.biocuckoo.org/))

[LIBSVM: A library for support vector machines](http://www.csie.ntu.edu.tw/~cjlin/libsvm) (Chih-Jen Lin, 2011)
