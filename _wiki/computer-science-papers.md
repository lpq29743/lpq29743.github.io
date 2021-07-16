---
layout: wiki
title: Computer Science Papers
categories: Blog
description: omputer Science Papers
keywords: omputer Science Papers, 计算机科学论文, 深度学习论文, 自然语言处理论文
---

### Paper List

[ACL, EMNLP, COLING, SemEval Paper List](https://aclanthology.coli.uni-saarland.de/)

[JMLR Paper List](http://www.jmlr.org/papers/)

[ICCV CVPR Paper List](http://openaccess.thecvf.com/menu.py)

[AAAI Paper List](http://www.aaai.org/Library/AAAI/aaai-library.php)

[IJCAI Paper List](https://www.ijcai.org/proceedings/2017/)

### Natural Language Processing

#### Pretrained Language Model

[A Primer in BERTology: What We Know About How BERT Works](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00349) (Anna Rogers, TACL 2020)

[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) (Xipeng Qiu, 2020)

##### Word Embedding

[A Neural Probabilistic Language Model](http://www.iro.umontreal.ca/~vincentp/Publications/lm_jmlr.pdf) (Yoshua Bengio, JMLR 2003, [note](https://zhuanlan.zhihu.com/p/21101055))

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) (Tomas Mikolov, 2013, [code](https://github.com/danielfrg/word2vec), [note](https://zhuanlan.zhihu.com/p/34718114))

[GloVe: Global Vectors for Word Representation](http://www.anthology.aclweb.org/D/D14/D14-1162.pdf) (Jeffrey Pennington, EMNLP 2014, [code](https://github.com/stanfordnlp/GloVe), [note](https://zhuanlan.zhihu.com/p/34959040))

[Character-Aware Neural Language Models](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017) (Yoon Kim, AAAI 2016, [note](https://zhuanlan.zhihu.com/p/21242454))

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf) (Armand Joulin, 2016, [code](https://github.com/facebookresearch/fastText), [note](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650716942&idx=3&sn=0d48c0218131de502ac5e2ef9b700967))

[Enriching Word Vectors with Subword Information](https://pdfs.semanticscholar.org/e2db/a792360873aef125572812f3673b1a85d850.pdf) (Piotr Bojanowski, 2016, [note](https://github.com/xwzhong/papernote/blob/master/embedding/Enriching%20Word%20Vectors%20with%20Subword%20Information.md))

[Advances in Pre-Training Distributed Word Representations](https://arxiv.org/pdf/1712.09405.pdf) (Tomas Mikolov, 2017)

[Learning Word Vectors for 157 Languages](https://arxiv.org/pdf/1802.06893.pdf) (Edouard Grave, LREC 2018)

[Learning Chinese Word Embeddings from Stroke, Structure and Pinyin of Characters](https://dl.acm.org/doi/10.1145/3357384.3358005) (Yun Zhang, CIKM 2019)

[Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/pdf/1901.10125.pdf) (Yuxian Meng, NeuIPS 2019, [code](https://github.com/ShannonAI/glyce), [note](https://zhuanlan.zhihu.com/p/55967737))

##### Sentence/Paragraph/Document Embedding

[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053.pdf) (Quoc Le, ICML 2014, [code](https://github.com/jhlau/doc2vec), [note](https://blog.acolyer.org/2016/06/01/distributed-representations-of-sentences-and-documents/))

[Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf) (Hamid Palangi, TASLP 2017, [code](https://github.com/zhaosm/dssm-lstm))

[A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf) (Zhouhan Lin, ICLR 2017, [code](https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding), [note](https://www.sohu.com/a/130767150_505880))

[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf) (Daniel Cer, 2018, [code](https://tfhub.dev/google/universal-sentence-encoder/1), [note](https://zhuanlan.zhihu.com/p/35174235))

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf) (Nils Reimers, EMNLP 2019)

##### Contextual Language Model

[Semi-supervised Sequence Learning](https://arxiv.org/pdf/1511.01432.pdf) (Andrew M. Dai, NeuIPS 2015, [note](https://zhuanlan.zhihu.com/p/21313501))

[Semi-supervised Sequence Tagging with Bidirectional Language Models](https://arxiv.org/pdf/1705.00108.pdf) (Matthew E. Peters, ACL 2017, [note](https://zhuanlan.zhihu.com/p/105879581))

[Learned in Translation: Contextualized Word Vectors](https://arxiv.org/pdf/1708.00107.pdf) (Bryan McCann, NeuIPS 2017, [code](https://github.com/salesforce/cove), [note](https://www.sohu.com/a/162634620_610300))

[Deep Contextualized Word Representations](https://arxiv.org/pdf/1802.05365.pdf) (Matthew E. Peters, NAACL 2018, [code](https://github.com/allenai/allennlp), [note](https://zhuanlan.zhihu.com/p/38254332))

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf) (Jeremy Howard, ACL 2018, [code](http://nlp.fast.ai/category/classification.html), [note](https://zhuanlan.zhihu.com/p/61590026))

[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Alec Radford, 2018, [code](https://github.com/openai/finetune-transformer-lm), [note](https://zhuanlan.zhihu.com/p/54754171))

[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf) (Mandar Joshi, TACL 2019, [note](https://zhuanlan.zhihu.com/p/75893972))

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Jacob Devlin, NAACL 2019, [code](https://github.com/google-research/bert), [note](https://zhuanlan.zhihu.com/p/46652512))

[Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/pdf/1903.07785.pdf) (Alexei Baevski, EMNLP 2019)

[Revealing the Dark Secrets of BERT](https://arxiv.org/pdf/1908.08593.pdf) (Olga Kovaleva, EMNLP 2019, [note](https://zhuanlan.zhihu.com/p/117645185))

[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/pdf/1905.03197.pdf) (Li Dong, NeuIPS 2019, [note](https://zhuanlan.zhihu.com/p/68755034))

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) (Zhilin Yang, NeuIPS 2019, [code](https://github.com/zihangdai/xlnet), [note](https://zhuanlan.zhihu.com/p/70257427))

[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf) (Yiming Cui, 2019, [code](https://github.com/ymcui/Chinese-BERT-wwm), [note](https://zhuanlan.zhihu.com/p/96792453))

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) (Yinhan Liu, 2019, [note](https://zhuanlan.zhihu.com/p/82804993))

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf) (Mohammad Shoeybi, 2019, [code](https://github.com/NVIDIA/Megatron-LM), [note](https://www.infoq.cn/article/Ex_tDlV5VoMzLKpOObAf))

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Alec Radford, 2019, [code](https://github.com/openai/gpt-2), [note](https://zhuanlan.zhihu.com/p/79714797))

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) (Colin Raffel, 2019, [code](https://github.com/google-research/text-to-text-transfer-transformer), [note](https://zhuanlan.zhihu.com/p/88377084))

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf) (Zhenzhong Lan, ICLR 2020, [note](https://www.zhihu.com/question/347898375/answer/863537122))

[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf) (Kevin Clark, ICLR 2020, [code](https://github.com/google-research/electra), [note](https://zhuanlan.zhihu.com/p/89763176))

[Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space](https://arxiv.org/pdf/2004.04092.pdf) (Chunyuan Li, EMNLP 2020, [code](https://github.com/ChunyuanLI/Optimus), [note](https://zhuanlan.zhihu.com/p/143517152))

[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) (Tom B. Brown, 2020, [code](https://github.com/openai/gpt-3), [note](https://zhuanlan.zhihu.com/p/144764546))

[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) (William Fedus, 2020)

[Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/pdf/2105.03322.pdf) (Yi Tay, ACL 2021)

##### Knowledge-Enriched Language Model

[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf) (Zhengyan Zhang, ACL 2019, [code](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE), [note](https://zhuanlan.zhihu.com/p/87008569))

[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf) (Yu Sun, AAAI 2020, [note](https://zhuanlan.zhihu.com/p/76125042))

[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf) (Yu Sun, 2021)

[Semantics-aware BERT for Language Understanding](https://arxiv.org/pdf/1909.02209.pdf) (Zhuosheng Zhang, AAAI 2020, [note](https://zhuanlan.zhihu.com/p/115457267))

[K-ADAPTER: Infusing Knowledge into Pre-Trained Models with Adapters](https://arxiv.org/pdf/2002.01808.pdf) (Ruize Wang, 2020)

##### Compressed Language Model

[Fine-tune BERT with Sparse Self-Attention Mechanism](https://www.aclweb.org/anthology/D19-1361.pdf) (Baiyun Cui, EMNLP 2019)

[DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter](https://arxiv.org/pdf/1910.01108.pdf) (Victor Sahn, 2019, [code](https://github.com/huggingface/transformers/tree/master/examples/distillation), [note](https://zhuanlan.zhihu.com/p/89522799))

[AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search](https://arxiv.org/pdf/2001.04246.pdf) (Daoyuan Chen, IJCAI 2020, [note](https://zhuanlan.zhihu.com/p/144549207))

[FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://www.aclweb.org/anthology/2020.acl-main.537.pdf) (Weijie Liu, ACL 2020, [note](https://zhuanlan.zhihu.com/p/127869267))

#### Text Classification

[One-Class SVMs for Document Classification](http://www.jmlr.org/papers/v2/manevitz01a.html) (Larry M. Manevitz, JMLR 2001)

[Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) (Yoon Kim, EMNLP 2014, [code](https://github.com/brightmart/text_classification), [note](https://zhuanlan.zhihu.com/p/21242710))

[Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf) (Siwei Lai, AAAI 2015, [code](https://github.com/brightmart/text_classification), [note](https://zhuanlan.zhihu.com/p/21253220))

[Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://www.aclweb.org/anthology/N15-1011) (Rie Johnson, NAACL 2015, [code](https://github.com/riejohnson/ConText))

[Deep Unordered Composition Rivals Syntactic Methods for Text Classification](http://www.aclweb.org/anthology/P15-1162) (Mohit Iyyer, ACL 2015, [code1](https://github.com/lpq29743/text_classification/blob/master/models/dl_models/dan.py), [code2](https://github.com/miyyer/dan))

[Discriminative Neural Sentence Modeling by Tree-Based Convolution](https://aclweb.org/anthology/D15-1279) (Lili Mou, EMNLP 2015)

[Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf) (Xiang Zhang, NeuIPS 2015, [code](https://github.com/dongjun-Lee/text-classification-models-tf), [note](https://zhuanlan.zhihu.com/p/51698513))

[Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding](https://papers.nips.cc/paper/5849-semi-supervised-convolutional-neural-networks-for-text-categorization-via-region-embedding.pdf) (Rie Johnson, NeuIPS 2015, [code](https://github.com/riejohnson/ConText))

[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) (Zichao Yang, NAACL 2016, [code](https://github.com/brightmart/text_classification), [note](https://zhuanlan.zhihu.com/p/26892711))

[Supervised and Semi-Supervised Text Categorization using LSTM for Region Embeddings](https://arxiv.org/pdf/1602.02373.pdf) (Rie Johnson, ICML 2016, [code](https://github.com/riejohnson/ConText))

[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf) (Pengfei Liu, IJCAI 2016, [code](https://github.com/brightmart/text_classification), [note](https://zhuanlan.zhihu.com/p/27562717))

[Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](https://www.aclweb.org/anthology/C16-1329) (Peng Zhou, COLING 2016)

[Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://arxiv.org/pdf/1602.00367.pdf) (Yijun Xiao, 2016)

[A Hybrid CNN-RNN Alignment Model for Phrase-Aware Sentence Classification](http://www.aclweb.org/anthology/E17-2071) (Shiou Tian Hsu, EACL 2017, [note](https://zhuanlan.zhihu.com/p/35008282))

[Very Deep Convolutional Networks for Text Classification](http://cn.arxiv.org/pdf/1606.01781) (Alexis Conneau, EACL 2017, [code](https://github.com/zonetrooper32/VDCNN), [note](https://zhuanlan.zhihu.com/p/39593725))

[Adversarial Multi-task Learning for Text Classification](https://www.aclweb.org/anthology/P17-1001) (Pengfei Liu, ACL 2017, [code](https://github.com/FrankWork/fudan_mtl_reviews), [note](https://zhuanlan.zhihu.com/p/31653852))

[Deep Pyramid Convolutional Neural Networks for Text Categorization](https://www.aclweb.org/anthology/P17-1052) (Rie Johnson, ACL 2017, [code](https://github.com/riejohnson/ConText), [note](https://zhuanlan.zhihu.com/p/56189443))

[Multi-Task Label Embedding for Text Classification](https://aclweb.org/anthology/D18-1484) (Honglun Zhang, EMNLP 2017, [note](https://zhuanlan.zhihu.com/p/37669263))

[Learning Structured Representation for Text Classification via Reinforcement Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/zhang.pdf) (Tianyang Zhang, AAAI 2018, [code](https://github.com/keavil/AAAI18-code), [note](https://zhuanlan.zhihu.com/p/36836402))

[Translations as Additional Contexts for Sentence Classification](https://arxiv.org/pdf/1806.05516.pdf) (Reinald Kim Amplayo, IJCAI 2018, [code](https://github.com/rktamplayo/MCFA))

[Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms](https://arxiv.org/pdf/1805.09843.pdf) (Dinghan Shen, ACL 2018, [code](https://github.com/dinghanshen/SWEM), [note](https://zhuanlan.zhihu.com/p/38056365))

[Joint Embedding of Words and Labels for Text Classification](https://www.aclweb.org/anthology/P18-1216) (Guoyin Yang, ACL 2018, [code](https://github.com/guoyinwang/LEAM), [note](https://zhuanlan.zhihu.com/p/54734708))

[Marrying Up Regular Expressions with Neural Networks: A Case Study for Spoken Language Understanding](https://arxiv.org/pdf/1805.05588.pdf) (Bingfeng Luo, ACL 2018, [note](https://zhuanlan.zhihu.com/p/43815470))

[Graph Convolutional Networks for Text Classification](https://arxiv.org/pdf/1809.05679v1.pdf) (Liang Yao, AAAI 2019, [code](https://github.com/yao8839836/text_gcn), [note](https://zhuanlan.zhihu.com/p/75708556))

[Topics to Avoid: Demoting Latent Confounds in Text Classification](https://arxiv.org/pdf/1909.00453.pdf) (Sachin Kumar, EMNLP 2019)

[DocBERT: BERT for Document Classification](https://arxiv.org/pdf/1904.08398.pdf) (Ashutosh Adhikari, 2019, [code](https://github.com/castorini/hedwig))

[Text Classification Using Label Names Only: A Language Model Self-Training Approach](https://arxiv.org/pdf/2010.07245.pdf) (Yu Meng, EMNLP 2020, [code](https://github.com/yumeng5/LOTClass))

[Inductive Topic Variational Graph Auto-Encoder for Text Classification](https://www.aclweb.org/anthology/2021.naacl-main.333.pdf) (Qianqian Xie, NAACL 2021)

##### Multi-Label Text Classification

[Semantic-Unit-Based Dilated Convolution for Multi-Label Text Classification](https://arxiv.org/pdf/1808.08561.pdf) (Junyang Lin, EMNLP 2018, [code](https://github.com/lancopku/SU4MLC), [note](https://zhuanlan.zhihu.com/p/59546989))

[SGM: Sequence Generation Model for Multi-Label Classification](http://aclweb.org/anthology/C18-1330) (Pengcheng Yang, COLING 2018, [code](https://github.com/lancopku/SGM), [note](https://zhuanlan.zhihu.com/p/53910836))

[A Deep Reinforced Sequence-to-Set Model for Multi-Label Classification](https://www.aclweb.org/anthology/P19-1518.pdf) (Pengcheng Yang, ACL 2019, [code](https://github.com/lancopku/Seq2Set), [note](https://blog.csdn.net/MaybeForever/article/details/102822057))

[AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](http://papers.nips.cc/paper/8817-attentionxml-label-tree-based-attention-aware-deep-model-for-high-performance-extreme-multi-label-text-classification.pdf) (Ronghui You, NeuIPS 2019, [code](https://github.com/yourh/AttentionXML), [note](https://zhuanlan.zhihu.com/p/96759318))

#### Text Matching

[Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://dl.acm.org/doi/abs/10.1145/2505515.2505665) (Po-Sen Huang, CIKM 2013, [code](https://github.com/liaha/dssm), [note](https://zhuanlan.zhihu.com/p/53326791))

[Learning Semantic Representations Using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf) (Yelong Shen, WWW 2014)

[A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf) (Yelong Shen, CIKM 2014, [code](https://github.com/airalcorn2/Deep-Semantic-Similarity-Model), [note](https://zhuanlan.zhihu.com/p/32915377))

[Convolutional Neural Network Architectures for Matching Natural Language Sentences](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/hu-etal-nips2014.pdf) (Baotian Hu, NeuIPS 2014, [code](https://github.com/ddddwy/ARCII-for-Matching-Natural-Language-Sentences))

[Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf) (Aliaksei Severyn, SIGIR 2015, [code](https://github.com/zhangzibin/PairCNN-Ranking), [note](https://zhuanlan.zhihu.com/p/32915377))

[A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations](https://arxiv.org/pdf/1511.08277.pdf) (Shengxian Wan, 2015, [code](https://github.com/coderbyr/MV-LSTM), [note](https://zhuanlan.zhihu.com/p/40741576))

[ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf) (Wenpeng Yin, TACL 2016, [code](https://github.com/shamalwinchurkar/question-classification?utm_source=catalyzex.com))

[Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf) (Liang Pang, AAAI 2016, [code](https://github.com/ddddwy/MatchPyramid-for-semantic-matching), [note](https://zhuanlan.zhihu.com/p/40741576))

[Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](http://www.aclweb.org/anthology/N16-1108) (Hua He, NAACL 2016, [code](https://github.com/lanwuwei/Subword-PWIM))

[Improved Representation Learning for Question Answer Matching](http://www.aclweb.org/anthology/P16-1044) (Ming Tan, ACL 2016, [code](https://github.com/person-lee/qa_lstm), [note](https://zhuanlan.zhihu.com/p/23163137))

[A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/pdf/1711.08611.pdf) (Jiafeng Guo, CIKM 2016, [code](https://github.com/sebastian-hofstaetter/neural-ranking-drmm), [note](https://zhuanlan.zhihu.com/p/94195125))

[aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240) (Liu Yang, CIKM 2016, [code](https://github.com/yangliuy/aNMM-CIKM16))

[A Compare-Aggregate Model for Matching Text Sequences](https://arxiv.org/pdf/1611.01747.pdf) (Shuohang Wang, 2016, [code](https://github.com/pcgreat/SeqMatchSeq), [note](https://zhuanlan.zhihu.com/p/27805225))

[Learning to Match using Local and Distributed Representations of Text for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) (Bhaskar Mitra, WWW 2017, [note](https://www.jianshu.com/p/ab24387a076b))

[End-to-End Neural Ad-hoc Ranking with Kernel Pooling](http://delivery.acm.org/10.1145/3090000/3080809/p55-xiong.pdf?ip=218.19.145.8&id=3080809&acc=CHORUS&key=BF85BBA5741FDC6E%2E3D07CFA6C3F555EA%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1545743485_1f9609809da82437ccf634dc7f881b4b) (Chenyan Xiong, SIGIR 2017, [code](https://github.com/AdeDZY/K-NRM), [note](https://blog.csdn.net/SrdLaplace/article/details/86481422))

[Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf) (Zhiguo Wang, IJCAI 2017, [code](https://github.com/zhiguowang/BiMPM), [note](https://zhuanlan.zhihu.com/p/50184415))

[Sentence Similarity Learning by Lexical Decomposition and Composition](https://www.aclweb.org/anthology/C/C16/C16-1127.pdf) (Zhiguo Wang, COLING 2017, [code](https://github.com/mcrisc/lexdecomp), [note](http://octopuscoder.github.io/2017/08/18/%E8%AE%BA%E6%96%87%E7%AE%80%E8%AF%BB-Sentence-Similarity-Learning-by-Lexical-Decomposition-and-Composition/))

[Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://delivery.acm.org/10.1145/3160000/3159659/p126-dai.pdf?ip=218.19.145.8&id=3159659&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E3D07CFA6C3F555EA%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1545800172_2be1e8f46ecb3f388bf3ceab8566848e) (Zhuyun Dai, WSDM 2018, [code](https://github.com/thunlp/EntityDuetNeuralRanking), [note](https://blog.csdn.net/SrdLaplace/article/details/86481422))

[Deep Relevance Ranking Using Enhanced Document-Query Interactions](http://www2.aueb.gr/users/ion/docs/emnlp2018.pdf) (Ryan McDonald, EMNLP 2018, [code](https://github.com/nlpaueb/deep-relevance-ranking), [note](https://zhuanlan.zhihu.com/p/46755219))

[Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information](https://arxiv.org/pdf/1805.11360.pdf) (Seonhoon Kim, AAAI 2019, [note](https://zhuanlan.zhihu.com/p/47948866))

#### Natural Language Inference

[A Large Annotated Corpus for Learning Natural Language Inference](https://arxiv.org/pdf/1508.05326.pdf) (Samuel R. Bowman, EMNLP 2015, [code](https://nlp.stanford.edu/projects/snli/), [note](https://blog.eson.org/pub/a1c27ad7/))

[Natural Language Inference by Tree-Based Convolution and Heuristic Matching](https://arxiv.org/pdf/1512.08422.pdf) (Lili Mou, ACL 2016, [note](https://www.paperweekly.site/papers/notes/344))

[A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/pdf/1606.01933.pdf) (Ankur P. Parikh, EMNLP 2016, [code](https://github.com/harvardnlp/decomp-attn), [note](https://zhuanlan.zhihu.com/p/26237357))

[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf) (Chen Qian, ACL 2017, [code](https://github.com/coetaur0/ESIM), [note](https://zhuanlan.zhihu.com/p/47580077))

[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364.pdf) (Alexis Conneau, EMNLP 2017, [code](https://github.com/facebookresearch/InferSent), [note](https://blog.csdn.net/sinat_31188625/article/details/77992960))

[Natural Language Inference over Interaction Space](https://arxiv.org/pdf/1709.04348.pdf) (Yichen Gong, ICLR 2018, [code](https://github.com/YichenGong/Densely-Interactive-Inference-Network), [note](https://www.cnblogs.com/databingo/p/9311892.html))

[Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference](https://aclweb.org/anthology/P18-1091) (Boyuan Pan, ACL 2018, [code](https://github.com/ZJULearning/DMP), [note](https://zhuanlan.zhihu.com/p/37899900))

[Neural Natural Language Inference Models Enhanced with External Knowledge](https://arxiv.org/pdf/1711.04289.pdf) (Chen Qian, ACL 2018, [code](https://github.com/feifengwhu/NLP_External_knowledge))

[Improving Natural Language Inference Using External Knowledge in the Science Questions Domain](https://arxiv.org/pdf/1809.05724.pdf) (Xiaoyan Wang, 2018, [note](https://zhuanlan.zhihu.com/p/77646912))

[Gaussian Transformer: A Lightweight Approach for Natural Language Inference](https://www.aaai.org/ojs/index.php/AAAI/article/view/4614) (Maosheng Guo, AAAI 2019, [code](https://github.com/lzy1732008/GaussionTransformer), [note](https://zhuanlan.zhihu.com/p/75411024))

[Are Natural Language Inference Models IMPPRESsive? Learning IMPlicature and PRESupposition](https://arxiv.org/pdf/2004.03066.pdf) (Paloma Jeretic, ACL 2020)

[Do Neural Models Learn Systematicity of Monotonicity Inference in Natural Language?](https://arxiv.org/pdf/2004.14839.pdf) (Hitomi Yanaka, ACL 2020)

[Uncertain Natural Language Inference](https://www.aclweb.org/anthology/2020.acl-main.774.pdf) (Tongfei Chen, ACL 2020)

#### Text Summarization

[SEQ3 : Differentiable Sequence-to-Sequence-to-Sequence Autoencoder for Unsupervised Abstractive Sentence Compression](https://arxiv.org/pdf/1904.03651.pdf) (Christos Baziotis, NAACL 2019)

[Answers Unite! Unsupervised Metrics for Reinforced Summarization Models](https://arxiv.org/pdf/1909.01610.pdf) (Thomas Scialom, EMNLP 2019, [code](https://github.com/recitalAI/summa-qa?utm_source=catalyzex.com))

[Better Rewards Yield Better Summaries: Learning to Summarise Without References](https://arxiv.org/pdf/1909.01214.pdf) (Florian Bohm, EMNLP 2019, [code](https://github.com/yg211/summary-reward-no-reference?utm_source=catalyzex.com))

[Neural Text Summarization: A Critical Evaluation](https://arxiv.org/pdf/1908.08960.pdf) (Wojciech Kryscinski, EMNLP 2019)

[Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) (Yang Liu, EMNLP 2019, [code](https://github.com/nlpyang/PreSumm), [note](https://zhuanlan.zhihu.com/p/88953532))

[What Have We Achieved on Text Summarization?](https://arxiv.org/pdf/2010.04529.pdf) (Dandan Huang, EMNLP 2020)

[Re-evaluating Evaluation in Text Summarization](https://arxiv.org/pdf/2010.07100.pdf) (Manik Bhandari, EMNLP 2020, [code](https://github.com/neulab/REALSumm?utm_source=catalyzex.com))

[Unsupervised Reference-Free Summary Quality Evaluation via Contrastive Learning](https://arxiv.org/pdf/2010.01781.pdf) (Hanlu Wu, EMNLP 2020, [code](https://github.com/whl97/LS-Score?utm_source=catalyzex.com))

[The Style-Content Duality of Attractiveness: Learning to Write Eye-Catching Headlines via Disentanglement](https://arxiv.org/pdf/2012.07419.pdf) (Mingzhe Li, AAAI 2021)

##### Extractive Summarization

[The Automatic Creation of Literature Abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf) (H. P. Luhn, 1958, [code](https://github.com/miso-belica/sumy))

[New Methods in Automatic Extracting](http://courses.ischool.berkeley.edu/i256/f06/papers/edmonson69.pdf) (H. P. Edmundson, 1969, [code](https://github.com/miso-belica/sumy))

[TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) (Rada Mihalcea, EMNLP 2004, [code](https://github.com/miso-belica/sumy), [note](https://zhuanlan.zhihu.com/p/55270310))

[Using Latent Semantic Analysis in Text Summarization and Summary Evaluation](http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf) (Josef Steinberger, 2004, [code](https://github.com/miso-belica/sumy))

[LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://arxiv.org/pdf/1109.2128.pdf) (Gunes Erkan, 2004, [code](https://github.com/miso-belica/sumy))

[Beyond SumBasic: Task-Focused Summarization with Sentence Simplification and Lexical Expansion](http://www.cis.upenn.edu/~nenkova/papers/ipm.pdf) (Lucy Vanderwende, 2007, [code](https://github.com/miso-belica/sumy))

[SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents](https://ojs.aaai.org/index.php/AAAI/article/view/10958) (Ramesh Nallapati, AAAI 2017)

[Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf) (Shashi Narayan, NAACL 2018, [code](https://github.com/EdinburghNLP/Refresh?utm_source=catalyzex.com))

[Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies](https://arxiv.org/pdf/1804.11283.pdf) (Max Grusky, NAACL 2018, [code](https://github.com/SumUpAnalytics/goldsum?utm_source=catalyzex.com))

[BanditSum: Extractive Summarization as a Contextual Bandit](https://arxiv.org/pdf/1809.09672.pdf) (Yue Dong, EMNLP 2018, [code](https://github.com/yuedongP/BanditSum?utm_source=catalyzex.com))

[Neural Latent Extractive Document Summarization](https://arxiv.org/pdf/1808.07187.pdf) (Xingxing Zhang, EMNLP 2018)

[Guiding Extractive Summarization with Question-Answering Rewards](https://arxiv.org/pdf/1904.02321.pdf) (Kristjan Arumae, NAACL 2019, [code](https://github.com/ucfnlp/summ_qa_rewards?utm_source=catalyzex.com))

[Single Document Summarization as Tree Induction](https://www.aclweb.org/anthology/N19-1173.pdf) (Yang Liu, NAACL 2019)

[HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://www.aclweb.org/anthology/P19-1499.pdf) (Xingxing Zhang, ACL 2019, [note](https://zhuanlan.zhihu.com/p/93598582))

[Searching for Effective Neural Extractive Summarization: What Works and What’s Next](https://arxiv.org/pdf/1907.03491.pdf) (Ming Zhong, ACL 2019, [code](https://github.com/maszhongming/Effective_Extractive_Summarization?utm_source=catalyzex.com))

[Neural Extractive Text Summarization with Syntactic Compression](https://arxiv.org/pdf/1902.00863.pdf) (Jiacheng Xu, EMNLP 2019, [code](https://github.com/jiacheng-xu/neu-compression-sum?utm_source=catalyzex.com))

[Extractive Summarization as Text Matching](https://arxiv.org/pdf/2004.08795.pdf) (MIng Zhong, ACL 2020, [code](https://github.com/maszhongming/MatchSum), [note](https://zhuanlan.zhihu.com/p/138351484))

[Discourse-Aware Neural Extractive Text Summarization](https://arxiv.org/pdf/1910.14142.pdf) (Jiacheng Xu, ACL 2020, [code](https://github.com/jiacheng-xu/DiscoBERT), [note](https://procjx.github.io/2019/11/02/%E3%80%90%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%E3%80%91Discourse-Aware-Neural-Extractive-Model-for-Text-Summarization/))

[Heterogeneous Graph Neural Networks for Extractive Document Summarization](https://arxiv.org/pdf/2004.12393.pdf) (Danqing Wang, ACL 2020, [code](https://github.com/brxx122/HeterSumGraph), [note](https://zhuanlan.zhihu.com/p/138600416))

##### Abstractive Summarization

[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/pdf/1509.00685.pdf) (Alexander M. Rush, EMNLP 2015, [code](https://github.com/Ganeshpadmanaban/Neural-Attention-Model-Abstractive-Summarization?utm_source=catalyzex.com))

[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://arxiv.org/pdf/1602.06023.pdf) (Ramesh Nallapati, CoNLL 2016, [note](https://zhuanlan.zhihu.com/p/21388527))

[Abstractive Document Summarization with a Graph-Based Attentional Neural Model](https://www.aclweb.org/anthology/P17-1108.pdf) (Jiwei Tan, ACL 2017)

[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf) (Abigail See, ACL 2017)

[A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/pdf/1705.04304.pdf) (Romain Paulus, 2017, [code](https://github.com/oceanypt/A-DEEP-REINFORCED-MODEL-FOR-ABSTRACTIVE-SUMMARIZATION), [note](https://zhuanlan.zhihu.com/p/59510696))

[Deep Communicating Agents for Abstractive Summarization](https://arxiv.org/pdf/1803.10357.pdf) (Asli Celikyilmaz, NAACL 2018, [code](https://github.com/theDoctor2013/DCA-AbstractiveSummarization?utm_source=catalyzex.com))

[A Reinforced Topic-Aware Convolutional Sequence-to-Sequence Model for Abstractive Text Summarization](https://www.ijcai.org/proceedings/2018/0619.pdf) (Li Wang, IJCAI 2018, [note](https://blog.csdn.net/imsuhxz/article/details/82655811))

[Controllable Abstractive Summarization](https://aclweb.org/anthology/W18-2706) (Angela Fan, ACL 2018)

[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/pdf/1805.11080.pdf) (Yen-Chun Chen, ACL 2018, [code](https://github.com/ChenRocks/fast_abs_rl?utm_source=catalyzex.com))

[Bottom-Up Abstractive Summarization](https://arxiv.org/pdf/1808.10792.pdf) (Sebastian Gehrmann, EMNLP 2018, [code](https://github.com/sebastianGehrmann/bottom-up-summary?utm_source=catalyzex.com))

[Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclweb.org/anthology/D18-1206) (Shashi Narayan, EMNLP 2018, [code](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Topic-ConvS2S), [note](https://zhuanlan.zhihu.com/p/92994889))

[Abstractive Summarization: A Survey of the State of the Art](https://ojs.aaai.org//index.php/AAAI/article/view/5056) (Hui Lin, AAAI 2019)

[Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://arxiv.org/pdf/1811.00783.pdf) (Byeongchang Kim, NAACL 2019, [code](https://github.com/ctr4si/MMN), [note](https://zhuanlan.zhihu.com/p/62333393))

[Scoring Sentence Singletons and Pairs for Abstractive Summarization](https://arxiv.org/pdf/1906.00077.pdf) (Logan Lebanoff, ACL 2019, [code](https://github.com/ucfnlp/summarization-sing-pair-mix), [note](https://www.aminer.cn/research_report/5d47c526d5e908133c9468eb))

[How to Write Summaries with Patterns? Learning towards Abstractive Summarization through Prototype Editing](https://arxiv.org/pdf/1909.08837.pdf) (Shen Gao, EMNLP 2019, [code](https://github.com/gsh199449/proto-summ), [note](https://www.icst.pku.edu.cn/xwgg/xwdt/2019/1318876.htm))

[Controlling the Amount of Verbatim Copying in Abstractive Summarization](https://ojs.aaai.org/index.php/AAAI/article/view/6420) (Kaiqiang Song, AAAI 2020)

[Joint Parsing and Generation for Abstractive Summarization](https://ojs.aaai.org/index.php/AAAI/article/view/6419) (Kaiqiang Song, AAAI 2020)

[Keywords-Guided Abstractive Sentence Summarization](https://ojs.aaai.org/index.php/AAAI/article/view/6333) (Haoran Li, AAAI 2020)

[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf) (Jingqing Zhang, ICML 2020, [code](https://github.com/google-research/pegasus), [note](https://www.linkresearcher.com/theses/7054cdb3-b934-4fd8-9e5b-b4a320f5c6c7))

[Discriminative Adversarial Search for Abstractive Summarization](https://arxiv.org/pdf/2002.10375.pdf) (Thomas Scialom, ICML 2020)

[Fact-based Content Weighting for Evaluating Abstractive Summarisation](https://www.aclweb.org/anthology/2020.acl-main.455.pdf) (Xinnuo Xu, ACL 2020)

[On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/pdf/2005.00661.pdf) (Joshua Maynez, ACL 2020, [code](https://github.com/google-research-datasets/xsum_hallucination_annotations?utm_source=catalyzex.com))

[Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports](https://arxiv.org/pdf/1911.02541.pdf) (Yuhao Zhang, ACL 2020, [note](https://zhuanlan.zhihu.com/p/166193118))

[Self-Attention Guided Copy Mechanism for Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.125.pdf) (Song Xu, ACL 2020)

[FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization](https://arxiv.org/pdf/2005.03754.pdf) (Esin Durmus, ACL 2020)

[Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/pdf/2004.04228.pdf) (Alex Wang, ACL 2020, [note](https://zhuanlan.zhihu.com/p/130280217))

[The Summary Loop: Learning to Write Abstractive Summaries Without Examples](http://people.ischool.berkeley.edu/~hearst/papers/Laban_ACL2020_Abstractive_Summarization.pdf) (Philippe Laban, ACL 2020)

[Evaluating the Factual Consistency of Abstractive Text Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.750.pdf) (Wojciech Kryscinski, EMNLP 2020)

[Reducing Quantity Hallucinations in Abstractive Summarization](https://arxiv.org/pdf/2009.13312.pdf) (Zheng Zhao, EMNLP 2020 Findings)

[Learning to Summarize from Human Feedback](https://arxiv.org/pdf/2009.01325.pdf) (Nisan Stiennon, NeuIPS 2020)

##### Multi-Document Summarization

[Exploring Content Models for Multi-Document Summarization](https://www.aclweb.org/anthology/N09-1041) (Aria Haghighi, NAACL 2009, [code](https://github.com/miso-belica/sumy))

[Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization](https://arxiv.org/pdf/1808.06218.pdf) (Logan Lebanoff, EMNLP 2018, [code](https://github.com/ucfnlp/multidoc_summarization?utm_source=catalyzex.com))

[Abstractive Multi-Document Summarization Based on Semantic Link Network](https://ieeexplore.ieee.org/abstract/document/8736808) (Wei Li, TKDE 2019)

[Hierarchical Transformers for Multi-Document Summarization](https://arxiv.org/pdf/1905.13164.pdf) (Yang Liu, ACL 2019)

[Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization](https://arxiv.org/pdf/1906.00072.pdf) (Sangwoo Cho, ACL 2019, [code](https://github.com/ucfnlp/summarization-dpp-capsnet), [note](https://wemp.app/posts/beb15b57-0dd1-4ec8-8d7a-47a1de75dbbb))

[Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model](https://arxiv.org/pdf/1906.01749.pdf) (Alexander R. Fabbri, ACL 2019, [code](https://github.com/Alex-Fabbri/Multi-News), [note](https://zhuanlan.zhihu.com/p/83768781))

[Leveraging Graph to Improve Abstractive Multi-Document Summarization](https://arxiv.org/pdf/2005.10043.pdf) (Wei Li, ACL 2020, [code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-GraphSum?utm_source=catalyzex.com))

[Multi-document Summarization with Maximal Marginal Relevance-guided Reinforcement Learning](https://arxiv.org/pdf/2010.00117.pdf) (Yuning Mao, EMNLP 2020, [code](https://github.com/morningmoni/RL-MMR?utm_source=catalyzex.com))

##### Opinion Summarization

[Unsupervised Opinion Summarization with Content Planning](https://arxiv.org/pdf/2012.07808.pdf) (Reinald Kim Amplayo, AAAI 2021)

##### Cross-Lingual Summarization

[Attend, Translate and Summarize: An Efficient Method for Neural Cross-Lingual Summarization](https://www.aclweb.org/anthology/2020.acl-main.121.pdf) (Junnan Zhu, ACL 2020)

#### Text Style Transfer

[Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/pdf/1705.09655.pdf) (Tianxiao Shen, NeuIPS 2017, [code](https://github.com/shentianxiao/language-style-transfer?utm_source=catalyzex.com))

[Style Transfer in Text: Exploration and Evaluation](https://arxiv.org/pdf/1711.06861.pdf) (Zhenxin Fu, AAAI 2018, [code](https://github.com/fuzhenxin/text_style_transfer?utm_source=catalyzex.com), [note](https://zhuanlan.zhihu.com/p/32300981))

[Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer](https://www.aclweb.org/anthology/N18-1169.pdf) (Juncen Li, NAACL 2018, [code](https://github.com/rpryzant/delete_retrieve_generate), [note](https://blog.csdn.net/u014475479/article/details/81945534))

[Style Transfer Through Back-Translation](https://arxiv.org/pdf/1804.09000.pdf) (Shrimai Prabhumoye, ACL 2018, [code](https://github.com/shrimai/Style-Transfer-Through-Back-Translation?utm_source=catalyzex.com))

[Unsupervised Text Style Transfer using Language Models as Discriminators](https://arxiv.org/pdf/1805.11749.pdf) (Zichao Yang, NeuIPS 2018, [code](https://github.com/asyml/texar/tree/master/examples/text_style_transfer?utm_source=catalyzex.com))

[Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus](https://arxiv.org/pdf/1903.10671.pdf) (Hongyu Gong, NAACL 2019, [code](https://github.com/HongyuGong/TextStyleTransfer))

[A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer](https://arxiv.org/pdf/1905.10060.pdf) (Fuli Luo, IJCAI 2019, [code](https://github.com/luofuli/DualRL?utm_source=catalyzex.com))

[Mask and Infill: Applying Masked Language Model for Sentiment Transfer](https://arxiv.org/pdf/1908.08039.pdf) (Xing Wu, IJCAI 2019, [code](https://github.com/IIEKES/MLM_transfer))

[Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://www.aclweb.org/anthology/P19-1041.pdf) (Vineet John, ACL 2019, [code](https://github.com/h3lio5/linguistic-style-transfer-pytorch), [note](https://zhuanlan.zhihu.com/p/102783862))

[A Hierarchical Reinforced Sequence Operation Method for Unsupervised Text Style Transfer](https://arxiv.org/pdf/1906.01833.pdf) (Chen Wu, ACL 2019, [code](https://github.com/ChenWu98/Point-Then-Operate), [note](https://www.cnblogs.com/bernieloveslife/p/12748942.html))

[Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/pdf/1905.05621.pdf) (Ning Dai, ACL 2019, [code](https://github.com/fastnlp/style-transformer?utm_source=catalyzex.com), [note](http://ziyangluo.tech/2020/05/12/TSTpaper2StyleTran/))

[Semi-supervised Text Style Transfer: Cross Projection in Latent Space](https://arxiv.org/pdf/1909.11493.pdf) (Mingyue Shang, EMNLP 2019)

[Multiple-Attribute Text Style Transfer](https://openreview.net/pdf?id=H1g2NhC5KQ) (Guillaume Lample, ICLR 2019)

#### Topic Modeling

##### Unsupervised Topic Modeling

[An Introduction to Latent Semantic Analysis](http://lsa.colorado.edu/papers/dp1.LSAintro.pdf) (Thomas K Landauer, 1998, [code](https://github.com/josephwilk/semanticpy), [note](https://zhuanlan.zhihu.com/p/37873878))

[Probabilistic Latent Semantic Analysis](http://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf) (Thomas K Landauer, 1999, [code](https://github.com/laserwave/plsa), [note](https://zhuanlan.zhihu.com/p/37873878))

[Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (David M. Blei, JMLR 2003, [code](https://github.com/lda-project/lda), [note](https://zhuanlan.zhihu.com/p/37873878))

[Correlated Topic Models](http://papers.neurips.cc/paper/2906-correlated-topic-models.pdf) (David M. Blei, NIPS 2005)

[A Neural Autoregressive Topic Model](https://papers.nips.cc/paper/4613-a-neural-autoregressive-topic-model.pdf) (Hugo Larochelle, NIPS 2012, [code](https://github.com/AYLIEN/docnade))

[LightLDA: Big Topic Models on Modest Computer Clusters](https://arxiv.org/pdf/1412.1576.pdf) (Jinhui Yuan, WWW 2015, [code](https://github.com/microsoft/LightLDA), [note](http://d0evi1.com/lightlda/))

[Short and Sparse Text Topic Modeling via Self-Aggregation](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/10847/10978) (Xiaojun Quan, IJCAI 2015, [code](https://github.com/WHUIR/SATM))

[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/pdf/1605.02019.pdf) (Christopher Moody, 2016, [code](https://github.com/cemoody/lda2vec), [note](https://zhuanlan.zhihu.com/p/37873878))

[Topic Modeling of Short Texts: A Pseudo-Document View](https://www.kdd.org/kdd2016/papers/files/rpp1190-zuoA.pdf) (Yuan Zuo, KDD 2016)

[A Word Embeddings Informed Focused Topic Model](http://proceedings.mlr.press/v77/zhao17a/zhao17a.pdf) (He Zhao, ACML 2017)

[Incorporating Knowledge Graph Embeddings into Topic Modeling](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14170/14086) (Liang Yao, AAAI 2017, [note](https://blog.csdn.net/smileyk/article/details/78221342))

[ASTM: An Attentional Segmentation Based Topic Model for Short Texts](https://ieeexplore.ieee.org/abstract/document/8594882) (Jiamiao Wang, ICDM 2018, [code](https://github.com/wjmzjx/ASTM))

[Short-Text Topic Modeling via Non-negative Matrix Factorization Enriched with Local Word-Context Correlations](http://dmkd.cs.vt.edu/papers/WWW18.pdf) (Tian Shi, WWW 2018, [code](https://github.com/tshi04/SeaNMF))

[Improving Topic Quality by Promoting Named Entities in Topic Modeling](https://www.aclweb.org/anthology/P18-2040.pdf) (Katsiaryna Krasnashchok, ACL 2018)

[Inter and Intra Topic Structure Learning with Word Embeddings](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf) (He Zhao, ICML 2018, [code](https://github.com/ethanhezhao/WEDTM))

[Document Informed Neural Autoregressive Topic Models with Distributional Prior](https://arxiv.org/pdf/1809.06709.pdf) (Pankaj Gupta, AAAI 2019, [code](https://github.com/pgcool/iDocNADEe))

[textTOvec: Deep Contextualized Neural Autoregressive Topic Models of Language with Distributed Compositional Prior](https://arxiv.org/pdf/1810.03947.pdf) (Pankaj Gupta, ICLR 2019, [code](https://github.com/pgcool/textTOvec))

[CluWords: Exploiting Semantic Word Clustering Representation for Enhanced Topic Modeling](https://dl.acm.org/doi/abs/10.1145/3289600.3291032) (Felipe Viegas, WSDM 2019)

[The Dynamic Embedded Topic Model](https://arxiv.org/pdf/1907.05545.pdf) (Adji B. Dieng, 2019, [code](https://github.com/adjidieng/DETM))

[Topic Modeling in Embedding Spaces](https://arxiv.org/pdf/1907.04907.pdf) (Adji B. Dieng, 2019, [code](https://github.com/adjidieng/ETM), [note](https://zhuanlan.zhihu.com/p/105741773))

[Neural Mixed Counting Models for Dispersed Topic Discovery](https://www.aclweb.org/anthology/2020.acl-main.548.pdf) (Jiemin Wu, ACL 2020)

[Graph Attention Topic Modeling Network](https://yangliang.github.io/pdf/www20.pdf) (Liang Yang, WWW 2020)

##### Supervised Topic Modeling

[Supervised Topic Models](https://papers.nips.cc/paper/3328-supervised-topic-models.pdf) (David M. Blei, NIPS 2008)

[Labeled LDA: A Supervised Topic Model for Credit Attribution in Multi-Labeled Corpora](https://www.aclweb.org/anthology/D09-1026.pdf) (Daniel Ramage, EMNLP 2009, [code](https://github.com/JoeZJH/Labeled-LDA-Python), [note](https://blog.csdn.net/qy20115549/article/details/90771054))

[DiscLDA: Discriminative Learning for Dimensionality Reduction and Classification](http://papers.nips.cc/paper/3599-disclda-discriminative-learning-for-dimensionality-reduction-and-classification.pdf) (Simon Lacoste-Julien, NIPS 2009, [code](https://github.com/teffland/disclda))

[Replicated Softmax: an Undirected Topic Model](https://papers.nips.cc/paper/3856-replicated-softmax-an-undirected-topic-model.pdf) (Ruslan Salakhutdinov, NIPS 2009)

[Partially Labeled Topic Models for Interpretable Text Mining](http://susandumais.com/kdd2011-pldp-final.pdf) (Daniel Ramage, KDD 2011)

[MedLDA: Maximum Margin Supervised Topic Models](http://www.jmlr.org/papers/volume13/zhu12a/zhu12a.pdf) (Jun Zhu, JMLR 2013)

[A Biterm Topic Model for Short Texts](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) (Xiaohui Yan, WWW 2013, [code](https://github.com/markoarnauto/biterm), [note](https://blog.csdn.net/windows2/article/details/16812363))

[BTM: Topic Modeling over Short Texts](https://ieeexplore.ieee.org/abstract/document/6778764) (Xueqi Chen, TKDE 2014)

[Efficient Methods for Incorporating Knowledge into Topic Models](https://www.aclweb.org/anthology/D15-1037.pdf) (Yi Yang, EMNLP 2015)

[Improving Topic Models with Latent Feature Word Representations](https://www.aclweb.org/anthology/Q15-1022v2.pdf) (Dat Quoc Nguyen, TACL 2015, [code](https://github.com/datquocnguyen/LFTM))

[Topic Modeling for Short Texts with Auxiliary Word Embeddings](https://www.ntu.edu.sg/home/AXSun/paper/sigir16text.pdf) (Chenliang Li, SIGIR 2016, [code](https://github.com/NobodyWHU/GPUDMM))

[Efficient Correlated Topic Modeling with Topic Embedding](https://arxiv.org/pdf/1707.00206.pdf) (Junxian He, KDD 2017)

[Adapting Topic Models using Lexical Associations with Tree Priors](https://www.aclweb.org/anthology/D17-1203.pdf) (Weiwei Yang, EMNLP 2017)

[MetaLDA: A Topic Model that Efficiently Incorporates Meta Information](https://arxiv.org/pdf/1709.06365.pdf) (He Zhao, ICDM 2017, [code](https://github.com/ethanhezhao/MetaLDA))

[Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge](https://arxiv.org/pdf/1611.10277.pdf) (Ryan J. Gallagher, TACL 2017, [code](https://github.com/gregversteeg/corex_topic))

[PhraseCTM: Correlated Topic Modeling on Phrases within Markov Random Fields](https://www.aclweb.org/anthology/P18-2083.pdf) (Weijie Huang, ACL 2018)

[Dirichlet Belief Networks for Topic Structure Learning](http://papers.nips.cc/paper/8020-dirichlet-belief-networks-for-topic-structure-learning.pdf) (He Zhao, NIPS 2018)

[Discriminative Topic Mining via Category-Name Guided Text Embedding](https://arxiv.org/pdf/1908.07162.pdf) (Yu Meng, WWW 2020, [code](https://github.com/yumeng5/CatE))

#### Keyphrase Extraction

[TextRank: Bringing Order into Texts](http://202.116.81.74/cache/16/03/web.eecs.umich.edu/ed616fd7b9f50b15ac2f92467a16c9f7/mihalcea.emnlp04.pdf) (Rada Mihalcea, EMNLP 2014, [code](https://github.com/summanlp/textrank), [note](https://www.jiqizhixin.com/articles/2018-12-28-18))

[Deep Keyphrase Generation](http://memray.me/uploads/acl17-keyphrase-generation.pdf) (Rui Meng, ACL 2017, [code](https://github.com/memray/seq2seq-keyphrase), [note](https://www.jianshu.com/p/5492e7c916f8))

[Semi-Supervised Learning for Neural Keyphrase Generation](https://arxiv.org/pdf/1808.06773.pdf) (Hai Ye, EMNLP 2018)

[Keyphrase Generation with Correlation Constraints](https://arxiv.org/pdf/1808.07185.pdf) (Jun Chen, EMNLP 2018)

[Title-Guided Encoding for Keyphrase Generation](https://arxiv.org/pdf/1808.08575.pdf) (Wang Chen, AAAI 2019)

[An Integrated Approach for Keyphrase Generation via Exploring the Power of Retrieval and Extraction](https://arxiv.org/pdf/1904.03454.pdf) (Wang Chen, NAACL 2019, [code](https://github.com/Chen-Wang-CUHK/KG-KE-KR-M), [note](https://github.com/Chen-Wang-CUHK/KG-KE-KR-M))

[Glocal: Incorporating Global Information in Local Convolution for Keyphrase Extraction](https://www.aclweb.org/anthology/N19-1182.pdf) (Animesh Prasad, NAACL 2019)

[Keyphrase Generation: A Text Summarization Struggle](https://www.aclweb.org/anthology/N19-1070.pdf) (Erion Cano, NAACL 2019)

[Incorporating Linguistic Constraints into Keyphrase Generation](https://www.aclweb.org/anthology/P19-1515.pdf) (Jing Zhao, ACL 2019)

[Topic-Aware Neural Keyphrase Generation for Social Media Language](https://arxiv.org/pdf/1906.03889.pdf) (Yue Wang, ACL 2019, [code](https://github.com/yuewang-cuhk/TAKG))

[Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards](https://arxiv.org/pdf/1906.04106.pdf) (Hou Pong Chan, ACL 2019, [code](https://github.com/kenchan0226/keyphrase-generation-rl), [note](https://procjx.github.io/2019/10/31/%E3%80%90%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%E3%80%91Neural-Keyphrase-Generation-via-Reinforcement-Learning-with-Adaptive-Rewards/))

[Using Human Attention to Extract Keyphrase from Microblog Post](https://www.aclweb.org/anthology/P19-1588.pdf) (Yingyi Zhang, ACL 2019, [note](https://blog.csdn.net/qq_34325086/article/details/102365847))

[Open Domain Web Keyphrase Extraction Beyond Language Modeling](https://arxiv.org/pdf/1911.02671.pdf) (Lee Xiong, EMNLP 2019)

#### Word Segmentation

[Adversarial Multi-Criteria Learning for Chinese Word Segmentation](https://www.aclweb.org/anthology/P17-1110.pdf) (Xinchi Chen, ACL 2017)

[State-of-the-art Chinese Word Segmentation with Bi-LSTMs](https://arxiv.org/pdf/1808.06511.pdf) (Ji Ma, 2018, [code](https://github.com/efeatikkan/Chinese_Word_Segmenter?utm_source=catalyzex.com))

[Improving Chinese Word Segmentation with Wordhood Memory Networks](https://www.aclweb.org/anthology/2020.acl-main.734v2.pdf) (Yuanhe Tian, ACL 2020, [code](https://github.com/SVAIGBA/WMSeg))

[A Concise Model for Multi-Criteria Chinese Word Segmentation with Transformer Encoder](https://arxiv.org/pdf/1906.12035.pdf) (Xipeng Qiu, EMNLP 2020, [code](https://github.com/acphile/MCCWS))

[Towards Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning](https://arxiv.org/pdf/1903.04190.pdf) (Weipeng Huang, COLING 2020)

#### Sequence Labeling

[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf) (Xuezhe Ma, ACL 2016, [code](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial), [note](https://jeffchy.github.io/2018/09/24/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%EF%BC%9AEnd-to-End-Sequence-Labeling-via-Bi-directional-LSTM-CNN-CRF/))

[Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks](https://arxiv.org/pdf/1703.06345.pdf) (Zhilin Yang, ICLR 2017, [code](https://github.com/kimiyoung/transfer), [note](https://blog.csdn.net/Raina_qing/article/details/88830027))

[Semi-supervised Multitask Learning for Sequence Labeling](https://www.aclweb.org/anthology/P17-1194) (Marek Rei, ACL 2017, [note](https://zhuanlan.zhihu.com/p/34643000))

[Semi-supervised Sequence Tagging with Bidirectional Language Models](https://arxiv.org/pdf/1705.00108.pdf) (Matthew E. Peters, ACL 2017, [note](https://zhuanlan.zhihu.com/p/38140507))

[Empower Sequence Labeling with Task-Aware Neural Language Model](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17123/16075) (Liyuan Liu, AAAI 2018, [code](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF), [note](https://zhuanlan.zhihu.com/p/32716990))

[Contextual String Embeddings for Sequence Labeling](https://www.aclweb.org/anthology/C18-1139.pdf) (Alan Akbik, COLING 2018, [code](https://github.com/flairNLP/flair), [note](https://www.cnblogs.com/Arborday/p/9960031.html))

[Hierarchically-Refined Label Attention Network for Sequence Labeling](https://arxiv.org/pdf/1908.08676.pdf) (Leyang Cui, EMNLP 2019, [code](https://github.com/Nealcly/BiLSTM-LAN), [note](https://zhuanlan.zhihu.com/p/92672564))

##### Named Entity Recognition

[Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf) (Yue Zhang, ACL 2018)

[Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network](https://www.aclweb.org/anthology/D19-1396.pdf) (Dianbo Sui, EMNLP 2019)

[A Survey on Deep Learning for Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/9039685) (Jing Li, TKDE 2020)

[A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/pdf/1910.11476.pdf) (Xiaoya Li, ACL 2020, [code](https://github.com/ShannonAI/mrc-for-flat-nested-ner?utm_source=catalyzex.com))

##### Semantic Role Labeling

[The Berkeley FrameNet Project](https://dl.acm.org/doi/pdf/10.3115/980845.980860?download=true) (Collin F. Baker, ACL 1998)

[Introduction to the CoNLL-2005 Shared Task: Semantic Role Labeling](https://www.aclweb.org/anthology/W05-0620.pdf) (Xavier Carreras, 2005)

[End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks](https://www.aclweb.org/anthology/P15-1109.pdf) (Jie Zhou, ACL 2015, [code](https://github.com/sanjaymeena/semantic_role_labeling_deep_learning))

[A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling](https://www.aclweb.org/anthology/K17-1041.pdf) (Diego Marcheggiani, 2017, [code](https://github.com/diegma/neural-dep-srl))

[Deep Semantic Role Labeling: What Works and What’s Next](https://www.aclweb.org/anthology/P17-1044.pdf) (Luheng He, ACL 2017, [code](https://github.com/luheng/deep_srl), [note](https://www.sohu.com/a/154327575_473283))

[Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/pdf/1703.04826.pdf) (Diego Marcheggiani, EMNLP 2017, [code](https://github.com/diegma/neural-dep-srl), [note](https://zhuanlan.zhihu.com/p/31805975))

[Deep Semantic Role Labeling with Self-Attention](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16725/16025) (Zhixing Tan, AAAI 2018, [code](https://github.com/XMUNLP/Tagger), [note](https://zhuanlan.zhihu.com/p/35179449))

[Linguistically-Informed Self-Attention for Semantic Role Labeling](https://arxiv.org/pdf/1804.08199.pdf) (Emma Strubell, EMNLP 2018, [code](https://github.com/strubell/LISA))

[A Span Selection Model for Semantic Role Labeling](https://arxiv.org/pdf/1810.02245.pdf) (Hiroki Ouchi, EMNLP 2018, [code](https://github.com/hiroki13/span-based-srl), [note](https://github.com/BrambleXu/knowledge-graph-learning/issues/136))

[Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling](https://arxiv.org/pdf/1805.04787.pdf) (Luheng He, ACL 2018, [code](https://github.com/luheng/lsgn), [note](https://blog.csdn.net/choose_c/article/details/90273333))

[Dependency or Span, End-to-End Uniform Semantic Role Labeling Sentiment Analysis](https://arxiv.org/pdf/1901.05280.pdf) (Zuchao Li, AAAI 2019, [code](https://github.com/bcmi220/unisrl), [note](https://www.sohu.com/a/295644284_129720))

[Semantic Role Labeling with Associated Memory Network](https://arxiv.org/pdf/1908.02367.pdf) (Chaoyu Guan, NAACL 2019, [code](https://github.com/Frozenmad/AMN_SRL))

#### Dependency Parsing

[Statistical Dependency Analysis with Support Vector machines](https://pdfs.semanticscholar.org/f0e1/883cf9d1b3c911125f46359f908557fc5827.pdf) (Hiroyasu Yamada, 2003, [code](https://github.com/rohit-jain/parzer))

[A Dynamic Oracle for Arc-Eager Dependency Parsing](https://www.aclweb.org/anthology/C12-1059) (Yoav Goldberg, COLING 2012, [code](https://github.com/dpressel/arcs-py))

[Training Deterministic Parsers with Non-Deterministic Oracles](https://www.aclweb.org/anthology/Q13-1033.pdf) (Yoav Goldberg, TACL 2013, [code](https://github.com/dpressel/arcs-py))

[A Fast and Accurate Dependency Parser using Neural Networks](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf) (Danqi Chen, EMNLP 2014, [code](https://github.com/khenrix/stanford_nn_parser), [note](https://nocater.github.io/2018/11/13/%E8%AE%BA%E6%96%87-A-Fast-and-Accurate-Dependency-Parserusing-Neural-Networks/))

[An Improved Non-monotonic Transition System for Dependency Parsing](https://aclweb.org/anthology/D15-1162) (Matthew Honnibal, EMNLP 2015)

[Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023) (Eliyahu Kiperwasser, TACL 2016, [code](https://github.com/elikip/bist-parser), [note](http://fancyerii.github.io/books/nndepparser/))

[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/pdf/1611.01734.pdf) (Timothy Dozat, ICLR 2017, [code](https://github.com/yzhangcs/parser), [note](https://www.hankcs.com/nlp/parsing/deep-biaffine-attention-for-neural-dependency-parsing.html))

[Deep Multitask Learning for Semantic Dependency Parsing](https://arxiv.org/pdf/1704.06855.pdf) (Hao Peng, ACL 2017, [code](https://github.com/Noahs-ARK/NeurboParser), [note](https://chao1224.gitbooks.io/running-paper/content/nlp/acl/acl2017/deep-multitask-learning-for-semantic-dependency-parsing.html))

[Simpler but More Accurate Semantic Dependency Parsing](https://www.aclweb.org/anthology/P18-2077.pdf) (Timothy Dozat, ACL 2018, [code](https://github.com/tdozat/Parser-v3))

[Multi-Task Semantic Dependency Parsing with Policy Gradient for Learning Easy-First Strategies](https://arxiv.org/pdf/1906.01239.pdf) (Shuhei Kurita, ACL 2019)

#### Sentiment Analysis

##### Overview

[Opinion Mining and Sentiment Analysis](https://www.cse.iitb.ac.in/~pb/cs626-449-2009/prev-years-other-things-nlp/sentiment-analysis-opinion-mining-pang-lee-omsa-published.pdf) (Bo Pang, 2008)

[Sentiment Analysis and Opinion Mining](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.244.9480&rep=rep1&type=pdf) (Bing Liu, 2012)

##### Dataset

[SemEval-2014 Task 4: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S14-2004) (Maria Pontiki, SemEval 2014, [code](http://alt.qcri.org/semeval2014/task4/), [note](https://zhuanlan.zhihu.com/p/59494279))

[SemEval-2015 Task 12: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S15-2082) (Maria Pontiki, SemEval 2015, [code](http://alt.qcri.org/semeval2015/task12/), [note](https://zhuanlan.zhihu.com/p/59791999))

[SemEval-2016 Task 5: Aspect Based Sentiment Analysis](http://www.aclweb.org/anthology/S16-1002) (Maria Pontiki, SemEval 2016, [code](http://alt.qcri.org/semeval2016/task5/))

##### Sentiment Lexicon

[Building Large-Scale Twitter-Specific Sentiment Lexicon: A Representation Learning Approach](http://www.aclweb.org/anthology/C14-1018) (Duyu Tang, COLING 2014)

[Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora](http://www.aclweb.org/anthology/D16-1057) (William L. Hamilton, EMNLP 2016)

##### Sentiment Embedding

[Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification](http://www.anthology.aclweb.org/P/P14/P14-1146.pdf) (Duyu Tang, ACL 2014, [note](https://zhuanlan.zhihu.com/p/24217324))

[SenticNet 5: Discovering Conceptual Primitives for Sentiment Analysis by Means of Context Embeddings](http://sentic.net/senticnet-5.pdf) (Erik Cambria, AAAI 2018, [code1](http://sentic.net/downloads/), [code2](https://github.com/yurimalheiros/senticnetapi))

##### Sentiment Classification

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) (Richard Socher, EMNLP 2013)

[Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts](https://www.aclweb.org/anthology/C14-1008.pdf) (C´ıcero Nogueira dos Santos, COLING 2014)

[Document Modeling with Gated Recurrent Neural Network for Sentiment Classification](https://www.aclweb.org/anthology/D15-1167.pdf) (Duyu Tang, EMNLP 2015)

[Cached Long Short-Term Memory Neural Networks for Document-Level Sentiment Classification](https://arxiv.org/pdf/1610.04989.pdf) (Jiacheng Xu, EMNLP 2016)

[Neural Sentiment Classification with User and Product Attention](https://www.aclweb.org/anthology/D16-1171.pdf) (Huimin Chen, EMNLP 2016)

[Combination of Convolutional and Recurrent Neural Network for Sentiment Analysis of Short Texts](https://www.aclweb.org/anthology/C16-1229.pdf) (Xingyou Wang, COLING 2016)

[A Cognition Based Attention Model for Sentiment Analysis](https://www.aclweb.org/anthology/D17-1048.pdf) (Yunfei Long, EMNLP 2017)

[Improving Review Representations with User Attention and Product Attention for Sentiment Classification](https://arxiv.org/pdf/1801.07861.pdf) (Zhen Wu, AAAI 2018)

[SNNN: Promoting Word Sentiment and Negation in Neural Sentiment Classification](https://pdfs.semanticscholar.org/e82d/6ac78f83ceca584ed56f6c5591e964bf2406.pdf) (Qinmin Hu, AAAI 2018)

[A Helping Hand: Transfer Learning for Deep Sentiment Analysis](https://www.aclweb.org/anthology/P18-1235.pdf) (Xin Dong, ACL 2018)

[Cold-Start Aware User and Product Attention for Sentiment Classification](https://arxiv.org/pdf/1806.05507.pdf) (Reinald Kim Amplayo, ACL 2018)

[A Lexicon-Based Supervised Attention Model for Neural Sentiment Analysis](https://www.aclweb.org/anthology/C18-1074.pdf) (Yicheng Zou, COLING 2018)

[Neural Review Rating Prediction with User and Product Memory](https://dl.acm.org/doi/abs/10.1145/3357384.3358138) (Zhiguan Yuan, CIKM 2019)

[Sentiment Lexicon Enhanced Neural Sentiment Classification](https://dl.acm.org/doi/abs/10.1145/3357384.3357973) (Chuhan Wu, CIKM 2019)

##### Opinion Target Extraction

[Mining and Summarizing Customer Reviews](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf) (Minqing Hu, KDD 2004, [note](https://zhuanlan.zhihu.com/p/76436724))

[Extracting Product Features and Opinions from Reviews](http://turing.cs.washington.edu/papers/emnlp05_opine.pdf) (Ana-Maria Popescu, EMNLP 2005)

[Modeling Online Reviews with Multi-grain Topic Models](http://ivan-titov.org/papers/www08.pdf) (Ivan Titov, WWW 2008, [code](https://github.com/m-ochi/mglda))

[Phrase Dependency Parsing for Opinion Mining](https://www.aclweb.org/anthology/D09-1159) (Yuanbin Wu, EMNLP 2009)

[A Novel Lexicalized HMM-based Learning Framework for Web Opinion Mining](http://people.cs.pitt.edu/~huynv/research/aspect-sentiment/A%20novel%20lexicalized%20HMM-based%20learning%20framework%20for%20web%20opinion%20mining.pdf) (Wei Jin, ICML 2009)

[Structure-Aware Review Mining and Summarization](https://pdfs.semanticscholar.org/1256/c05bd50a80bb0a223ca94674c71fd61fad5a.pdf) (Fangtao Li, COLING 2010, [note](https://www.jianshu.com/p/b3ccdf21fef0))

[Opinion Target Extraction in Chinese News Comments](https://www.aclweb.org/anthology/C10-2090) (Tengfei Ma, COLING 2010)

[Extracting Opinion Targets in a Single- and Cross-Domain Setting](https://www.aclweb.org/anthology/D10-1101) (Niklas Jakob, EMNLP 2010)

[Opinion Word Expansion and Target Extraction through Double Propagation](https://web.science.mq.edu.au/~rdale/transfer/CL/10-010.pdf) (Guang Qiu, CL 2011)

[Opinion Target Extraction Using Word-Based Translation Model](http://www.nlpr.ia.ac.cn/cip/ZhaoJunPublications/paper/EMNLP2012.LK.pdf) (Kang Liu, EMNLP 2012)

[Opinion Target Extraction Using Partially-Supervised Word Alignment Model](https://pdfs.semanticscholar.org/9751/81c84a0991bb69f5af825e2019080d22cfcd.pdf) (Kang Liu, IJCAI 2013)

[Exploiting Domain Knowledge in Aspect Extraction](https://www.aclweb.org/anthology/D13-1172.pdf) (Zhiyuan Chen, EMNLP 2013)

[Recursive Neural Conditional Random Fields for Aspect-based Sentiment Analysis](http://www.aclweb.org/anthology/D16-1059) (Wenya Wang, EMNLP 2016, [code](https://github.com/happywwy/Recursive-Neural-Conditional-Random-Field), [note](https://www.jianshu.com/p/419cee7f2814))

[Improving Opinion Aspect Extraction Using Semantic Similarity and Aspect Associations](https://aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11973/12051) (Qian Liu, AAAI 2016, [note](https://blog.csdn.net/qifeiyang112358/article/details/82849248))

[Unsupervised word and dependency path embeddings for aspect term extraction](https://arxiv.org/pdf/1605.07843.pdf) (Yichun Yin, IJCAI 2016, [note](https://zhuanlan.zhihu.com/p/27246419))

[Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14441/14256) (Wenya Wang, AAAI 2017, [code](https://github.com/happywwy/Coupled-Multi-layer-Attentions), [note](https://zhuanlan.zhihu.com/p/33088676))

[Recurrent Neural Networks with Auxiliary Labels for Cross-Domain Opinion Target Extraction](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14865/14130) (Ying Ding, AAAI 2017)

[Multi-task Memory Networks for Category-specific Aspect and Opinion Terms Co-extraction](https://arxiv.org/pdf/1702.01776.pdf) (Wenya Wang, 2017)

[An Unsupervised Neural Attention Model for Aspect Extraction](http://www.aclweb.org/anthology/P17-1036) (Ruidan He, ACL 2017, [code](https://github.com/ruidan/Unsupervised-Aspect-Extraction), [note](https://www.jianshu.com/p/241cb238e21f))

[Lifelong Learning CRF for Supervised Aspect Extraction](https://www.aclweb.org/anthology/P17-2023) (Lei Shu, ACL 2017)

[Deep Multi-Task Learning for Aspect Term Extraction with Memory Interaction](http://www.aclweb.org/anthology/D17-1310) (Xin Li, EMNLP 2017, [note](https://zhuanlan.zhihu.com/p/51632476))

[Aspect Term Extraction with History Attention and Selective Transformation](https://arxiv.org/pdf/1805.00760.pdf) (Xin Li, IJCAI 2018, [code](https://github.com/lixin4ever/HAST), [note](https://zhuanlan.zhihu.com/p/51189078))

[Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction](https://www.aclweb.org/anthology/P18-2094) (Hu Xu, ACL 2018, [code](https://github.com/howardhsu/DE-CNN), [note](https://zhuanlan.zhihu.com/p/72092287))

[ExtRA: Extracting Prominent Review Aspects from Customer Feedback](https://aclweb.org/anthology/D18-1384) (Zhiyi Luo, EMNLP 2018, [note](https://zhuanlan.zhihu.com/p/51767759))

[Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling](https://www.aclweb.org/anthology/N19-1259) (Zhifang Fan, NAACL 2019, [code](https://github.com/NJUNLP/TOWE), [note](https://www.linkresearcher.com/theses/761656d7-5d1e-4d54-a723-361ee1eaa113))

##### Aspect-Based Sentiment Classification

[Target-dependent twitter sentiment classification](http://www.anthology.aclweb.org/P/P11/P11-1016.pdf) (Long Jiang, ACL 2011)

[Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](http://www.aclweb.org/anthology/P14-2009) (Li Dong, ACL 2014, [note](https://blog.csdn.net/VictoriaW/article/details/51943563))

[Target-dependent twitter sentiment classification with rich automatic features](http://www.ijcai.org/Proceedings/15/Papers/194.pdf) (Duy-Tin Vo, IJCAI 2015, [code](https://github.com/duytinvo/ijcai2015))

[Effective LSTMs for Target-Dependent Sentiment Classification](http://www.aclweb.org/anthology/C16-1311) (Duyu Tang, COLING 2016, [code](https://github.com/scaufengyang/TD-LSTM), [note](https://zhuanlan.zhihu.com/p/33986102))

[Attention-based LSTM for Aspect-level Sentiment Classification](http://www.aclweb.org/anthology/D16-1058) (Yequan Wang, EMNLP 2016, [code](https://github.com/scaufengyang/TD-LSTM), [note](https://zhuanlan.zhihu.com/p/34005136))

[Aspect Level Sentiment Classification with Deep Memory Network](http://www.aclweb.org/anthology/D16-1021) (Duyu Tang, EMNLP 2016, [code](https://github.com/pcgreat/mem_absa), [note](https://zhuanlan.zhihu.com/p/34033477))

[A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis](http://www.aclweb.org/anthology/D16-1103) (Sebastian Ruder, EMNLP 2016, [note](https://zhuanlan.zhihu.com/p/23477057))

[Interactive Attention Networks for Aspect-Level Sentiment Classification](https://www.ijcai.org/proceedings/2017/0568.pdf) (Dehong Ma, IJCAI 2017, [code](https://github.com/lpq29743/IAN), [note](https://zhuanlan.zhihu.com/p/34041012))

[Recurrent Attention Network on Memory for Aspect Sentiment Analysis](http://www.aclweb.org/anthology/D17-1047) (Peng Chen, EMNLP 2017, [code](https://github.com/lpq29743/RAM), [note](https://zhuanlan.zhihu.com/p/34043504))

[Attention Modeling for Targeted Sentiment](http://leoncrashcode.github.io/Documents/EACL2017.pdf) (Jiangming Liu, EACL 2017, [code](https://github.com/vipzgy/AttentionTargetClassifier))

[Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16541/16152) (Yukun Ma, AAAI 2018, [note](https://zhuanlan.zhihu.com/p/53251543))

[Modeling Inter-Aspect Dependencies for Aspect-Based Sentiment Analysis](http://www.aclweb.org/anthology/N18-2043) (Devamanyu Hazarika, NAACL 2018, [code](https://github.com/xgy221/lstm-inter-aspect), [note](https://zhuanlan.zhihu.com/p/54327441))

[IARM: Inter-aspect relation modeling with memory networks in aspect-based sentiment analysis](http://www.aclweb.org/anthology/D18-1377) (Navonil Majumder, EMNLP 2018, [code](https://github.com/SenticNet/IARM))

[Content Attention Model for Aspect Based Sentiment Analysis](https://dl.acm.org/citation.cfm?doid=3178876.3186001) (Qiao Liu, WWW 2018, [code1](https://github.com/uestcnlp/Cabasc), [code2](https://github.com/songyouwei/ABSA-PyTorch), [note](https://zhuanlan.zhihu.com/p/61575551))

[Convolution-based Memory Network for Aspect-based Sentiment Analysis](https://dl.acm.org/citation.cfm?id=3209978.3210115) (Chuang Fan, SIGIR 2018)

[Aspect Based Sentiment Analysis with Gated Convolutional Networks](http://www.aclweb.org/anthology/P18-1234) (Wei Xue, ACL 2018, [code](https://github.com/wxue004cs/GCAE), [note](https://zhuanlan.zhihu.com/p/50284374))

[Exploiting Document Knowledge for Aspect-level Sentiment Classification](https://www.aclweb.org/anthology/P18-2092.pdf) (Ruidan He, ACL 2018, [code](https://github.com/ruidan/Aspect-level-sentiment), [note](https://zhuanlan.zhihu.com/p/52123748))

[Transformation Networks for Target-Oriented Sentiment Classification](https://www.aclweb.org/anthology/P18-1087) (Xin Li, ACL 2018, [code](https://github.com/lixin4ever/TNet), [note](https://zhuanlan.zhihu.com/p/61586882))

[Multi-grained Attention Network for Aspect-Level Sentiment Classification](https://www.aclweb.org/anthology/D18-1380.pdf) (Feifan Fan, EMNLP 2018, [note](https://zhuanlan.zhihu.com/p/64301255))

[A Position-aware Bidirectional Attention Network for Aspect-level Sentiment Analysis](https://www.aclweb.org/anthology/C18-1066.pdf) (Shuqin Gu, COLING 2018, [code](https://github.com/hiyouga/PBAN-PyTorch), [note](https://zhuanlan.zhihu.com/p/62696026))

[Attentional Encoder Network for Targeted Sentiment Classification](https://arxiv.org/pdf/1902.09314.pdf) (Youwei Song, 2019, [code](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aen.py), [note](https://zhuanlan.zhihu.com/p/68858093))

[A Human-Like Semantic Cognition Network for Aspect-Level Sentiment Classification](https://www.aaai.org/ojs/index.php/AAAI/article/view/4635) (Zeyang Lei, AAAI 2019, [code](https://github.com/eeGuoJun/AAAI2019_HSCN))

[Adapting BERT for Target-Oriented Multimodal Sentiment Classification](https://www.ijcai.org/Proceedings/2019/0751.pdf) (Jianfei Yu, IJCAI 2019, [code](https://github.com/jefferyYu/TomBERT), [note](https://zhuanlan.zhihu.com/p/239892083))

[Deep Mask Memory Networks with Semantic Dependency and Context Moment for Aspect-based Sentiment Analysis](https://www.ijcai.org/Proceedings/2019/0707.pdf) (Peiqin Lin, IJCAI 2019, [code](https://github.com/lpq29743/DMMN-SDCM), [note](https://zhuanlan.zhihu.com/p/150462314))

[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/pdf/1904.02232.pdf) (Hu Xu, NAACL 2019, [code](https://github.com/howardhsu/BERT-for-RRC-ABSA), [note](https://zhuanlan.zhihu.com/p/72092287))

[Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf) (Chi Sun, NAACL 2019, [code](https://github.com/HSLCY/ABSA-BERT-pair), [note](https://zhuanlan.zhihu.com/p/69786643))

[Replicate, Walk, and Stop on Syntax: an Effective Neural Network Model for Aspect-Level Sentiment Classification](https://www.researchgate.net/profile/Samuel_Mensah8/publication/342238197_Replicate_Walk_and_Stop_on_Syntax_An_Effective_Neural_Network_Model_for_Aspect-Level_Sentiment_Classification/links/5f0ceff392851c38a51ccd83/Replicate-Walk-and-Stop-on-Syntax-An-Effective-Neural-Network-Model-for-Aspect-Level-Sentiment-Classification.pdf) (Yaowei Zheng, AAAI 2020, [code](https://github.com/hiyouga/RepWalk))

[Inducing Target-Specific Latent Structures for Aspect Sentiment Classification](https://www.aclweb.org/anthology/2020.emnlp-main.451.pdf) (Chenhua Chen, EMNLP 2020, [note](https://zhuanlan.zhihu.com/p/311246774))

##### Aspect-Based Sentiment Analysis

[A Joint Model of Text and Aspect Ratings for Sentiment Summarization](http://ivan-titov.org/papers/acl08.pdf) (Ivan Titov, ACL 2008)

[Bidirectional Inter-dependencies of Subjective Expressions and Targets and their Value for a Joint Model](https://pdfs.semanticscholar.org/6047/235275b2b8d414b8ac472fd19f2a1a6144b6.pdf) (Roman Klinger, ACL 2013)

[Joint Inference for Fine-grained Opinion Extraction](https://www.aclweb.org/anthology/P13-1161) (Bishan Yang, ACL 2013)

[Open Domain Targeted Sentiment](https://www.aclweb.org/anthology/D13-1171) (Margaret Mitchell, EMNLP 2013)

[Joint Modeling of Opinion Expression Extraction and Attribute Classification](https://www.aclweb.org/anthology/Q14-1039.pdf) (Bishan Yang, TACL 2014)

[Neural Networks for Open Domain Targeted Sentiment](https://www.aclweb.org/anthology/D15-1073) (Meishan Zhang, EMNLP 2015, [code](https://github.com/SUTDNLP/OpenTargetedSentiment))

[Learning Latent Sentiment Scopes for Entity-Level Sentiment Analysis](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14931/14137) (Hao Li, AAAI 2017)

[Joint Learning for Targeted Sentiment Analysis](https://www.aclweb.org/anthology/D18-1504)  (Dehong Ma, EMNLP 2018)

[A Unified Model for Opinion Target Extraction and Target Sentiment Prediction](https://arxiv.org/pdf/1811.05082.pdf) (Xin Li, AAAI 2019, [code](https://github.com/lixin4ever/E2E-TBSA), [note](https://zhuanlan.zhihu.com/p/52705613))

[A Span-based Joint Model for Opinion Target Extraction and Target Sentiment Classification](https://www.ijcai.org/Proceedings/2019/0762.pdf) (Yan Zhou, IJCAI 2019)

[An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis](https://arxiv.org/pdf/1906.06906.pdf) (Ruidan He, ACL 2019, [code](https://github.com/ruidan/IMN-E2E-ABSA), [note](https://blog.csdn.net/BeforeEasy/article/details/104219019))

[DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction](https://arxiv.org/pdf/1906.01794.pdf) (Huaishao Luo, ACL 2019, [code](https://github.com/ArrowLuo/DOER), [note](https://blog.csdn.net/weixin_44740082/article/details/103281743))

[Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification](https://arxiv.org/pdf/1906.03820.pdf) (Minghao Hu, ACL 2019, [code](https://github.com/huminghao16/SpanABSA), [note](https://zhuanlan.zhihu.com/p/144393570))

[A Shared-Private Representation Model with Coarse-to-Fine Extraction for Target Sentiment Analysis](https://www.aclweb.org/anthology/2020.findings-emnlp.382.pdf) (Peiqin Lin, EMNLP 2020 Findings, [code](https://github.com/lpq29743/SPRM), [note](https://zhuanlan.zhihu.com/p/268419578))

[Understanding Pre-trained BERT for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2011.00169.pdf) (Hu Xu, COLING 2020, [code](https://github.com/howardhsu/BERT-for-RRC-ABSA))

[A Unified Generative Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/pdf/2106.04300.pdf) (Hang Yan, ACL 2021)

##### Emotion Cause Detection

[Emotion Cause Events: Corpus Construction and Analysis](https://www.researchgate.net/profile/Chu-Ren_Huang/publication/220746716_Emotion_Cause_Events_Corpus_Construction_and_Analysis/links/0912f508ff080541ac000000/Emotion-Cause-Events-Corpus-Construction-a) (Sophia Yat Mei Lee, LREC 2010)

[A Text-driven Rule-based System for Emotion Cause Detection](https://dl.acm.org/doi/pdf/10.5555/1860631.1860637?download=true) (Sophia Yat Mei Lee, NAACL 2010)

[Emotion Cause Detection with Linguistic Constructions](https://dl.acm.org/doi/pdf/10.5555/1873781.1873802?download=true) (Ying Chen, COLING 2010)

[EMOCause: An Easy-adaptable Approach to Emotion Cause Contexts](https://www.aclweb.org/anthology/W11-1720.pdf) (Irene Russo, 2011)

[Text-based Emotion Classification Using Emotion Cause Extraction](https://www.sciencedirect.com/science/article/pii/S0957417413006945) (Weiyuan Li, 2013)

[Event-Driven Emotion Cause Extraction with Corpus Construction](https://www.aclweb.org/anthology/D16-1170.pdf) (Lin Gui, EMNLP 2016)

[A Question Answering Approach to Emotion Cause Extraction](https://arxiv.org/pdf/1708.05482.pdf) (Liu Gui, EMNLP 2017)

[A Co-Attention Neural Network Model for Emotion Cause Analysis with Emotional Context Awareness](https://www.aclweb.org/anthology/D18-1506.pdf) (Xiangju Li, EMNLP 2018)

[Who Feels What and Why? Annotation of a Literature Corpus with Semantic Roles of Emotions](https://www.aclweb.org/anthology/C18-1114.pdf) (Evgeny Kim, COLING 2018)

[Context-Aware Emotion Cause Analysis with Multi-Attention-Based Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S0950705119301273) (Xiangju Li, KBS 2019)

[Multiple Level Hierarchical Network-Based Clause Selection for Emotion Cause Extraction](https://ieeexplore.ieee.org/abstract/document/8598785) (Xinyi Yu, IEEE Access 2019)

[From Independent Prediction to Reordered Prediction: Integrating Relative Position and Global Label Information to Emotion Cause Identification](https://arxiv.org/pdf/1906.01230.pdf) (Zixiang Ding, AAAI 2019, [code](https://github.com/NUSTM/PAEDGL), [note](https://zhuanlan.zhihu.com/p/240460324))

[RTHN: A RNN-Transformer Hierarchical Network for Emotion Cause Extraction](https://arxiv.org/pdf/1906.01236.pdf) (Rui Xia, IJCAI 2019, [code](https://github.com/NUSTM/RTHN))

[A Knowledge Regularized Hierarchical Approach for Emotion Cause Analysis](https://www.aclweb.org/anthology/D19-1563.pdf) (Chuang Fan, EMNLP 2019)

[Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction](https://arxiv.org/pdf/2106.03518.pdf) (Hanqi Yan, ACL 2021, [code](https://github.com/hanqi-qi/Position-Bias-Mitigation-in-Emotion-Cause-Analysis))

##### Emotion Cause Analysis

[Joint Learning for Emotion Classification and Emotion Cause Detection](https://www.aclweb.org/anthology/D18-1066) (Ying Chen, EMNLP 2018)

[Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts](https://www.aclweb.org/anthology/P19-1096) (Rui Xia, ACL 2019, [code](https://github.com/NUSTM/ECPE), [note](https://mikito.mythsman.com/post/5d2bf2685ed28235d7573179/))

[ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction](https://www.aclweb.org/anthology/2020.acl-main.288.pdf) (Zixiang Ding, ACL 2020, [code](https://github.com/NUSTM/ECPE-2D))

#### Dialogue System

[A Survey on Dialogue Systems: Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf) (Hongshen Chen, 2017, [note](https://cloud.tencent.com/developer/article/1337267))

[AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine](http://www.aclweb.org/anthology/P/P17/P17-2079.pdf) (Minghui Qiu, ACL 2017, [note](https://blog.csdn.net/u011239443/article/details/83829265))

[Neural Approaches to Conversational AI](https://dl.acm.org/doi/abs/10.1145/3209978.3210183) (Jianfeng Gao, SIGIR 2018)

[The Design and Implementation of XiaoIce, an Empathetic Social Chatbot](https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00368) (Li Zhou, CL 2020)

[Challenges in Building Intelligent Open-domain Dialog Systems](https://arxiv.org/pdf/1905.05709.pdf) (Minlie Huang, TIS 2020)

[Towards a Human-like Open-Domain Chatbot](https://arxiv.org/pdf/2001.09977.pdf) (Daniel Adiwardana, 2020, [code](https://github.com/rustyoldrake/Character-Cartridges-Embodied-Identity?utm_source=catalyzex.com))

##### Dataset

[MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://arxiv.org/pdf/1810.00278.pdf) (Paweł Budzianowski, EMNLP 2018, [code](https://github.com/budzianowski/multiwoz))

[Towards Exploiting Background Knowledge for Building Conversation Systems](https://arxiv.org/pdf/1809.08205.pdf) (Mikita Moghe, EMNLP 2018, [code](https://github.com/nikitacs16/Holl-E?utm_source=catalyzex.com))

[Training Millions of Personalized Dialogue Agents](https://arxiv.org/pdf/1809.01984.pdf) (Pierre-Emmanuel Mazare, EMNLP 2018)

[MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines](https://arxiv.org/pdf/1907.01669.pdf) (Mihail Eric, 2019, [code](https://github.com/budzianowski/multiwoz?utm_source=catalyzex.com))

[Wizard of Wikipedia: Knowledge-Powered Conversational Agents](https://arxiv.org/pdf/1811.01241.pdf) (Emily Dinan, ICLR 2019)

[MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://arxiv.org/pdf/1810.02508.pdf) (Soujanya Poria, ACL 2019, [code](https://github.com/declare-lab/MELD))

[How to Build User Simulators to Train RL-based Dialog Systems](https://arxiv.org/pdf/1909.01388.pdf) (Weiyan Shi, EMNLP 2019, [code](https://github.com/wyshi/user-simulator))

[Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf) (Abhinav Rastogi, AAAI 2020, [code](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), [note](https://blog.csdn.net/weixin_44385551/article/details/103098092))

[CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) (Qi Zhu, TACL 2020, [code](https://github.com/thu-coai/CrossWOZ), [note](https://zhuanlan.zhihu.com/p/115366490))

[A Large-Scale Chinese Short-Text Conversation Dataset](https://link.springer.com/chapter/10.1007/978-3-030-60450-9_8) (Yida Wang, NLPCC 2020, [code](https://github.com/thu-coai/CDial-GPT))

##### Dialogue State Tracking

[The Second Dialog State Tracking Challenge](http://www.aclweb.org/anthology/W14-4337) (Matthew Henderson, SIGDAIL 2014, [code](http://camdial.org/~mh521/dstc/))

[Word-Based Dialog State Tracking with Recurrent Neural Networks](http://www.aclweb.org/anthology/W14-4340) (Matthew Henderson, SIGDAIL 2014)

[Machine Learning for Dialog State Tracking: A Review](https://ai.google/research/pubs/pub44018) (Matthew Henderson, 2015)

[The Dialog State Tracking Challenge Series: A Review](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/williams2016dstc_overview-1.pdf) (Jason D. Williams  2016)

[A Network-based End-to-End Trainable Task-oriented Dialogue System](http://www.aclweb.org/anthology/E17-1042) (Tsung-Hsien Wen, EACL 2017, [code](https://github.com/edward-zhu/dialog), [note](https://www.jianshu.com/p/96c8fd2d2876))

[Neural Belief Tracker: Data-Driven Dialogue State Tracking](http://aclweb.org/anthology/P17-1163) (Nikola Mrksic, ACL 2017, [code](https://github.com/nmrksic/neural-belief-tracker), [note](https://zhuanlan.zhihu.com/p/27470864))

[Fully Statistical Neural Belief Tracking](http://aclweb.org/anthology/P18-2018) (Nikola Mrksic, ACL 2018, [code](https://github.com/nmrksic/neural-belief-tracker))

[Global-Locally Self-Attentive Dialogue State Tracker](http://aclweb.org/anthology/P18-1135) (Victor Zhong, ACL 2018, [code](https://github.com/salesforce/glad))

[Towards Universal Dialogue State Tracking](https://www.aclweb.org/anthology/D18-1299.pdf) (Liliang Ren, EMNLP 2018, [code](https://github.com/renll/StateNet?utm_source=catalyzex.com))

[Dialog State Tracking: A Neural Reading Comprehension Approach](https://arxiv.org/pdf/1908.01946.pdf) (Shuyang Gao, SIGDIAL 2019)

[Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf) (Chien-Sheng Wu, ACL 2019, [code](https://github.com/jasonwu0731/trade-dst), [note](https://zhuanlan.zhihu.com/p/72580652))

[SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking](https://arxiv.org/pdf/1907.07421.pdf) (Hwaran Lee, ACL 2019)

[Scalable and Accurate Dialogue State Tracking via Hierarchical Sequence Generation](https://arxiv.org/pdf/1909.00754.pdf) (Liliang Ren, EMNLP 2019)

[HyST: A Hybrid Approach for Flexible and Accurate Dialogue State Tracking](https://arxiv.org/pdf/1907.00883.pdf) (Rahul Goel, Interspeech 2019)

[Efficient Dialogue State Tracking by Selectively Overwriting Memory](https://arxiv.org/pdf/1911.03906.pdf) (Sungdong Kim, ACL 2020)

[Parallel Interactive Networks for Multi-Domain Dialogue State Generation](https://www.aclweb.org/anthology/2020.emnlp-main.151.pdf) (Junfan Chen, EMNLP 2020)

[Efficient Context and Schema Fusion Networks for Multi-Domain Dialogue State Tracking](https://arxiv.org/pdf/2004.03386.pdf) (Su Zhu, EMNLP 2020 Findings)

[GCDST: A Graph-based and Copy-augmented Multi-domain Dialogue State Tracking](https://www.aclweb.org/anthology/2020.findings-emnlp.95.pdf) (Peng Wu, EMNLP 2020 Findings)

[Non-Autoregressive Dialog State Tracking](https://arxiv.org/pdf/2002.08024.pdf) (Hung Le, ICLR 2020, [code](https://github.com/henryhungle/NADST))

##### Dialogue Act Recognition

[Dialogue Act Modeling for Automatic Tagging and Recognition of Conversational Speech](https://www.aclweb.org/anthology/J00-3003.pdf) (Andreas Stolcke, CL 2000)

[Dialogue Act Classification in Domain-Independent Conversations Using a Deep Recurrent Neural Network](https://www.aclweb.org/anthology/C16-1189.pdf) (Hamed Khanpour, COLING 2016)

[Multi-level Gated Recurrent Neural Network for Dialog Act Classification](https://arxiv.org/pdf/1910.01822.pdf) (Wei Li, COLING 2016)

[Neural-based Context Representation Learning for Dialog Act Classification](https://arxiv.org/pdf/1708.02561.pdf) (Daniel Ortega, SIGDIAL 2017)

[Using Context Information for Dialog Act Classification in DNN Framework](https://www.aclweb.org/anthology/D17-1231.pdf) (Yang Liu, EMNLP 2017)

[A Hierarchical Neural Model for Learning Sequences of Dialogue Acts](https://www.aclweb.org/anthology/E17-1041.pdf) (Quan Hung Tran, EACL 2017)

[Dialogue Act Recognition via CRF-Attentive Structured Network](https://arxiv.org/pdf/1711.05568.pdf) (Zheqian Chen, SIGIR 2018)

[Dialogue Act Sequence Labeling using Hierarchical encoder with CRF](https://arxiv.org/pdf/1709.04250.pdf) (Harshit Kumar, AAAI 2018, [code](https://github.com/YanWenqiang/HBLSTM-CRF))

[A Context-based Approach for Dialogue Act Recognition using Simple Recurrent Neural Networks](https://arxiv.org/pdf/1805.06280.pdf) (Chandrakant Bothe, LREC 2018)

[Conversational Analysis using Utterance-level Attention-based Bidirectional Recurrent Neural Networks](https://arxiv.org/pdf/1805.06242.pdf) (Chandrakant Bothe, INTERSPEECH 2018)

[A Dual-Attention Hierarchical Recurrent Neural Network for Dialogue Act Classification](https://arxiv.org/pdf/1810.09154.pdf) (Ruizhe Li, CONLL 2019)

[Dialogue Act Classification with Context-Aware Self-Attention](https://www.aclweb.org/anthology/N19-1373.pdf) (Vipul Raheja, NAACL 2019)

[Modeling Long-Range Context for Concurrent Dialogue Acts Recognition](https://dl.acm.org/doi/pdf/10.1145/3357384.3358145) (Yue Yu, CIKM 2019)

[Towards Emotion-aided Multi-modal Dialogue Act Classification](https://www.aclweb.org/anthology/2020.acl-main.402.pdf) (Tulika Saha, ACL 2020)

[Integrating User History into Heterogeneous Graph for Dialogue Act Recognition](https://www.aclweb.org/anthology/2020.coling-main.372.pdf) (Dong Wang, COLING 2020)

##### Dialogue Emotion Recognition

[Toward Detecting Emotions in Spoken Dialogs](https://ieeexplore.ieee.org/abstract/document/1395974) (Chul Min Lee, 2005)

[Real-Life Emotions Detection with Lexical and Paralinguistic Cues on Human-Human Call Center Dialogs](https://www.isca-speech.org/archive/archive_papers/interspeech_2006/i06_1636.pdf) (Laurence Devillers, 2006)

[Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos](https://w.sentic.net/conversational-memory-network.pdf) (Devamanyu Hazarika, NAACL 2018)

[ICON: Interactive Conversational Memory Network for Multimodal Emotion Detection](https://www.aclweb.org/anthology/D18-1280.pdf) (Devamanyu Hazarika, EMNLP 2018, [note](https://zhuanlan.zhihu.com/p/63506119))

[DialogueRNN: An Attentive RNN for Emotion Detection in Conversations](https://arxiv.org/pdf/1811.00405.pdf) (Navonil Majumder, AAAI 2019, [note](https://zhuanlan.zhihu.com/p/68497862))

[HiGRU: Hierarchical Gated Recurrent Units for Utterance-level Emotion Recognition](https://arxiv.org/pdf/1904.04446.pdf) (Wenxiang Jiao, NAACL 2019)

[Modeling both Context- and Speaker-Sensitive Dependence for Emotion Detection in Multi-speaker Conversations](https://www.ijcai.org/Proceedings/2019/0752.pdf) (Dong Zhang, IJCAI 2019)

[DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation](https://arxiv.org/pdf/1908.11540.pdf) (Deepanway Ghosal, EMNLP 2019)

[Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations](https://arxiv.org/pdf/1909.10681.pdf) (Peixiang Zhong, EMNLP 2019, [code](https://github.com/zhongpeixiang/KET), [note](https://zhuanlan.zhihu.com/p/90548422))

##### Dialogue Summarization

[Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization](https://www.aclweb.org/anthology/P18-1062.pdf) (Guokan Shang, ACL 2018)

[Abstractive Dialogue Summarization with Sentence-Gated Modeling Optimized by Dialogue Acts](https://arxiv.org/pdf/1809.05715.pdf) (Chih-Wen Goo SLT 2018)

[Abstractive Meeting Summarization via Hierarchical Adaptive Segmental Network Learning](https://dl.acm.org/doi/10.1145/3308558.3313619) (Zhou Zhao, WWW 2019)

[Automatic Dialogue Summary Generation for Customer Service](https://dl.acm.org/doi/10.1145/3292500.3330683) (Chunyi Liu, KDD 2019)

[Topic-aware Pointer-Generator Networks for Summarizing Spoken Conversations](https://arxiv.org/pdf/1910.01335.pdf) (Zhengyuan Liu, ASRU 2019)

[Keep Meeting Summaries on Topic: Abstractive Multi-Modal Meeting Summarization](https://www.aclweb.org/anthology/P19-1210.pdf) (Manling Li, ACL 2019)

[A Hierarchical Network for Abstractive Meeting Summarization with Cross-Domain Pretraining](https://arxiv.org/abs/2004.02016v4) (Chenguang Zhu, EMNLP 2020, [code](https://github.com/JudeLee19/HMNet-End-to-End-Abstractive-Summarization-for-Meetings?utm_source=catalyzex.com))

[How Domain Terminology Affects Meeting Summarization Performance](https://arxiv.org/pdf/2011.00692.pdf) (Jia Jin Koay, COLING 2020)

##### Task-Oriented Dialogue System

[Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems](https://arxiv.org/pdf/1804.08217.pdf) (Andrea Madotto, ACL 2018)

[Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures](https://www.aclweb.org/anthology/P18-1133.pdf) (Wenqiang Lei, ACL 2018, [code](https://github.com/WING-NUS/sequicity), [note](https://blog.csdn.net/weixin_40533355/article/details/82997788))

[Multi-level Memory for Task Oriented Dialogs](https://arxiv.org/pdf/1810.10647.pdf) (Revanth Reddy, NAACL 2019, [note](https://zhuanlan.zhihu.com/p/64595503))

[A Working Memory Model for Task-oriented Dialog Response Generation](https://www.aclweb.org/anthology/P19-1258.pdf) (Xiuyi Chen, ACL 2019, [note](https://blog.csdn.net/weixin_44487404/article/details/105665796))

[Global-to-local Memory Pointer Networks for Task-Oriented Dialogue](https://arxiv.org/pdf/1901.04713.pdf) (Chien-Sheng Wu, ICLR 2019, [code](https://github.com/jasonwu0731/GLMP), [note](https://zhuanlan.zhihu.com/p/57535074))

[Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever](https://arxiv.org/pdf/1909.06762.pdf) (Libo Qin, EMNLP 2019, [code](https://github.com/yizhen20133868/Retriever-Dialogue))

[Hello, It's GPT-2 -- How Can I Help You? Towards the Use of Pretrained Language Models for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1907.05774.pdf) (Paweł Budzianowski, 2019)

[TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf) (Chien-Sheng Wu, EMNLP 2020)

[MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/2009.12005.pdf) (Zhaojiang Lin, EMNLP 2020)

[Learning Knowledge Bases with Parameters for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/2009.13656.pdf) (Andrea Madotto, EMNLP 2020 Findings)

##### Dialogue Modeling and Generation

[Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://www.aclweb.org/anthology/D15-1199.pdf) (Tsung-Hsien Wen, EMNLP 2015)

[Building End-to-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11957/12160) (Iulian V.Serban, AAAI 2016)

[A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf) (Jiwei Li, NAACL 2016, [note](https://zhuanlan.zhihu.com/p/35496909))

[How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/pdf/1603.08023.pdf) (Chia-Wei Liu, EMNLP 2016)

[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541.pdf) (Jiwei Li, 2016, [note](https://zhuanlan.zhihu.com/p/21587758))

[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](https://ojs.aaai.org/index.php/AAAI/article/view/10983) (Iulian Serban, AAAI 2017, [code](https://github.com/mike-n-7/ADEM?utm_source=catalyzex.com))

[Mechanism-Aware Neural Machine for Dialogue Response Generation](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14471/14267) (Ganbin Zhou, AAAI 2017)

[A Conditional Variational Framework for Dialog Generation](https://www.aclweb.org/anthology/P17-2080.pdf) (Xiaoyu Shen, ACL 2017)

[Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://www.aclweb.org/anthology/P17-1061.pdf) (Tiancheng Zhao, ACL 2017, [note](http://www.xuwei.io/2019/04/05/%E3%80%8Alearning-discourse-level-diversity-for-neural-dialog-models-using-conditional-variational-autoencoders%E3%80%8B%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/))

[Generating High-Quality and Informative Conversation Responses with Sequence-to-Sequence Models](https://arxiv.org/pdf/1701.03185.pdf) (Louis Shao, EMNLP 2017)

[Improving Variational Encoder-Decoders in Dialogue Generation](https://ojs.aaai.org/index.php/AAAI/article/view/11960) (Xiaoyu Shen, AAAI 2018)

[RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems](https://ojs.aaai.org/index.php/AAAI/article/view/11321) (Chongyang Tao, AAAI 2018)

[Hierarchical Variational Memory Network for Dialogue Generation](https://dl.acm.org/doi/abs/10.1145/3178876.3186077) (Hongshen Chen, WWW 2018, [code](https://github.com/chenhongshen/HVMN), [note](https://blog.csdn.net/qq_38684093/article/details/84038264))

[Variational Autoregressive Decoder for Neural Response Generation](https://www.aclweb.org/anthology/D18-1354.pdf) (Jiachen Du, EMNLP 2018)

[Explicit State Tracking with Semi-Supervisionfor Neural Dialogue Generation](https://dl.acm.org/doi/abs/10.1145/3269206.3271683) (Xisen Jin, CIKM 2018, [code](https://github.com/AuCson/SEDST), [note](https://zhuanlan.zhihu.com/p/62306940))

[Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization](https://papers.nips.cc/paper/2018/file/23ce1851341ec1fa9e0c259de10bf87c-Paper.pdf) (Yizhe Zhang, NeuIPS 2018)

[Jointly Optimizing Diversity and Relevance in Neural Response Generation](https://arxiv.org/pdf/1902.11205.pdf) (Xiang Gao, NAACL 2019, [code](https://github.com/golsun/SpaceFusion?utm_source=catalyzex.com))

[Domain Adaptive Dialog Generation via Meta Learning](https://arxiv.org/pdf/1906.03520.pdf) (Kun Qian, ACL 2019, [code](https://github.com/qbetterk/DAML), [note](https://liusih.github.io/2019/09/09/Domain%20Adaptive%20Dialog%20Generation%20via%20Meta%20Learning/))

[Pretraining Methods for Dialog Context Representation Learning](https://arxiv.org/pdf/1906.00414.pdf) (Shikib Mehri, ACL 2019, [note](https://zhuanlan.zhihu.com/p/82001834))

[Incremental Transformer with Deliberation Decoder for Document Grounded Conversations](https://arxiv.org/pdf/1907.08854.pdf) (Zekang Li, ACL 2019)

[Improving Neural Conversational Models with Entropy-Based Data Filtering](https://www.aclweb.org/anthology/P19-1567.pdf) (Richard Csaky, ACL 2019, [code](https://github.com/ricsinaruto/dialog-eval))

[ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation](https://arxiv.org/pdf/1907.05339.pdf) (Hainan Zhang, ACL 2019, [code](https://github.com/zhanghainan/ReCoSa), [note](https://zhuanlan.zhihu.com/p/74229505))

[Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention](https://arxiv.org/pdf/1905.12866.pdf) (Wenhu Chen, ACL 2019, [code](https://github.com/wenhuchen/HDSA-Dialog), [note](https://zhuanlan.zhihu.com/p/82460398))

[Hierarchical Prediction and Adversarial Learning For Conditional Response Generation](https://ieeexplore.ieee.org/document/9020173) (Yanran Li, TKDE 2020)

[Hierarchical Reinforcement Learning for Open-Domain Dialog](https://ojs.aaai.org/index.php/AAAI/article/view/6400) (Abdelrhman Saleh, AAAI 2020)

[DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/pdf/1911.00536.pdf) (Yizhe Zhang, ACL 2020, [code](https://github.com/microsoft/DialoGPT?utm_source=catalyzex.com))

[PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://arxiv.org/pdf/1910.07931.pdf) (Siqi Bao, ACL 2020)

[Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks](https://arxiv.org/pdf/1910.14326.pdf) (Yiping Song, ACL 2020, [code](https://github.com/zequnl/CMAML?utm_source=catalyzex.com))

[Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation](https://www.aclweb.org/anthology/2020.acl-main.166.pdf) (Jun Xu, ACL 2020)

[Group-wise Contrastive Learning for Neural Dialogue Generation](https://arxiv.org/pdf/2009.07543.pdf) (Hengyi Cai, EMNLP 2020)

[Plug-and-Play Conversational Models](https://arxiv.org/pdf/2010.04344.pdf) (Andrea Madotto, EMNLP 2020 Findings)

[An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation](https://arxiv.org/pdf/2003.04195.pdf) (Piji Li, 2020)

[The Adapter-Bot: All-In-One Controllable Conversational Model](https://arxiv.org/pdf/2008.12579.pdf) (Andrea Madotto, 2020)

[Variational Transformers for Diverse Response Generation](https://arxiv.org/pdf/2003.12738.pdf) (Zhaojiang Lin, 2020)

##### Stylized Response Generation

[Polite Dialogue Generation Without Parallel Data](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00027) (Tong Niu, TACL 2018)

[Structuring Latent Spaces for Stylized Response Generation](https://arxiv.org/pdf/1909.05361.pdf) (Xiang Gao, EMNLP 2019, [code](https://github.com/golsun/StyleFusion?utm_source=catalyzex.com))

[Stylized Dialogue Response Generation Using Stylized Unpaired Texts](https://arxiv.org/pdf/2009.12719.pdf) (Yinhe Zheng, AAAI 2021)

##### Empathetic Dialogue Generation

[Predicting and Eliciting Addressee’s Emotion in Online Dialogue](https://www.aclweb.org/anthology/P13-1095.pdf) (Takayuki Hasegawa, ACL 2013)

[Large-scale Analysis of Counseling Conversations: An Application of Natural Language Processing to Mental Health](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00111/43371/Large-scale-Analysis-of-Counseling-Conversations) (Tim Althoff, TACL 2016, [code](http://snap.stanford.edu/counseling/))

[Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory](https://ojs.aaai.org/index.php/AAAI/article/view/11325) (Hao Zhou, AAAI 2018, [code](https://github.com/tuxchow/ecm?utm_source=catalyzex.com))

[Eliciting Positive Emotion through Affect-Sensitive Dialogue Response Generation: A Neural Network Approach](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16317/16080) (Nurul Lubis, AAAI 2018)

[Automatic Dialogue Generation with Expressed Emotions](https://www.aclweb.org/anthology/N18-2008.pdf) (Chenyang Huang, NAACL 2018)

[A Syntactically Constrained Bidirectional-Asynchronous Approach for Emotional Conversation Generation](https://www.aclweb.org/anthology/D18-1071.pdf) (Jingyuan Li, EMNLP 2018)

[MOJITALK: Generating Emotional Responses at Scale](https://www.aclweb.org/anthology/P18-1104.pdf) (Xianda Zhou, EMNLP 2018)

[Affective Neural Response Generation](https://link.springer.com/chapter/10.1007/978-3-319-76941-7_12) (Nabiha Asghar, ECIR 2018)

[Topic-Enhanced Emotional Conversation Generation with Attention Mechanism](https://www.sciencedirect.com/science/article/abs/pii/S095070511830457X) (Yehong Peng, KBS 2019)

[Positive Emotion Elicitation in Chat-Based Dialogue Systems](https://ieeexplore.ieee.org/document/8649596) (Nurul Lubis, TASLP 2019)

[An Affect-Rich Neural Conversational Model with Biased Attention and Weighted Cross-Entropy Loss](https://ojs.aaai.org//index.php/AAAI/article/view/4740) (Peixiang Zhong, AAAI 2019, [code](https://github.com/zhongpeixiang/affect-rich-conversational-model?utm_source=catalyzex.com))

[Affect-Driven Dialog Generation](https://arxiv.org/pdf/1904.02793.pdf) (Pierre Colombo, NAACL 2019)

[Generating Responses with a Specific Emotion in Dialog](https://www.aclweb.org/anthology/P19-1359.pdf) (Zhenqiao Song, ACL 2019, [note](https://www.dazhuanlan.com/2020/02/10/5e403d064876d/))

[Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset](https://arxiv.org/pdf/1811.00207.pdf) (Hannah Rashkin, ACL 2019)

[MoEL: Mixture of Empathetic Listeners](https://www.aclweb.org/anthology/D19-1012.pdf) (Zhaojiang Lin, EMNLP 2019, [code](https://github.com/HLTCHKUST/MoEL))

[What If Bots Feel Moods?](https://dl.acm.org/doi/abs/10.1145/3397271.3401108) (Lisong Qiu, SIGIR 2020)

[EmoElicitor: An Open Domain Response Generation Model with User Emotional Reaction Awareness](https://www.ijcai.org/proceedings/2020/0503.pdf) (Shifeng Li, IJCAI 2020)

[CDL: Curriculum Dual Learning for Emotion-Controllable Response Generation](https://arxiv.org/pdf/2005.00329.pdf) (Lei Shen, ACL 2020)

[Balancing Objectives in Counseling Conversations: Advancing Forwards or Looking Backwards](https://arxiv.org/pdf/2005.04245.pdf) (Justine Zhang, ACL 2020)

[A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support](https://arxiv.org/pdf/2009.08441.pdf) (Ashish Sharma, EMNLP 2020)

[Towards Empathetic Dialogue Generation over Multi-type Knowledge](https://arxiv.org/pdf/2009.09708.pdf) (Qintong Li, 2020)

[EmpDG: Multiresolution Interactive Empathetic Dialogue Generation](https://arxiv.org/pdf/1911.08698.pdf) (Qintong Li, COLING 2020)

[Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach](https://arxiv.org/pdf/2101.07714.pdf) (Ashish Sharma, WWW 2021)

[Towards Emotional Support Dialog Systems](https://arxiv.org/pdf/2106.01144.pdf) (Siyang Liu, ACL 2021, [code](https://github.com/thu-coai/Emotional-Support-Conversation), [note](https://mp.weixin.qq.com/s/Gj_h4YSK0cxDxh6EIiTWKw))

[CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation](https://arxiv.org/pdf/2105.08316.pdf) (Chujie Zheng, ACL 2021 Findings)

[Emotion Eliciting Machine: Emotion Eliciting Conversation Generation based on Dual Generator](https://arxiv.org/pdf/2105.08251.pdf) (Hao Jiang, 2021)

##### Persona-Based Dialogue System

[A Persona-Based Neural Conversation Model](https://arxiv.org/pdf/1603.06155.pdf) (Jiwei Li, ACL 2016, [code](https://github.com/shrebox/Personified-Chatbot?utm_source=catalyzex.com))

[Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://www.aclweb.org/anthology/P18-1205.pdf) (Saizheng Zhang, ACL 2018)

[Exploiting Persona Information for Diverse Generation of Conversational Responses](https://arxiv.org/pdf/1905.12188.pdf) (Haoyu Song, IJCAI 2019, [code](https://github.com/vsharecodes/percvae?utm_source=catalyzex.com))

[Personalizing Dialogue Agents via Meta-Learning](https://www.aclweb.org/anthology/P19-1542.pdf) (Andrea Madotto, ACL 2019)

[Generating Persona Consistent Dialogues by Exploiting Natural Language Inference](https://ojs.aaai.org/index.php/AAAI/article/view/6417) (Haoyu Song, AAAI 2020)

[A Neural Topical Expansion Framework for Unstructured Persona-oriented Dialogue Generation](https://arxiv.org/pdf/2002.02153.pdf) (Minghong Xu, ECAI 2020)

[Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation](https://arxiv.org/pdf/2004.07672.pdf) (Hanyu Song, ACL 2020)

[You Impress Me: Dialogue Generation via Mutual Persona Perception](https://arxiv.org/pdf/2004.05388.pdf) (Qian Liu, ACL 2020, [code](https://github.com/SivilTaram/Persona-Dialogue-Generation?utm_source=catalyzex.com))

[Like hiking? You probably enjoy nature: Persona-grounded Dialog with Commonsense Expansions](https://arxiv.org/pdf/2010.03205.pdf) (Bodhisattwa Prasad Majumder, EMNLP 2020)

[Towards Persona-Based Empathetic Conversational Model](https://www.aclweb.org/anthology/2020.emnlp-main.531.pdf) (Peixiang Zhong, EMNLP 2020, [code](https://github.com/zhongpeixiang/PEC?utm_source=catalyzex.com))

##### Knowledge-Grounded Dialogue System

[Incorporating Loose-Structured Knowledge into LSTM with Recall Gate for Conversation Modeling](https://arxiv.org/pdf/1605.05110.pdf) (IJCNN 2017, [note](https://zhuanlan.zhihu.com/p/92605720))

[A Knowledge-Grounded Neural Conversation Model](https://ojs.aaai.org/index.php/AAAI/article/view/11977) (Marjan Ghazvininejad, AAAI 2018, [code](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling?utm_source=catalyzex.com))

[Augmenting End-to-End Dialogue Systems With Commonsense Knowledge](https://ojs.aaai.org/index.php/AAAI/article/view/11923) (Tom Young, AAAI 2018)

[Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/Proceedings/2018/0643.pdf) (Hao Zhou, IJCAI 2018)

[Knowledge Diffusion for Neural Dialogue Generation](https://www.aclweb.org/anthology/P18-1138.pdf) (Shuman Liu, ACL 2018, [note](https://zhuanlan.zhihu.com/p/51939126))

[Learning to Select Knowledge for Response Generation in Dialog Systems](https://www.ijcai.org/Proceedings/2019/0706.pdf) (Rongzhong Lian, IJCAI 2019)

[Enhancing Conversational Dialogue Models with Grounded Knowledge](https://dl.acm.org/doi/abs/10.1145/3357384.3357889) (Wen Zheng, CIKM 2019)

[Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs](https://arxiv.org/pdf/1903.10245.pdf) (Zhibin Liu, EMNLP 2019, [code](https://github.com/PaddlePaddle/models?utm_source=catalyzex.com))

[Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation](https://ojs.aaai.org/index.php/AAAI/article/view/6395) (Pengjie Ren, AAAI 2020)

[RefNet: A Reference-Aware Network for Background Based Conversation](https://ojs.aaai.org/index.php/AAAI/article/view/6370) (Chuan Meng, AAAI 2020)

[Low-Resource Knowledge-Grounded Dialogue Generation](https://arxiv.org/pdf/2002.10348.pdf) (Xueliang Zhao, ICLR 2020)

[Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue](https://arxiv.org/pdf/2002.07510.pdf) (Byeongchang Kim, ICLR 2020, [code](https://github.com/bckim92/sequential-knowledge-transformer?utm_source=catalyzex.com))

[TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact](https://www.ijcai.org/Proceedings/2020/0521.pdf) (Sixing Wu, IJCAI 2020)

[Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness](https://www.aclweb.org/anthology/2020.acl-main.515.pdf) (Sixing Wu, ACL 2020)

[Knowledge-Grounded Dialogue Generation with Pre-trained Language Models](https://arxiv.org/pdf/2010.08824.pdf) (Xueliang Zhao, EMNLP 2020)

[Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation](https://www.aclweb.org/anthology/2020.emnlp-main.275.pdf) (Xiuyi Chen, EMNLP 2020)

[Retrieval-Free Knowledge-Grounded Dialogue Response Generation with Adapters](https://arxiv.org/pdf/2105.06232.pdf) (Yan Xu, 2021)

##### Conversational Recommender System

[Towards Conversational Recommender Systems](https://www.kdd.org/kdd2016/papers/files/rfp0063-christakopoulouA.pdf) (Konstantina Christakopoulou, KDD 2016)

[Conversational Recommender System](https://arxiv.org/pdf/1806.03277.pdf) (Yueming Sun, SIGIR 2018)

[Towards Deep Conversational Recommendations](https://papers.nips.cc/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf) (Raymond Li, NIPS 2018, [note](http://www.xuwei.io/2019/05/02/%E3%80%8Atowards-deep-conversational-recommendations%E3%80%8B%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/))

[Towards Knowledge-Based Recommender Dialog System](https://arxiv.org/pdf/1908.05391.pdf) (Qibin Chen, EMNLP 2019, [code](https://github.com/THUDM/KBRD), [note](https://zhuanlan.zhihu.com/p/270386920))

[Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue](https://arxiv.org/pdf/1909.03922.pdf) (Dongyeop Kang, EMNLP 2019)

[Leveraging Historical Interaction Data for Improving Conversational Recommender System](https://arxiv.org/pdf/2008.08247.pdf) (Kun Zhou, CIKM 2020)

[Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion](https://arxiv.org/pdf/2007.04032.pdf) (Kun Zhou, KDD 2020, [note](https://cloud.tencent.com/developer/article/1663741))

[Towards Conversational Recommendation over Multi-Type Dialogs](https://arxiv.org/pdf/2005.03954.pdf) (Zeming Liu, ACL 2020, [note](https://www.jiqizhixin.com/articles/2020-09-10-3))

#### Question Answering and Machine Reading Comprehension

##### Dataset

[Mctest: A challenge dataset for the open-domainmachine comprehension of text](https://www.aclweb.org/anthology/D13-1020) (Matthew Richardson, EMNLP 2013)

[WIKIQA: A Challenge Dataset for Open-Domain Question Answering](https://aclweb.org/anthology/D15-1237) (Yi Yang, EMNLP 2015)

[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf) (Pranav Rajpurkar, 2016, [code](https://rajpurkar.github.io/SQuAD-explorer/))

[MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268.pdf) (Payal Bajaj, 2016, [code](http://www.msmarco.org/), [note](https://bingning.wang/research/Article/?id=88))

[DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications](https://arxiv.org/pdf/1711.05073.pdf) (Wei He, 2017, [code](https://github.com/baidu/DuReader))

[TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://aclweb.org/anthology/P17-1147) (Mandar Joshi, ACL 2017, [code](https://github.com/mandarjoshi90/triviaqa))

[RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/pdf/1704.04683.pdf) (Guokun Lai, 2017, [code](https://github.com/qizhex/RACE_AR_baselines))

[The NarrativeQA Reading Comprehension Challenge](https://www.aclweb.org/anthology/Q18-1023) (Tomas Kocisky, TACL 2018, [code](https://github.com/deepmind/narrativeqa), [note](https://www.paperweekly.site/papers/notes/400))

[Know What You Don’t Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf) (Pranav Rajpurkar, ACL 2018, [code](https://rajpurkar.github.io/SQuAD-explorer/), [note](https://www.sohu.com/a/235513642_129720))

[CoQA: A Conversational Question Answering Challenge](https://arxiv.org/pdf/1808.07042.pdf) (Siva Reddy, 2018, [code](https://stanfordnlp.github.io/coqa/))

[Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/pdf/1803.05457.pdf) (Peter Clark, 2018, [code](http://data.allenai.org/arc/))

[QuAC : Question Answering in Context](https://arxiv.org/pdf/1808.07036v1.pdf) (Eunsol Choi, EMNLP 2018, [code](http://quac.ai/), [note](https://zhuanlan.zhihu.com/p/84110287))

[A Dataset and Baselines for Sequential Open-Domain Question Answering](https://www.aclweb.org/anthology/D18-1134) (Ahmed Elgohary, EMNLP 2018, [code](http://sequential.qanta.org/))

[Interpretation of Natural Language Rules in Conversational Machine Reading](https://arxiv.org/pdf/1809.01494.pdf) (Marzieh Saeidi, EMNLP 2018)

[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/pdf/1809.02789.pdf) (Todor Mihaylov, EMNLP 2018)

[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/pdf/1903.00161.pdf) (Dheeru Dua, NAACL 2019, [code](https://allennlp.org/drop))

[CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/pdf/1811.00937.pdf) (Alon Talmor, NAACL 2019, [note](https://little1tow.github.io/2019/06/14/2019-05-31/))

[Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning](https://arxiv.org/pdf/1908.05803.pdf) (Pradeep Dasigi, EMNLP 2019, [code](https://allennlp.org/quoref))

[COSMOS QA: Machine Reading Comprehension with Contextual Commonsense Reasoning](https://arxiv.org/pdf/1909.00277.pdf) (Lifu Huang, EMNLP 2019)

[SocialIQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/pdf/1904.09728.pdf) (Maarten Sap, EMNLP 2019)

[PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/pdf/1911.11641.pdf) (Yonatan Bisk, AAAI 2020)

##### Machine Reading Comprehension

[Teaching Machines to Read and Comprehend](https://lanl.arxiv.org/pdf/1506.03340.pdf) (Karl Moritz Hermann, NIPS 2015, [code](https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend), [note](https://zhuanlan.zhihu.com/p/21343662))

[Text Understanding with the Attention Sum Reader Network](http://www.aclweb.org/anthology/P16-1086) (Rudolf Kadlec, ACL 2016, [code](https://github.com/rkadlec/asreader), [note](https://zhuanlan.zhihu.com/p/21354432))

[ReasoNet: Learning to Stop Reading in Machine Comprehension](https://arxiv.org/pdf/1609.05284.pdf) (Yelong Shen, KDD 2017, [note](http://cairohy.github.io/2017/05/22/deeplearning/NLP-RC-ReasoNet-NIPS2016-%E3%80%8AReasoNet%20Learning%20to%20Stop%20Reading%20in%20Machine%20Comprehension%E3%80%8B/))

[Machine Comprehension Using Match-LSTM and Answer Pointer](https://openreview.net/pdf?id=B1-q5Pqxl) (Shuohang Wang, ICLR 2017, [code](https://github.com/MurtyShikhar/Question-Answering), [note](https://zhuanlan.zhihu.com/p/55957106))

[Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf) (Minjoon Seo, ICLR 2017, [code](https://github.com/allenai/bi-att-flow), [note](https://zhuanlan.zhihu.com/p/55975534))

[Attention-over-Attention Neural Networks for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1055.pdf) (Yiming Cui, ACL 2017, [code](https://github.com/OlavHN/attention-over-attention), [note](https://zhuanlan.zhihu.com/p/56246026))

[Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/pdf/1710.10723.pdf) (Christopher Clark, 2017, [code](https://github.com/allenai/document-qa), [note](https://zhuanlan.zhihu.com/p/36812682))

[Gated Self-Matching Networks for Reading Comprehension and Question Answering](http://aclweb.org/anthology/P17-1018) (Wenhui Wang, ACL 2017, [code](https://github.com/HKUST-KnowComp/R-Net), [note](https://www.jianshu.com/p/71d3b4737c23))

[R-Net: Machine Reading Comprehension with Self-Matching Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) (Natural Language Computing Group, 2017, [code](https://github.com/HKUST-KnowComp/R-Net), [note](https://zhuanlan.zhihu.com/p/40271565))

[QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://openreview.net/pdf?id=B14TlG-RW) (Adams Wei Yu, ICLR 2018, [code](https://github.com/minsangkim142/Fast-Reading-Comprehension), [note](https://zhuanlan.zhihu.com/p/56285539))

[Multi-Granularity Hierarchical Attention Fusion Networks for Reading Comprehension and Question Answering](http://www.aclweb.org/anthology/P18-1158) (Wei Wang, ACL 2018, [code](https://github.com/SparkJiao/SLQA), [note](https://zhuanlan.zhihu.com/p/43556383))

[Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification](https://www.aclweb.org/anthology/P18-1178) (Yizhong Wang, ACL 2018, [note](https://indexfziq.github.io/2019/03/08/VNET/))

[Joint Training of Candidate Extraction and Answer Selection for Reading Comprehension](https://www.aclweb.org/anthology/P18-1159) (Zhen Wang, ACL 2018, [note](https://zhuanlan.zhihu.com/p/41161714))

[Knowledgeable Reader: Enhancing Cloze-Style Reading Comprehension with External Commonsense Knowledge](https://www.aclweb.org/anthology/P18-1076) (Todor Mihaylov, ACL 2018)

[Improving Machine Reading Comprehension with General Reading Strategies](https://arxiv.org/pdf/1810.13441.pdf) (Kai Sun, NAACL 2019, [code](https://github.com/nlpdata/strategy), [note](https://blog.csdn.net/mottled233/article/details/104535173))

[SG-Net: Syntax-Guided Machine Reading Comprehension](https://arxiv.org/pdf/1908.05147.pdf) (Zhuosheng Zhang, AAAI 2020, [code](https://github.com/cooelf/SG-Net), [note](https://zhuanlan.zhihu.com/p/82073864))

[Retrospective Reader for Machine Reading Comprehension](https://arxiv.org/pdf/2001.09694.pdf) (Zhuosheng Zhang, 2020, [note](https://zhuanlan.zhihu.com/p/137552707))

##### Answer Selection

[LSTM-based Deep Learning Models for Non-factoid Answer Selection](https://arxiv.org/abs/1511.04108) (Ming Tan, ICLR 2016, [note](https://blog.csdn.net/u010960155/article/details/86756911))

[Hierarchical Attention Flow for Multiple-Choice Reading Comprehension](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16331/16177) (Haichao Zhu, AAAI 2018)

[A Co-Matching Model for Multi-choice Reading Comprehension](https://www.aclweb.org/anthology/P18-2118.pdf) (Shuohang Ming, ACL 2018)

[Option Comparison Network for Multiple-choice Reading Comprehension](https://arxiv.org/pdf/1903.03033.pdf) (Qiu Ran, 2019, [note](https://www.zybuluo.com/songying/note/1428013))

##### Knowledge Based Question Answering

[Information Extraction over Structured Data: Question Answering with Freebase](http://cs.jhu.edu/~xuchen/paper/yao-jacana-freebase-acl2014.pdf) (Xuchen Yao, ACL 2014, [note](https://blog.csdn.net/LAW_130625/article/details/78398888))

[Question Answering over Freebase with Multi-Column Convolutional Neural Networks](http://www.aclweb.org/anthology/P15-1026) (Li Dong, ACL 2015, [code](https://github.com/Evergcj/QA_multi-columnCNN), [note](https://blog.csdn.net/LAW_130625/article/details/78447156))

[Question Answering on Freebase via Relation Extraction and Textual Evidence](https://arxiv.org/pdf/1603.00957v3.pdf) (Kun Xu, ACL 2016, [note](https://zhuanlan.zhihu.com/p/22630320?refer=c_51425207))

[Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks](https://arxiv.org/pdf/1704.08384.pdf) (Rajarshi Das, ACL 2017, [note](https://zhuanlan.zhihu.com/p/26791788))

[Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN](https://arxiv.org/vc/arxiv/papers/1804/1804.03317v2.pdf) (Yingqi Qu, ISMC 2018, [code](https://github.com/quyingqi/kbqa-ar-smcnn), [note](https://blog.csdn.net/Evaooooes/article/details/88691356))

[Variational Reasoning for Question Answering with Knowledge Graph](https://arxiv.org/pdf/1709.04071.pdf) (Yuyu Zhang, AAAI 2018, [note](http://blog.openkg.cn/%E8%AE%BA%E6%96%87%E6%B5%85%E5%B0%9D-%E5%9F%BA%E4%BA%8E%E7%9F%A5%E8%AF%86%E5%9B%BE%E7%9A%84%E9%97%AE%E7%AD%94%E5%8F%98%E5%88%86%E6%8E%A8%E7%90%86/))

[Retrieve, Program, Repeat: Complex Knowledge Base Question Answering via Alternate Meta-learning](https://arxiv.org/pdf/2010.15875.pdf) (Yuncheng Hua, IJCAI 2020)

##### Conversational Question Answering

[SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering](https://arxiv.org/pdf/1812.03593.pdf) (Chenguang Zhu, 2018, [code](https://github.com/Microsoft/SDNet), [note](https://zhuanlan.zhihu.com/p/66100785))

[Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base](https://papers.nips.cc/paper/2018/file/d63fbf8c3173730f82b150c5ef38b8ff-Paper.pdf) (Daya Guo, NIPS 2018)

[FlowQA: Grasping Flow in History for Conversational Machine Comprehension](https://arxiv.org/pdf/1810.06683.pdf) (Hsin-Yuan Huang, ICLR 2019, [code](https://github.com/momohuang/FlowQA), [note](https://zhuanlan.zhihu.com/p/53028792))

[BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412.pdf) (Chen Qu, SIGIR 2019, [code](https://github.com/prdwb/bert_hae))

[Look before you Hop: Conversational Question Answering over Knowledge Graphs Using Judicious Context Expansion](https://dl.acm.org/doi/abs/10.1145/3357384.3358016) (Philipp Christmann, CIKM 2019)

[Attentive History Selection for Conversational Question Answering](https://arxiv.org/pdf/1908.09456.pdf) (Chen Qu, CIKM 2019, [code](https://github.com/prdwb/attentive_history_selection), [note](https://zhuanlan.zhihu.com/p/129800180))

[Multi-Task Learning for Conversational Question Answering over a Large-Scale Knowledge Base](https://arxiv.org/pdf/1910.05069.pdf) (Tao Shen, EMNLP 2019)

##### Visual Question Answering

[VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf) (Aishwarya Agrawal, ICCV 2015)

[Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/pdf/1606.00061.pdf) (Jiasen Lu, NIPS 2016)

[Explicit Knowledge-based Reasoning for Visual Question Answering](https://arxiv.org/pdf/1511.02570.pdf) (Peng Wang, IJCAI 2017)

[FVQA: Fact-based Visual Question Answering](https://arxiv.org/pdf/1606.05433.pdf) (Peng Wang, TPAMI 2018, [note](https://zhuanlan.zhihu.com/p/66282581))

[Straight to the Facts: Learning Knowledge Base Retrieval for Factual Visual Question Answering](https://arxiv.org/pdf/1809.01124.pdf) (Medhini Narasimhan, ECCV 2018)

[Out of the Box: Reasoning with Graph Convolution Nets for Factual Visual Question Answering](https://arxiv.org/pdf/1811.00538.pdf) (Medhini Narasimhan, NIPS 2018)

[OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge](https://arxiv.org/pdf/1906.00067.pdf) (Kenneth Marino, CVPR 2019, [code](https://okvqa.allenai.org/), [note](https://blog.csdn.net/z704630835/article/details/100095787))

[KnowIT VQA: Answering Knowledge-Based Questions about Videos](https://ojs.aaai.org/index.php/AAAI/article/view/6713) (Noa Garcia, AAAI 2020)

[BERT Representations for Video Question Answering](http://openaccess.thecvf.com/content_WACV_2020/html/Yang_BERT_representations_for_Video_Question_Answering_WACV_2020_paper.html) (Zekun Yang, WACV 2020)

#### Knowledge Representation and Reasoning

##### Knowledge Base

[DBpedia: A Nucleus for a Web of Open Data](https://cis.upenn.edu/~zives/research/dbpedia.pdf) (Soren Auer, 2007, [code](https://wiki.dbpedia.org/))

[Freebase: A Collaboratively Created Graph Database For Structuring Human Knowledge](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.538.7139&rep=rep1&type=pdf) (Kurt Bollacker, 2008, [code](https://developers.google.com/freebase/))

[CN-DBpedia: A Never-Ending Chinese Knowledge Extraction System](https://link.springer.com/chapter/10.1007/978-3-319-60045-1_44) (Bo Xu, IEA-AIE 2017, [code](http://openkg.cn/dataset/cndbpedia), [note](https://blog.csdn.net/u013007703/article/details/90376440))

[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/pdf/1612.03975.pdf) (Robyn Speer, AAAI 2017, [code](http://conceptnet.io/))

[ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning](https://arxiv.org/pdf/1811.00146.pdf) (Maarten Sap, AAAI 2019)

[GenericsKB: A Knowledge Base of Generic Statements](https://arxiv.org/pdf/2005.00660.pdf) (Sumithra Bhakthavatsalam, 2020, [code](https://allenai.org/data/genericskb))

##### Knowledge Base Construction

[COMET : Commonsense Transformers for Automatic Knowledge Graph Construction](https://arxiv.org/pdf/1906.05317.pdf) (Antoine Bosselut, ACL 2019)

##### Knowledge Graph Embedding and Completion

[Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) (Antoine Bordes, NIPS 2013, [code](https://github.com/thunlp/KB2E), [note](https://zhuanlan.zhihu.com/p/32993044))

[Knowledge Graph Embedding by Translating on Hyperplanes](https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf) (Zhen Wang, AAAI 2014, [code](https://github.com/thunlp/KB2E), [note](https://zhuanlan.zhihu.com/p/32993044))

[Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.698.8922&rep=rep1&type=pdf) (Yankai Lin, AAAI 2015, [code](https://github.com/thunlp/KB2E), [note](https://zhuanlan.zhihu.com/p/32993044))

[Knowledge Graph Embedding via Dynamic Mapping Matrix](http://anthology.aclweb.org/P/P15/P15-1067.pdf) (Guoliang Ji, ACL 2015, [code](https://github.com/thunlp/OpenKE), [note](https://zhuanlan.zhihu.com/p/32993044))

[TransA: An Adaptive Approach for Knowledge Graph Embedding](https://arxiv.org/pdf/1509.05490.pdf) (Han Xiao, 2015, [note](https://blog.csdn.net/junruitian/article/details/87006668))

[Modeling Relation Paths for Representation Learning of Knowledge Bases](https://arxiv.org/pdf/1506.00379.pdf) (Yankai Lin, EMNLP 2015, [code](https://github.com/thunlp/KB2E), [note](https://www.jianshu.com/p/c3ace92cd6ef))

[TransG : A Generative Model for Knowledge Graph Embedding](http://www.aclweb.org/anthology/P16-1219) (Han Xiao, ACL 2016, [code](https://github.com/BookmanHan/Embedding), [note](https://blog.csdn.net/junruitian/article/details/87006668))

[Knowledge Graph Completion with Adaptive Sparse Transfer Matrix](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693) (Guoliang Ji, AAAI 2016, [code](https://github.com/thunlp/Fast-TransX), [note](https://blog.csdn.net/qq_36426650/article/details/103483838))

[Knowledge Graph Embedding: A Survey of Approaches and Applications](https://www.computer.org/csdl/trans/tk/2017/12/08047276-abs.html) (Quan Wang, TKDE 2017, [note](https://zhuanlan.zhihu.com/p/106024679))

[Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/pdf/1707.01476.pdf) (Pasquale Minervini, AAAI 2018, [code](https://github.com/TimDettmers/ConvE), [note](https://blog.csdn.net/damuge2/article/details/87974995))

[Open-World Knowledge Graph Completion](https://arxiv.org/pdf/1711.03438.pdf) (Baoxu Shi, AAAI 2018, [code](https://github.com/bxshi/ConMask), [note](https://juewang.me/posts/[2018.2.26]Open-World-Knowledge-Graph-Completion/))

[One-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/1808.09040.pdf) (Wenhan Xiong, EMNLP 2018, [code](https://github.com/xwhan/One-shot-Relational-Learning), [note](https://zhuanlan.zhihu.com/p/59646318))

##### Entity Discovery and Linking

[A Generative Entity-Mention Model for Linking Entities with Knowledge Base](https://www.aclweb.org/anthology/P/P11/P11-1095.pdf) (Xianpei Han, ACL 2011, [note](https://www.cnblogs.com/dhName/p/11078630.html))

[Overview of TAC-KBP2014 Entity Discovery and Linking Tasks](http://nlp.cs.rpi.edu/paper/edl2014overview.pdf) (Heng Ji, TAC 2014)

[An Attentive Neural Architecture for Fine-grained Entity Type Classification](https://www.aclweb.org/anthology/W16-1313) (Sonse Shimaoka, 2016, [note](https://blog.csdn.net/weixin_40485502/article/details/104019427))

[Neural Architectures for Fine-grained Entity Type Classification](https://www.aclweb.org/anthology/E17-1119) (Sonse Shimaoka, EACL 2017, [code](https://github.com/shimaokasonse/NFGEC))

[Fine-Grained Entity Type Classification by Jointly Learning Representations and Label Embeddings](http://aclweb.org/anthology/E17-1075) (Abhishek Abhishek, EACL 2017, [code](https://github.com/abhipec/fnet))

[Neural Fine-Grained Entity Type Classification with Hierarchy-Aware Loss](http://aclweb.org/anthology/N18-1002) (Peng Xu, NAACL 2018, [code](https://github.com/billy-inn/NFETC))

[Ultra-fine entity typing](https://www.aclweb.org/anthology/P18-1009) (Eunsol Choi, ACL 2018, [note](https://blog.csdn.net/xff1994/article/details/90293957))

##### Entity Set Expansion

[Web-Scale Distributional Similarity and Entity Set Expansion](https://aclanthology.org/D09-1098.pdf) (Patrick Pantel, EMNLP 2009)

[EgoSet: Exploiting Word Ego-networks and User-generated Ontology for Multifaceted Set Expansion](https://dl.acm.org/doi/abs/10.1145/2835776.2835808?casa_token=WO5mp1Qk4WMAAAAA:OWY3rVjXu57DFQafsDOP6mz3u6GROVN-Z1O9uvRPNgZ6-IYVlFY_jN5yXmYHjkcI63NfabkWOa8D1lU) (Xin Rong, WSDM 2016)

[SetExpan: Corpus-Based Set Expansion via Context Feature Selection and Rank Ensemble](https://link.springer.com/chapter/10.1007/978-3-319-71249-9_18) (Jiaming Shen, ECML PKDD 2017)

[HiExpan: Task-Guided Taxonomy Construction by Hierarchical Tree Expansion](https://dl.acm.org/doi/abs/10.1145/3219819.3220115) (Jiaming Shen, KDD 2018)

[TaxoExpan: Self-supervised Taxonomy Expansion with Position-Enhanced Graph Neural Network](https://dl.acm.org/doi/abs/10.1145/3366423.3380132?casa_token=ZxAWXTyFuO4AAAAA:J_673jkMVGIhJvBkHkyQGGlz6KeuD7aFghWvN7ARUVKqxABsD8G-at1HzLefvofpf_zo_xyAogcMfEI) (Jiaming Shen, WWW 2020)

[Empower Entity Set Expansion via Language Model Probing](https://arxiv.org/pdf/2004.13897.pdf) (Yunyi Zhang, ACL 2020)

##### Causal Knowledge

[Automatic Extraction of Causal Relations from Natural Language Texts: A Comprehensive Survey](https://arxiv.org/pdf/1605.07895.pdf) (Nabiha Asghar, 2016)

[Guided Generation of Cause and Effect](https://www.ijcai.org/Proceedings/2020/0502.pdf) (Zhongyang Li, IJCAI 2020)

##### Knowledge Graph Application

[Learning beyond datasets: Knowledge Graph Augmented Neural Networks for Natural language Processing](https://arxiv.org/pdf/1802.05930.pdf) (Annervaz K M, NAACL 2018, [note](https://blog.lorrin.info/posts/%5B2018.5.10%5DKnowledge-Graph-Augmented-Neural-Networks-for-NLP/))

[Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](http://aclweb.org/anthology/P18-1223) (Zhenghao Liu, ACL 2018, [code](https://github.com/thunlp/EntityDuetNeuralRanking), [note](https://blog.csdn.net/weixin_43087818/article/details/103764135))

#### Coreference Resolution

[Deep Reinforcement Learning for Mention-Ranking Coreference Models](https://www.aclweb.org/anthology/D16-1245.pdf) (Kevin Clark, EMNLP 2016, [code](https://github.com/clarkkev/deep-coref))

[Improving Coreference Resolution by Learning Entity-Level Distributed Representations](https://nlp.stanford.edu/pubs/clark2016improving.pdf) (Kevin Clark, ACL 2016, [code](https://github.com/clarkkev/deep-coref), [note](https://zhuanlan.zhihu.com/p/97097668))

[Higher-order Coreference Resolution with Coarse-to-fine Inference](https://www.aclweb.org/anthology/N18-2108.pdf) (Kenton Lee, NAACL 2018, [code](https://github.com/kentonl/e2e-coref), [note](https://zhuanlan.zhihu.com/p/93900881))

[Learning Word Representations with Cross-Sentence Dependency for End-to-End Co-reference Resolution](https://www.aclweb.org/anthology/D18-1518.pdf) (Hongyin Luo, EMNLP 2018)

[BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/pdf/1908.09091.pdf) (Mandar Joshi, EMNLP 2019, [code](https://github.com/mandarjoshi90/coref), [note](https://blog.csdn.net/BeiLi_ShanGui/article/details/103156488))

[GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue](https://arxiv.org/pdf/1909.12086.pdf) (Jun Quan, 2019, [note](https://blog.csdn.net/lwgkzl/article/details/102482928))

[Incorporating Structural Information for Better Coreference Resolution](https://www.ijcai.org/Proceedings/2019/0700.pdf) (Kong Fang, IJCAI 2019)

[End-to-end Deep Reinforcement Learning Based Coreference Resolution](https://www.aclweb.org/anthology/P19-1064.pdf) (Hongliang Fei, ACL 2019)

[The Referential Reader: A Recurrent Entity Network for Anaphora Resolution](https://www.aclweb.org/anthology/P19-1593.pdf) (Fei Liu, ACL 2019)

##### Pronoun Resolution

[Commonsense Knowledge Enhanced Embeddings for Solving Pronoun Disambiguation Problems in Winograd Schema Challenge](https://arxiv.org/pdf/1611.04146.pdf) (Quan Liu, 2016, [note](https://www.jianshu.com/p/b94bcacce74c))

[WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://arxiv.org/pdf/1908.08025.pdf) (Vid Kocijan, 2019)

[Look Again at the Syntax: Relational Graph Convolutional Network for Gendered Ambiguous Pronoun Resolution](https://arxiv.org/pdf/1905.08868.pdf) (Yinchuan Xu, 2019, [code](https://github.com/ianycxu/RGCN-with-BERT))

[Incorporating Context and External Knowledge for Pronoun Coreference Resolution](https://www.aclweb.org/anthology/N19-1093.pdf) (Hongming Zhang, NAACL 2019, [code](https://github.com/HKUST-KnowComp/Pronoun-Coref), [note](https://zhuanlan.zhihu.com/p/88020550))

[Knowledge-aware Pronoun Coreference Resolution](https://www.aclweb.org/anthology/P19-1083.pdf) (Hongming Zhang, ACL 2019, [code](https://github.com/HKUST-KnowComp/Pronoun-Coref-KG), [note](https://zhuanlan.zhihu.com/p/85180047))

[What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues](https://arxiv.org/pdf/1909.00421.pdf) (Xintong Yu, EMNLP 2019, [note](https://zhuanlan.zhihu.com/p/91231002))

##### Zero Pronoun Resolution

[Identification and Resolution of Chinese Zero Pronouns: A Machine Learning Approach](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=F05E1DD4B64B0771E279426984E7CDD1?doi=10.1.1.65.1935&rep=rep1&type=pdf) (Shanheng Zhao and Hwee Tou Ng, EMNLP 2007)

[Chinese Zero Pronoun Resolution: A Joint Unsupervised Discourse-Aware Model Rivaling State-of-the-Art Resolvers](http://www.aclweb.org/anthology/P15-2053) (Chen Chen and Vincent Ng, ACL 2015)

[Chinese Zero Pronoun Resolution with Deep Neural Networks](http://www.aclweb.org/anthology/P16-1074) (Chen Chen and Vincent Ng, ACL 2016)

[Chinese Zero Pronoun Resolution with Deep Memory Network](http://aclweb.org/anthology/D17-1135) (Qingyu Yin, EMNLP 2017)

[A Deep Neural Network for Chinese Zero Pronoun Resolution](https://arxiv.org/pdf/1604.05800.pdf) (Qingyu Yin, IJCAI 2017)

[Generating and Exploiting Large-Scale Pseudo Training Data for Zero Pronoun Resolution](https://arxiv.org/pdf/1606.01603.pdf) (Ting Liu, ACL 2017, [note](https://zhuanlan.zhihu.com/p/136544141))

[Deep Reinforcement Learning for Chinese Zero pronoun Resolution](http://aclweb.org/anthology/P18-1053) (Qingyu Yin, ACL 2018, [code](https://github.com/qyyin/Reinforce4ZP), [note](https://www.jiqizhixin.com/articles/2018-05-21-6))

[Zero Pronoun Resolution with Attention-based Neural Network](http://aclweb.org/anthology/C18-1002) (Qingyu Yin, COLING 2018, [code](https://github.com/qyyin/AttentionZP), [note](https://www.jiqizhixin.com/articles/2018-07-28-8))

[Hierarchical Attention Network with Pairwise Loss for Chinese Zero Pronoun Resolution](https://144.208.67.177/ojs/index.php/AAAI/article/view/6352) (Peiqin Lin, AAAI 2020, [code](https://github.com/lpq29743/HAN-PL), [note](https://zhuanlan.zhihu.com/p/151387067))

#### Natural Language Generation

[Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation](https://arxiv.org/pdf/1703.09902.pdf) (Albert Gatt, 2017)

[Neural Text Generation: A Practical Guide](https://arxiv.org/pdf/1711.09534.pdf) (Ziang Xie, 2017)

[A Hybrid Convolutional Variational Autoencoder for Text Generation](https://arxiv.org/pdf/1702.02390.pdf) (Stanislau Semeniuta, 2017)

[Natural Language Generation by Hierarchical Decoding with Linguistic Patterns](https://arxiv.org/pdf/1808.02747.pdf) (Shang-Yu Su, NAACL 2018)

[Topic-Guided Variational Autoencoders for Text Generation](https://arxiv.org/pdf/1903.07137.pdf) (Wenlin Wang, NAACL 2019)

[Generating Long and Informative Reviews with Aspect-Aware Coarse-to-Fine Decoding](https://arxiv.org/pdf/1906.05667.pdf) (Junyi Li, ACL 2019)

[Syntax-Infused Variational Autoencoder for Text Generation](https://arxiv.org/pdf/1906.02181.pdf) (Xinyuan Zhang, ACL 2019)

[Towards Generating Long and Coherent Text with Multi-Level Latent Variable Models](https://arxiv.org/pdf/1902.00154.pdf) (Dinghan Shan, ACL 2019)

[Keeping Notes: Conditional Natural Language Generation with a Scratchpad Mechanism](https://www.aclweb.org/anthology/P19-1407.pdf) (Ryan Y. Benmalek, ACL 2019)

[Long and Diverse Text Generation with Planning-based Hierarchical Variational Model](https://arxiv.org/pdf/1908.06605.pdf) (Zhihong Shao, EMNLP 2019)

[Best-First Beam Search](https://arxiv.org/pdf/2007.03909.pdf) (Clara Meister, TACL 2020, [code](https://github.com/huggingface/transformers/issues/6565), [note](https://zhuanlan.zhihu.com/p/187270580))

[The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf) (Ari Holtzman, ICLR 2020, [note](https://zhuanlan.zhihu.com/p/115076102))

[Neural Text Generation With Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf) (Sean Welleck, [note](https://zhuanlan.zhihu.com/p/78695564))

[The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics](https://arxiv.org/pdf/2102.01672.pdf) (Sebastian Gehrmann, 2021)

[Neural Text Generation with Part-of-Speech Guided Softmax](https://arxiv.org/pdf/2105.03641.pdf) (Zhixian Yang, 2021)

##### Automatic Metric

[Binary Codes Capable of Correcting Deletions, Insertions and Reversals](https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf) (VI Levenshtein, 1966)

[A New Quantitative Quality Measure for Machine Translation Systems](https://www.aclweb.org/anthology/C92-2067.pdf) (Keh-Yih Su, COLING 1992)

[An Evaluation Tool for Machine Translation: Fast Evaluation for MT Research](http://www.lrec-conf.org/proceedings/lrec2000/pdf/278.pdf) (Sonja Nießen, LREC 2000)

[Using Multiple Edit Distances to Automatically Rank Machine Translation Output](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.590.6755&rep=rep1&type=pdf) (Yasuhiro Akiba, 2001)

[BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P/P02/P02-1040.pdf) (Kishore Papineni, ACL 2002, [note](https://www.jianshu.com/p/320ffec4e99f))

[Automatic Evaluation of Machine Translation Quality using N-gram CoOccurrence Statistics](https://dl.acm.org/doi/abs/10.5555/1289189.1289273) (George R Doddington, 2002)

[A Novel String-to-String Distance Measure with Applications to Machine Translation Evaluation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.4.522&rep=rep1&type=pdf) (Gregor Leusch, 2003)

[ROUGE: A Package for Automatic Evaluation of Summaries](http://www.aclweb.org/anthology/W/W04/W04-1013.pdf) (Chin-Yew Lin, 2004)

[METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://www.aclweb.org/anthology/W05-0909.pdf) (Satanjeev Banerjee, ACL 2005 Workshop)

[∆BLEU: A Discriminative Metric for Generation Tasks with Intrinsically Diverse Targets](https://www.aclweb.org/anthology/P15-2073.pdf) (Michel Galley, ACL 2015)

[Sentence Mover’s Similarity: Automatic Evaluation for Multi-Sentence Texts](https://www.aclweb.org/anthology/P19-1264.pdf) (Elizabeth Clark, ACL 2019)

[MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/pdf/1909.02622.pdf) (Wei Zhao, EMNLP 2019)

[BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf) (Tianyi Zhang, ICLR 2020)

[BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/pdf/2004.04696.pdf) (Thibault Sellam, ACL 2020)

[Evaluation of Text Generation: A Survey](https://arxiv.org/pdf/2006.14799.pdf) (Asli Celikyilmaz, 2020)

##### Sequence to Sequence

[Convolutional Sequence to Sequence Learning](https://pdfs.semanticscholar.org/bb3e/bc09b65728d6eced04929df72a006fb5210b.pdf) (Jonas Gehring, ICML 2017, [code](https://github.com/tobyyouup/conv_seq2seq), [note](https://zhuanlan.zhihu.com/p/26918935))

[Deliberation Networks: Sequence Generation Beyond One-Pass Decoding](https://papers.nips.cc/paper/2017/file/c6036a69be21cb660499b75718a3ef24-Paper.pdf) (Yingce Xia, NIPS 2017)

[Deep Reinforcement Learning For Sequence to Sequence Models](https://arxiv.org/pdf/1805.09461.pdf) (Yaser Keneshloo, 2018, [code](https://github.com/yaserkl/RLSeq2Seq), [note](https://www.jianshu.com/p/1213de861491))

[MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/pdf/1905.02450.pdf) (Kaitao Song, 2019, [code](https://github.com/microsoft/MASS), [note](https://zhuanlan.zhihu.com/p/71022527))

##### Graph to Sequence

[Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/pdf/1904.02342.pdf) (Rik Koncel-Kedziorski, NAACL 2019, [code](https://github.com/rikdz/GraphWriter), [note](https://zhuanlan.zhihu.com/p/90084109))

##### Controlled Text Generation

[Toward Controlled Generation of Text](https://arxiv.org/pdf/1703.00955.pdf) (Zhiting Hu, 2017)

[T-CVAE: Transformer-Based Conditioned Variational Autoencoder for Story Completion](https://www.ijcai.org/proceedings/2019/0727.pdf) (Tianming Wang, IJCAI 2019, [note](https://zhuanlan.zhihu.com/p/91166636))

[Plug and Play Language Models: a Simple Approach to Controlled Text Generation](https://arxiv.org/pdf/1912.02164.pdf) (Sumanth Dathathri, ICLR 2020)

[DEXPERTS: On-the-Fly Controlled Text Generation with Experts and Anti-Experts](https://arxiv.org/pdf/2105.03023.pdf) (Alisa Liu, ACL 2021)

##### AMR-to-Text Generation

[Modeling Graph Structure in Transformer for Better AMR-to-Text Generation](https://arxiv.org/pdf/1909.00136.pdf) (Jie Zhu, EMNLP 2019)

[Enhancing AMR-to-Text Generation with Dual Graph Representations](https://arxiv.org/pdf/1909.00352.pdf) (Leonardo F. R. Ribeiro, EMNLP 2019, [code](https://github.com/UKPLab/emnlp2019-dualgraph), [note](https://zhuanlan.zhihu.com/p/105701258))

[Line Graph Enhanced AMR-to-Text Generation with Mix-Order Graph Attention Networks](https://www.aclweb.org/anthology/2020.acl-main.67.pdf) (Yanbin Zhao, ACL 2020)

[GPT-too: A language-model-first approach for AMR-to-text generation](https://arxiv.org/pdf/2005.09123.pdf) (Manuel Mager, ACL 2020, [code](https://github.com/IBM/GPT-too-AMR2text))

##### Data-to-Text Generation

[Data-to-Text Generation with Content Selection and Planning](https://www.aaai.org/ojs/index.php/AAAI/article/view/4668) (Ratish Puduppully, AAAI 2019, [code](https://github.com/ratishsp/data2text-plan-py), [note](https://zhuanlan.zhihu.com/p/85275520))

[Data-to-text Generation with Entity Modeling](https://arxiv.org/pdf/1906.03221.pdf) (Ratish Puduppully, ACL 2019, [code](https://github.com/ratishsp/data2text-entity-py), [note](https://zhuanlan.zhihu.com/p/82054729))

[Table-to-Text Generation with Effective Hierarchical Encoder on Three Dimensions (Row, Column and Time)](https://arxiv.org/pdf/1909.02304.pdf) (Heng Gong, EMNLP 2019)

#### Machine Translation

[Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/pdf/1406.1078.pdf) (Kyunghyun Cho, EMNLP 2014, [note](https://cuiqingcai.com/5737.html))

[On the properties of neural machine translation: Encoder–Decoder approaches](https://arxiv.org/pdf/1409.1259.pdf) (Kyunghyun Cho, 2014, [note](https://blog.csdn.net/BeforeEasy/article/details/80332497))

[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf) (Rico Sennrich, ACL 2016, [code](https://github.com/rsennrich/subword-nmt), [note](https://zhuanlan.zhihu.com/p/38574684))

[Modeling Coverage for Neural Machine Translation](https://www.aclweb.org/anthology/P16-1008.pdf) (Zhaopeng Tu, ACL 2016, [note](https://zhuanlan.zhihu.com/p/22993927))

[Does String-Based Neural MT Learn Source Syntax?](https://aclanthology.org/D16-1159.pdf) (Xing Shi, EMNLP 2016)

[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) (Yonghui Wu, 2016, [note](https://blog.csdn.net/Xiao_yanling/article/details/90290862))

[Neural Machine Translation with Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/10950) (Zhaopeng Tu, AAAI 2017)

[Sequence-to-Dependency Neural Machine Translation](http://www.aclweb.org/anthology/P17-1065) (Shuangzhi Wu, ACL 2017, [note](https://www.jianshu.com/p/2eb3c89234cb))

[Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf) (Graham Neubig, 2017, [note](https://google.github.io/seq2seq/nmt/))

[Multi-channel Encoder for Neural Machine Translation](https://arxiv.org/pdf/1712.02109.pdf) (Hao Xiong, AAAI 2018, [note](https://www.jiqizhixin.com/articles/2017-12-14-10))

[Translating Pro-Drop Languages with Reconstruction Models](https://ojs.aaai.org/index.php/AAAI/article/view/11913) (Longyue Wang, AAAI 2018)

[Unsupervised Neural Machine Translation](https://arxiv.org/pdf/1710.11041.pdf) (Mikel Artetxe, ICLR 2018, [code](https://github.com/artetxem/undreamt), [note](https://zhuanlan.zhihu.com/p/30649985))

[Non-Autoregressive Neural Machine Translation](https://arxiv.org/pdf/1711.02281.pdf) (Jiatao Gu, ICLR 2018, [code](https://github.com/salesforce/nonauto-nmt), [note](https://zhuanlan.zhihu.com/p/35866317))

[Word Translation Without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf) (Alexis Conneau, ICLR 2018)

[Learning to Jointly Translate and Predict Dropped Pronouns with a Shared Reconstruction Mechanism](https://arxiv.org/pdf/1810.06195.pdf) (Longyue Wang, EMNLP 2018)

[Phrase-Based & Neural Unsupervised Machine Translation](https://arxiv.org/pdf/1804.07755.pdf) (Guillaume Lample, EMNLP 2018, [code](https://github.com/facebookresearch/UnsupervisedMT), [note](https://blog.csdn.net/ljp1919/article/details/103074097))

[Rapid Adaptation of Neural Machine Translation to New Languages](https://arxiv.org/pdf/1808.04189.pdf) (Graham Neubig, EMNLP 2018)

[A Survey of Domain Adaptation for Neural Machine Translation](https://arxiv.org/pdf/1806.00258.pdf) (Chenhui Chu, COLING 2018)

[Pivot-based Transfer Learning for Neural Machine Translation between Non-English Languages](https://arxiv.org/pdf/1909.09524.pdf) (Yunsu Kim, EMNLP 2019)

[Mixed Multi-Head Self-Attention for Neural Machine Translation](https://www.aclweb.org/anthology/D19-5622.pdf) (Hongyi Cui, EMNLP 2019)

[Simple, Scalable Adaptation for Neural Machine Translation](https://arxiv.org/pdf/1909.08478.pdf) (Ankur Bapna, EMNLP 2019, [note](https://zhuanlan.zhihu.com/p/114955522))

[Dynamically Composing Domain-Data Selection with Clean-Data Selection by "Co-Curricular Learning" for Neural Machine Translation](https://arxiv.org/pdf/1906.01130.pdf) (Wei Wang, ACL 2019)

[Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies](https://arxiv.org/pdf/1905.05475.pdf) (Yunsu Kim, ACL 2019, [code](https://github.com/yunsukim86/sockeye-transfer))

[Latent Variable Model for Multi-modal Translation](https://www.aclweb.org/anthology/P19-1642.pdf) (Iacer Calixto, ACL 2019)

[Distilling Translations with Visual Awareness](https://www.aclweb.org/anthology/P19-1653.pdf) (Julia Ive, ACL 2019)

[Reducing Word Omission Errors in Neural Machine Translation: A Contrastive Learning Approach](https://www.aclweb.org/anthology/P19-1623.pdf) (Zonghan Yang, ACL 2019)

[Cross-Lingual Pre-Training Based Transfer for Zero-Shot Neural Machine Translation](https://www.aaai.org/ojs/index.php/AAAI/article/view/5341) (Baijun Ji, AAAI 2020)

[Mirror-Generative Neural Machine Translation](https://openreview.net/pdf?id=HkxQRTNYPH) (Zaixiang Zheng, ICLR 2020)

[Learning a Multi-Domain Curriculum for Neural Machine Translation](https://arxiv.org/pdf/1908.10940.pdf) (Wei Wang, ACL 2020)

[Multimodal Transformer for Multimodal Machine Translation](https://www.aclweb.org/anthology/2020.acl-main.400.pdf) (Shaowei Yao, ACL 2020)

[Exploring Supervised and Unsupervised Rewards in Machine Translation](https://arxiv.org/pdf/2102.11403.pdf) (Julia Ive, EACL 2020)

[Context-Aware Cross-Attention for Non-Autoregressive Translation](https://arxiv.org/pdf/2011.00770.pdf) (Liang Ding, COLING 2020)

##### Word Alignment

[The Mathematics of Statistical Machine Translation: Parameter Estimation](https://aclanthology.org/J93-2003.pdf) (Peter E Brown, CL 1993)

[A Systematic Comparison of Various Statistical Alignment Models](https://watermark.silverchair.com/089120103321337421.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAs8wggLLBgkqhkiG9w0BBwagggK8MIICuAIBADCCArEGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMkCEvPs_rjOQ1vujTAgEQgIICgq9VNVpiuBa5_Kw7zn35yQ0srXxnm6UXC7v1TCL2et0s17aTNOiFRpU2Eyi32wUlJPZ4IImNscvUcXbI-h_I1n5DO16iUfRadsoF4wlP-jBkuJ6Hi-wVp4L9sRcQ2ivzLj2QsMvBy-ZyRgoWP91RaQTeV4kkD7ggRB03n5ujWF0IDtn-83BA-oHYhrFAzfyo8mIxGV9PFqfymtZjg-61JdU1I2OHkXPQT-SaGIy1CWJyuLbMmg1bm_IsuhiQPrfZDq2V7I5XXjxH7k024-RD3oJ17wTN53W6f7YbMnHHilVSzmDfqZE0QdM_HF47dklvse4ZD_0qBnDl_APKFG7ctloY3T19fznYviPcF5zSFbyyax3FKjw1DFYqR80FsuRZmzzuTiGw156Ntml_SfE9dObmdE6L8la8kWzHaaa1uo4_NgZ_0qSeLEmFH9hQB7MJZZbwKDjEj8_0M6deMLTtHkZ1OUiqMRgdP6mNa4JHOf3R1vz8LFDgiEVOyR9LN-KUffamhjpN4ZDf47Xw9YS9VLGlYAwOiAivqTM0InPjW8kEwXP07_VX2n1HbY21FYE1cvZkRb-V-KfORmjj3zWqNGKO-EUXMtoXUF2G1JyNfYMhMEvgL-IsCv39SUTiWhO_bK2fxBb7j7JcrRDuRQ_RapQ06wSJWXshvvap2QLg5CFvLBSZIUT1d6d6GNLuYu0WhW9T_PiWuJSbwFpPjbRdcYTGt-JEFfNJqTdBWsG4tzePJ6hcm3K6R93ytOxV1TjfLb4LIJ2vGTSZFukJbToKdpEuOjbmCzBtM4Ua275bddrQXVinUUCpS52KFAfIji0g2dET_1GkKpR2JjE_ThXgVFRMwg) (Franz Josef Och, CL 2013)

[A Simple, Fast, and Effective Reparameterization of IBM Model 2](https://aclanthology.org/N13-1073.pdf) (Chris Dyer, NAACL 2013)

[A Systematic Bayesian Treatment of the IBM Alignment Models](https://aclanthology.org/N13-1117.pdf) (Yarin Gal, NAACL 2013)

[Word Alignment Modeling with Context Dependent Deep Neural Network](https://aclanthology.org/P13-1017.pdf) (Nan Yang, ACL 2013)

[Recurrent Neural Networks for Word Alignment Model](https://aclanthology.org/P14-1138.pdf) (Akihiro Tamura, ACL 2014)

[MASK-ALIGN: Self-Supervised Neural Word Alignment](https://arxiv.org/pdf/2012.07162.pdf) (Chi Chen, ACL 2021, [code](https://github.com/THUNLP-MT/Mask-Align))

##### Low-Resource Machine Translation

[Generalized Data Augmentation for Low-Resource Translation](https://arxiv.org/pdf/1906.03785.pdf) (Mengzhou Xia, ACL 2019, [code](https://github.com/xiamengzhou/DataAugForLRL))

[Revisiting Low-Resource Neural Machine Translation: A Case Study](https://arxiv.org/pdf/1905.11901.pdf) (Rico Sennrich, ACL 2019)

[Handling Syntactic Divergence in Low-resource Machine Translation](https://arxiv.org/pdf/1909.00040.pdf) (Chunting Zhou, EMNLP 2019)

[The FLoRes Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English](https://arxiv.org/pdf/1902.01382.pdf) (Francisco Guzman, EMNLP 2019, [code](https://github.com/facebookresearch/flores))

##### Multilingual Machine Translation

[Multilingual Neural Machine Translation with Knowledge Distillation](https://arxiv.org/pdf/1902.10461.pdf) (Xu Tan, ICLR 2019, [code](https://github.com/RayeRen/multilingual-kd-pytorch))

[Multilingual Neural Machine Translation With Soft Decoupled Encoding](https://arxiv.org/pdf/1902.03499.pdf) (Xinyi Wang, ICLR 2019, [code](https://github.com/cindyxinyiwang/SDE), [note](https://zhuanlan.zhihu.com/p/60845246))

[Massively Multilingual Neural Machine Translation](https://arxiv.org/pdf/1903.00089.pdf) (Roee Aharoni, NAACL 2019)

[Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation](https://arxiv.org/pdf/1905.08212.pdf) (Xinyi Wang, ACL 2019)

[Multilingual Neural Machine Translation with Language Clustering](https://arxiv.org/pdf/1908.09324.pdf) (Xu Tan, EMNLP 2019)

[Balancing Training for Multilingual Neural Machine Translation](https://arxiv.org/pdf/2004.06748.pdf) (Xinyi Wang, ACL 2020)

[Knowledge Distillation for Multilingual Unsupervised Neural Machine Translation](https://arxiv.org/pdf/2004.10171.pdf) (Haipeng Sun, ACL 2020)

#### Paraphrase Generation

[Submodular Optimization-based Diverse Paraphrasing and its Effectiveness in Data Augmentation](https://www.aclweb.org/anthology/N19-1363.pdf) (Ashutosh Kumar, NAACL 2019, [code](https://github.com/malllabiisc/DiPS))

[A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](https://arxiv.org/pdf/1904.01173.pdf) (Mingda Chen, NAACL 2019, [code](https://github.com/mingdachen/disentangle-semantics-syntax))

[Paraphrase Generation with Latent Bag of Words](https://arxiv.org/pdf/2001.01941.pdf) (Yao Fu, NeuIPS 2019)

#### Storytelling

[Content Learning with Structure-Aware Writing: A Graph-Infused Dual Conditional Variational Autoencoder for Automatic Storytelling](https://www.aaai.org/AAAI21Papers/AAAI-10130.YuMH.pdf) (Meng-Hsuan Yu, AAAI 2021)

#### Natural Language Processing for Programming Language

[code2seq: Generating sequences from structured representations of code](https://arxiv.org/pdf/1808.01400.pdf) (Uri Alon, ICLR 2019, [code](https://github.com/tech-srl/code2seq))

##### Code Comment Generation

[Towards automatically generating summary comments for java methods](https://dl.acm.org/doi/abs/10.1145/1858996.1859006) (Giriprasad Sridhara, ASE 2010)

[On automatically generating commit messages via summarization of source code changes](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.714.7445&rep=rep1&type=pdf) (Luis Fernando Cortés-Coy, SCAM 2014)

[Source code analysis extractive approach to generate textual summary](http://www.jatit.org/volumes/Vol95No21/12Vol95No21.pdf) (Kareem Abbas Dawood, 2017)

##### Code Retrieval

[Deep code search](https://guxd.github.io/papers/deepcs.pdf) (Xiaodong Gu, ICSE 2018)

#### Interpretability in Natural Language Processing

[What Does BERT Look At? An Analysis of BERT’s Attention](https://arxiv.org/pdf/1906.04341.pdf) (Kevin Clark, ACL 2019 Workshop)

[Revealing the Dark Secrets of BERT](https://arxiv.org/pdf/1908.08593.pdf) (Olga Kovaleva, EMNLP 2019)

#### Fairness in Natural Language Processing

[Learning Gender-Neutral Word Embeddings](https://arxiv.org/pdf/1809.01496.pdf) (Jieyu Zhao, EMNLP 2018)

### Computer Vision

#### Feature Detector and Descriptor

[Towards Automatic Visual Obstacle Avoidance](https://philpapers.org/rec/MORTTA-3) (Hans P. Moravec, 1977)

[A Computational Approach to Edge Detection](https://ieeexplore.ieee.org/document/4767851) (John Canny, TPAMI 1986)

[A Combined Corner and Edge Detector](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.434.4816&rep=rep1&type=pdf) (Chris Harris, 1988)

[Scale-Space and Edge Detection Using Anisotropic Diffusion](https://ieeexplore.ieee.org/document/56205) (Pietro Perona, TPAMI 1990)

[SUSAN—A New Approach to Low Level Image Processing](https://link.springer.com/article/10.1023/A:1007963824710) (Stephen M. Smith, 1997)

[Object Recognition from Local Scale-Invariant Features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf) (David G. Lowe, ICCV 1999)

[Multiresolution Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/lbp.pdf) (Timo Ojala, TPAMI 2002)

[Robust Wide Baseline Stereo from Maximally Stable Extremal Regions](https://www.sciencedirect.com/science/article/pii/S0262885604000435) (Jiri Matas, BMVC 2002)

[Image Registration Methods: A Survey](https://www.sciencedirect.com/science/article/pii/S0262885603001379) (Barbara Zitova, 2003)

[Distinctive Image Features from Scale-Invariant Keypoints](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf) (David G. Lowe, IJCV 2004)

[A Comparison of Affine Region Detectors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/vibes_ijcv2004.pdf) (Krystian Mikolajczyk, IJCV 2004)

[A Performance Evaluation of Local Descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf) (Krystian Mikolajczyk, TPAMI 2005)

[Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (Navneet Dalal, CVPR 2005)

[Machine Learning for High-Speed Corner Detection](https://link.springer.com/chapter/10.1007/11744023_34) (Edward Rosten, ECCV 2006)

[SURF: Speeded Up Robust Features](https://link.springer.com/chapter/10.1007/11744023_32) (Herbert Bay, ECCV 2006)

[CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching](https://link.springer.com/chapter/10.1007/978-3-540-88693-8_8) (Motilal Agrawal, ECCV 2008)

[Local Invariant Feature Detectors: A Survey](https://homes.esat.kuleuven.be/~tuytelaa/FT_survey_interestpoints08.pdf) (Tinne Tuytelaars, 2008)

[Discriminative Learning of Local Image Descriptors](http://matthewalunbrown.com/papers/pami2010.pdf) (Matthew Brown, TPAMI 2010)

[DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo](https://ieeexplore.ieee.org/document/4815264) (Engin Tola, TPAMI 2010)

[ORB: An efficient alternative to SIFT or SURF](https://ieeexplore.ieee.org/document/6126544) (Ethan Rublee, ICCV 2011)

[LIFT: Learned Invariant Feature Transform](https://arxiv.org/pdf/1603.09114.pdf) (Kwang Moo Yi, ECCV 2016)

#### Object Detection

[Rapid object detection using a boosted cascade of simple features](https://ieeexplore.ieee.org/abstract/document/990517) (CVPR 2001, Paul Viola)

[Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=03500C970796A39BCC5A437E4AA79B10?doi=10.1.1.165.9750&rep=rep1&type=pdf) (Christoph H. Lampert, CVPR 2009, [code](https://github.com/ahmedmazariML/Learning-To-Detect-Unseen-Object-Classes-by-Between-Class-Attribute-Transfer), [note](https://tongtianta.site/paper/23631))

[DeViSE: A Deep Visual-Semantic Embedding Model](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=CD1A6E4145A750B305F3512006D12FE9?doi=10.1.1.466.176&rep=rep1&type=pdf) (Andrea Frome, NeuIPS 2013, [note](https://zhuanlan.zhihu.com/p/52352455))

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf) (Ross Girshick, CVPR 2014, [code](https://github.com/rbgirshick/rcnn), [note](https://zhuanlan.zhihu.com/p/47579399))

[Fast Region-based Convolutional Networks for object detection](https://arxiv.org/pdf/1504.08083.pdf) (Ross Girshick, ICCV 2015, [code](https://github.com/rbgirshick/fast-rcnn), [note](https://zhuanlan.zhihu.com/p/47579399))

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) (Shaoqing Ren, NeuIPS 2015, [code](https://github.com/rbgirshick/py-faster-rcnn), [note](https://zhuanlan.zhihu.com/p/47579399))

[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409.pdf) (Jifeng Dai, NeuIPS 2016, [code](https://github.com/daijifeng001/R-FCN), [note](https://zhuanlan.zhihu.com/p/30867916))

[You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) (Joseph Redmon, CVPR 2016, [code](https://github.com/gliese581gg/YOLO_tensorflow), [note](https://zhuanlan.zhihu.com/p/32525231))

[SSD: Single Shot MultiBox Detector](http://www.cs.unc.edu/~cyfu/pubs/ssd.pdf) (Wei Liu, ECCV 2016, [code](https://github.com/balancap/SSD-Tensorflow), [note](https://zhuanlan.zhihu.com/p/33544892))

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf) (Joseph Redmon, CVPR 2017, [code](https://github.com/experiencor/keras-yolo2), [note](https://zhuanlan.zhihu.com/p/25052190))

[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) (Kaiming He, ICCV 2017, [code](https://github.com/facebookresearch/Detectron), [note](https://zhuanlan.zhihu.com/p/47579399))

[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) (Joseph Redmon, 2018, [code](https://github.com/qqwweee/keras-yolo3), [note](https://zhuanlan.zhihu.com/p/76802514))

[TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/pdf/1903.12174.pdf) (Xinlei Chen, 2019, [note](https://zhuanlan.zhihu.com/p/60984659))

[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf) (Alexey Bochkovskiy, 2020, [code](https://github.com/AlexeyAB/darknet?utm_source=catalyzex.com))

[RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder](https://arxiv.org/pdf/2010.15831.pdf) (Cheng Chi, NIPS 2020)

[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070v3.pdf) (Mingxing Tan, CVPR 2020)

#### Semantic Segmentation

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf) (Jonathan Long, CVPR 2015, [code](https://github.com/anoushkrit/Knowledge?utm_source=catalyzex.com), [note](https://blog.csdn.net/qq_36269513/article/details/80420363))

[Dual Attention Network for Scene Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/html/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.html) (Jun Fu, CVPR 2019)

#### Image Super-Resolution

[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf) (Chao Dong, TPAMI 2015)

[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf) (Wenzhe Shi, CVPR 2016, [code](https://github.com/leftthomas/ESPCN), [note](https://zhuanlan.zhihu.com/p/76338220))

[Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) (Jiwon Kim, CVPR 2016, [code](https://github.com/Jongchan/tensorflow-vdsr), [note](https://blog.csdn.net/shwan_ma/article/details/78193777))

[Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/pdf/1608.00367.pdf) (Chao Dong, ECCV 2016, [code](https://github.com/Saafke/FSRCNN_Tensorflow), [note](https://blog.csdn.net/shwan_ma/article/details/78171649))

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) (Justin Johnson, ECCV 2016, [code](https://github.com/yusuketomoto/chainer-fast-neuralstyle), [note](https://blog.csdn.net/kid_14_12/article/details/85871965))

[Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections](https://arxiv.org/pdf/1606.08921.pdf) (Xiao-Jiao Mao, 2016, [code](https://github.com/ved27/RED-net), [note](https://blog.csdn.net/happyday_d/article/details/85239395))

[Image Super-Resolution Using Dense Skip Connections](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) (Tong Tong, ICCV 2017, [code](https://github.com/kweisamx/TensorFlow-SR-DenseNet), [note](https://blog.csdn.net/happyday_d/article/details/85461715))

[Image Super-Resolution via Deep Recursive Residual Network](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) (Tai Yang, CVPR 2017, [code](https://github.com/tyshiwo/DRRN_CVPR17), [note](https://blog.csdn.net/wangkun1340378/article/details/74542166))

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf) (Christian Ledig, CVPR 2017, [code](https://github.com/tensorlayer/srgan), [note](https://www.cnblogs.com/wangxiaocvpr/p/5989802.html))

[Pixel Recursive Super Resolution](https://arxiv.org/pdf/1702.00783.pdf) (Ryan Dahl, 2017, [code](https://github.com/nilboy/pixel-recursive-super-resolution))

[Deep Back-Projection Networks for Super-Resolution](http://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf) (Muhammad Haris, CVPR 2018, [code](https://github.com/alterzero/DBPN-Pytorch), [note](https://zhuanlan.zhihu.com/p/34400207))

#### Person Re-identification

[Joint Discriminative and Generative Learning for Person Re-identification](https://arxiv.org/pdf/1904.07223.pdf) (Zhedong Zheng, CVPR 2019, [code](https://github.com/NVlabs/DG-Net?utm_source=catalyzex.com))

### Machine Learning

#### Decision Tree

[Classification and Regression Trees](http://xxpt.ynjgy.com/resource/data/110102/U/705/pdfs/L3ClassTrees.pdf) (L. Breiman, 1984, [code1](https://github.com/bensadeghi/DecisionTree.jl), [code2](https://github.com/wreardan/cart), [note](https://www.cnblogs.com/wxquare/p/5379970.html))

[Induction of Decision Trees](http://hunch.net/~coms-4771/quinlan.pdf) (J. Ross Quinlan, 1986, [code](https://github.com/igrigorik/decisiontree), [note](https://www.cnblogs.com/wxquare/p/5379970.html))

[C4.5: Programs for Machine Learning](https://dl.acm.org/citation.cfm?id=152181) (J. Ross Quinlan, 1993, [code](https://github.com/yandongliu/learningjs), [note](https://www.cnblogs.com/wxquare/p/5379970.html))

#### Support Vector Machine

[A Training Algorithm for Optimal Margin Classifiers](http://www.svms.org/training/BOGV92.pdf) (Bernhard E Boser, 1992, [code](https://github.com/cjlin1/libsvm))

[Support-Vector Networks](http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf) (Corinna Cortes, 1995, [code](https://github.com/cjlin1/libsvm))

[Estimating the Support of a High-Dimensional Distribution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.9423&rep=rep1&type=pdf) (Bernhard Schölkopf, 1999, [code](https://github.com/cjlin1/libsvm))

[New Support Vector Algorithms](https://www.researchgate.net/publication/12413257_New_Support_Vector_Algorithms) (Bernhard Schölkopf, 2000, [code](https://github.com/cjlin1/libsvm))

#### Conditional Random Field

[An Introduction to Conditional Random Fields](https://arxiv.org/pdf/1011.4088.pdf) (Charles Sutton, 2010, [code](https://github.com/timvieira/crf), [note](https://zhuanlan.zhihu.com/p/70067113))

#### Expectation Maximization

[Maximum likelihood from incomplete data via the EM algorithm](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x) (Arthur Dempster, 1977, [note](https://zhuanlan.zhihu.com/p/40991784))

[The EM Algorithm and Extensions](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470191613) (Geoff McLachlan, 1997)

#### Ensemble Method

[Greedy function approximation: a gradient boosting machine](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451) (Jerome H. Friedman, 2001, [code](https://github.com/dmlc/xgboost), [note](https://www.jianshu.com/p/005a4e6ac775))

[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf#page=10&zoom=100,0,198) (Tianqi Chen, SIGKDD 2016, [code](https://github.com/dmlc/xgboost), [note](http://djjowfy.com/2017/08/01/XGBoost%E7%9A%84%E5%8E%9F%E7%90%86/))

#### Learning Theory

[On the Uniform Convergence of Relative Frequencies of Events to their Probabilities](https://link.springer.com/chapter/10.1007/978-3-319-21852-6_3) (Vladimir Vapnik, 1971)

[A Theory of the Learnable](https://dl.acm.org/doi/pdf/10.1145/1968.1972?casa_token=jkM2-kt89ncAAAAA:Eavindse5LyhN1lfVXUQHJej6wwPne4TV_AASPF6bkwDveiydsl6fpGD4prUdXEpYCoeF7kN4RfaElE) (Leslie Valiant, 1984)

[Occam's Razor](https://users.soe.ucsc.edu/~manfred/pubs/J9.pdf) (Anselm Blumer, 1987, [note](https://zh.wikipedia.org/zh-hans/%E5%A5%A5%E5%8D%A1%E5%A7%86%E5%89%83%E5%88%80))

#### Imbalanced Data

[SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/pdf/1106.1813.pdf) (Nitesh V. Chawla, 2002, [code](https://github.com/scikit-learn-contrib/imbalanced-learn), [note](https://blog.csdn.net/shine19930820/article/details/54143241))

[kNN approach to unbalanced data distributions: A case study involving information extraction](http://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf?attredirects=0) (Jianping Zhang, 2003, [code](http://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf?attredirects=0))

[Balancing Training Data for Automated Annotation of Keywords: a Case Study](https://pdfs.semanticscholar.org/c1a9/5197e15fa99f55cd0cb2ee14d2f02699a919.pdf) (Gustavo E. A. P. A. Batista, 2003, [code](https://github.com/scikit-learn-contrib/imbalanced-learn))

[A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.7757&rep=rep1&type=pdf) (Ronaldo C. Prati, 2004, [code](https://github.com/scikit-learn-contrib/imbalanced-learn))

[Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf) (Hui Han, 2005, [note](https://blog.csdn.net/shine19930820/article/details/54143241))

[ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf) (Haibo He, 2008, [code](https://github.com/scikit-learn-contrib/imbalanced-learn), [note](https://blog.csdn.net/weixin_40118768/article/details/80226423))

[Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling Technique for Handling the Class Imbalanced Problem](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2009-Bunkhumpornpat-LNCS.pdf) (Chumphol Bunkhumpornpat, 2009)

[Learning from Imbalanced Data](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5128907) (Haibo He, 2009, [note](https://blog.csdn.net/shine19930820/article/details/54143241))

#### Multi-Task Learning

[Multiple Kernel Learning, Conic Duality, and the SMO Algorithm](https://www.di.ens.fr/~fbach/skm_icml.pdf) (Francis R. Bach, ICML 2004)

[Large Scale Multiple Kernel Learning](http://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf) (Sören Sonnenburg, JMLR 2006)

[Factorized Latent Spaces with Structured Sparsity](https://papers.nips.cc/paper/3953-factorized-latent-spaces-with-structured-sparsity.pdf) (Yangqing Jia, NeuIPS 2010)

[Factorized Orthogonal Latent Spaces](https://ttic.uchicago.edu/~rurtasun/publications/SalzmannEkUrtasunDarrell10.pdf) (Mathieu Salzmann, 2010)

[Domain Separation Networks](https://papers.nips.cc/paper/6254-domain-separation-networks.pdf) (Konstantinos Bousmalis, NeuIPS 2016, [note](https://zhuanlan.zhihu.com/p/49479734))

[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf) (Xiaodong Liu, ACL 2019, [code](https://github.com/namisan/mt-dnn), [note](https://zhuanlan.zhihu.com/p/60282783))

### Deep Learning

#### Artificial Neural Network

[A logical Calculus of Ideas Immanent in Nervous Activity](https://link.springer.com/article/10.1007%252FBF02478259) (Warren McCulloch, 1943)

[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf) (Frank Rosenblatt, 1958)

[Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms](https://babel.hathitrust.org/cgi/pt?id=mdp.39015039846566&view=1up&seq=8) (Frank Rosenblatt, 1961)

[Phoneme Recognition Using Time-Delay Neural Networks](https://isl.anthropomatik.kit.edu/downloads/PhonemeRecognitionUsingTimeDelayNeuralNetworks_NEU(1).pdf) (Alexander Waibel, 1989)

#### Convolutional Neural Network

[Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/) (David Hunter Hubel, 1962)

[Backpropagation Applied to Handwritten Zip Code Recognition](https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf) (Yann LeCun, 1989, [note](https://blog.csdn.net/u012679707/article/details/80738633))

[Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (Yann LeCun, 1998, [note](https://blog.csdn.net/sunshine_010/article/details/79876255))

[Notes on Convolutional Neural Networks](http://202.116.81.74/cache/7/03/cogprints.org/036a03bc6027afc65c14907d0a1fae73/cnn_tutorial.pdf) (Jake Bouvrie, 2006, [note](https://blog.csdn.net/langb2014/article/details/48470181))

[ImageNet Classification with Deep Convolutional Neural Networks](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) (Alex Krizhevsky, NeuIPS 2012, [note](https://zhuanlan.zhihu.com/p/20324656))

[Simplifying ConvNets for Fast Learning](https://liris.cnrs.fr/Documents/Liris-5659.pdf) (Franck Mamalet, 2012)

[Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) (Matthew D. Zeiler, ECCV 2014, [note](https://www.zybuluo.com/lutingting/note/459569))

[Rigid-Motion Scattering for Texture Classification](https://arxiv.org/pdf/1403.1687.pdf) (Laurent Sifre, 2014)

[Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) (Christian Szegedy, CVPR 2015, [note](https://blog.csdn.net/lhanchao/article/details/55804968))

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) (Karen Simonyan, ICLR 2015, [note](https://zhuanlan.zhihu.com/p/32853559))

[Highway Networks](http://de.arxiv.org/pdf/1505.00387) (Rupesh Kumar Srivastava, 2015, [note](https://zhuanlan.zhihu.com/p/38130339))

[Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122.pdf) (Fisher Yu, ICLR 2016, [code](https://github.com/iesl/dilated-cnn-ner), [note](https://www.cnblogs.com/fourmi/p/10049998.html))

[Deep Residual Learning for Image Recognition](https://x-algo.cn/wp-content/uploads/2016/12/residual.pdf) (Kaiming He, CVPR 2016, [note](https://zhuanlan.zhihu.com/p/47199669))

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf) (Christian Szegedy, CVPR 2016, [code](https://github.com/pytorch/vision?utm_source=catalyzex.com), [note](https://zhuanlan.zhihu.com/p/50751422))

[Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf) (Gao Huang, ECCV 2016)

[Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/pdf/1603.08029.pdf) (Sasha Targ, ICLR 2016 Workshop)

[Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf) (Sergey Zagoruyko, BMVC 2016)

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf) (Christian Szegedy, AAAI 2017)

[Densely Connected Convolutional Networks](http://www.cs.cmu.edu/~jeanoh/16-785/papers/huang-cvpr2017-densenet.pdf) (Gao Huang, CVPR 2017, [code](https://github.com/liuzhuang13/DenseNet))

[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) (Saining Xie, CVPR 2017)

[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) (Francois Chollet, CVPR 2017)

[Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ioannou_Deep_Roots_Improving_CVPR_2017_paper.pdf) (Yani Ioannou, CVPR 2017)

[Factorized Convolutional Neural Networks](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w10/Wang_Factorized_Convolutional_Neural_ICCV_2017_paper.pdf) (Min Wang, ICCV 2017)

[Deformable Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf) (Jifeng Dai, ICCV 2017)

[Convolution with Logarithmic Filter Groups for Efficient Shallow CNN](https://arxiv.org/pdf/1707.09855.pdf) (Tae Kwan Lee, 2017)

[Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) (Jie Hu, CVPR 2018)

[Tree-CNN: A Deep Convolutional Neural Network for Lifelong Learning](https://arxiv.org/pdf/1802.05800.pdf) (Deboleena Roy, 2018, [code](https://github.com/magical2world/tensorflow-Tree-CNN), [note](https://blog.csdn.net/qq_24305433/article/details/79856672))

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) (Mingxing Tan, ICML 2019, [code](https://github.com/qubvel/efficientnet), [note](https://zhuanlan.zhihu.com/p/70369784))

[HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions](https://arxiv.org/pdf/1908.03888.pdf) (Duo Li, ICCV 2019, [code](https://github.com/d-li14/HBONet))

[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf) (Mingxing Tan, 2021)

#### Recurrent Neural Network

[Long short-term memory](http://xueshu.baidu.com/s?wd=paperuri%3A%28051bcc198724a1da0b831afe39380852%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D1) (Sepp Hochreiter, 1997)

[A Critical Review of Recurrent Neural Networks for Sequence Learning](http://pdfs.semanticscholar.org/0651/b333c2669227b0cc42de403268a4546ece70.pdf) (Zachary C. Lipton, 2015, [note](https://blog.csdn.net/xizero00/article/details/51225065))

[Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983.pdf) (Alex Graves, 2016, [note](https://blog.csdn.net/liuyuemaicha/article/details/53999091))

[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf) (Stephen Merity, 2017, [note](https://ldzhangyx.github.io/2019/07/31/awd-lstm/))

[Simple Recurrent Units for Highly Parallelizable Recurrence](https://arxiv.org/pdf/1709.02755.pdf) (Tao Lei, EMNLP 2018, [code](https://github.com/taolei87/sru))

[Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/pdf/1708.06834.pdf) (Victor Campos, ICLR 2018, [note](https://www.jianshu.com/p/5c4dd629b1ec))

#### Generative Adversarial Networks

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) (Ian Goodfellow, 2014)

[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf) (Mehdi Mirza, 2014, [note](https://zhuanlan.zhihu.com/p/23648795))

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) (Alec Radford, ICLR 2016, [note](https://www.cnblogs.com/wangxiaocvpr/p/5965434.html))

[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf) (Ian Goodfellow, NeuIPS 2016, [note](https://blog.csdn.net/cskywit/article/details/86612142))

[Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/pdf/1610.09585.pdf) (Augustus Odena, 2016, [note](https://blog.csdn.net/qq_24477135/article/details/85758496))

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) (Tim Salimans, 2016, [note](https://blog.csdn.net/zijin0802034/article/details/58643889))

[Generative Adversarial Networks: An Overview](https://arxiv.org/pdf/1710.07035.pdf) (Antonia Creswell, 2017)

[How Generative Adversarial Nets and its variants Work: An Overview of GAN](https://arxiv.org/pdf/1711.05914.pdf) (Yongjun Hong, 2017)

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) (Phillip Isola, CVPR 2017, [code](https://github.com/phillipi/pix2pix), [note](https://blog.csdn.net/Teeyohuang/article/details/82699781))

[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242.pdf) (Han Zhang, ICCV 2017, [code](https://github.com/hanzhanggit/StackGAN), [note](https://blog.csdn.net/a312863063/article/details/83574422))

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) (Jun-Yan Zhu, ICCV 2017, [code](https://github.com/junyanz/CycleGAN), [note](https://blog.csdn.net/hhy_csdn/article/details/82913776))

[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)  (Ting-Chun Wang, CVPR 2018, [code](https://github.com/NVIDIA/pix2pixHD), [note](https://zhuanlan.zhihu.com/p/35955531))

[Learning from Simulated and Unsupervised Images through Adversarial Training](https://www.arxiv-vanity.com/papers/1612.07828/) (Ashish Shrivastava, ICCV 2017, [code](https://github.com/carpedm20/simulated-unsupervised-tensorflow), [note](https://blog.csdn.net/daydayjump/article/details/81977479))

[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) (Martin Arjovsky, 2017, [code](https://github.com/martinarjovsky/WassersteinGAN), [note](https://zhuanlan.zhihu.com/p/25071913))

[Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf) (Han Zhang, 2018, [code](https://github.com/heykeetae/Self-Attention-GAN), [note](https://www.paperweekly.site/papers/notes/414))

[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf) (Tero Karras, ICLR 2018, [code](https://github.com/tkarras/progressive_growing_of_gans), [note](https://zhuanlan.zhihu.com/p/30637133))

[Transferring GANs: Generating Images from Limited Data](https://arxiv.org/pdf/1805.01677.pdf) (Yaxing Wang, ECCV 2018, [code](https://github.com/yaxingwang/Transferring-GANs), [note](https://medium.com/@xiaosean5408/transferring-gans%E7%B0%A1%E4%BB%8B-transferring-gans-generating-images-from-limited-data-90bcf6be7fd2))

[Self-Supervised GANs via Auxiliary Rotation Loss](https://arxiv.org/pdf/1811.11212.pdf) (Ting Chen, CVPR 2019, [code](https://github.com/vandit15/Self-Supervised-Gans-Pytorch), [note](https://blog.csdn.net/weixin_44363205/article/details/104918734))

[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf) (Tero Karras, CVPR 2019, [code](https://github.com/NVlabs/stylegan), [note](https://blog.csdn.net/weixin_42360095/article/details/89522153))

[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/pdf/1809.11096.pdf) (Andrew Brock, ICLR 2019, [note](https://zhuanlan.zhihu.com/p/46581611))

[Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf) (Tero Karras, 2019, [note](https://blog.csdn.net/lynlindasy/article/details/104495583))

#### Autoencoder

[Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) (Pascal Vincent, ICML 2008)

##### Variational Autoencoder

[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) (Diederik P. Kingma, ICLR 2014)

[Variational Inference with Normalizing Flows](http://proceedings.mlr.press/v37/rezende15.pdf) (Danilo Jimenez Rezende, ICML 2015, [note](https://bingning.wang/research/Article/?id=100))

[Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf) (Kihyuk Sohn, NeuIPS 2015, [note](https://zhuanlan.zhihu.com/p/25518643))

[Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/pdf/1606.04934.pdf) (Diederik P. Kingma, NeuIPS 2016)

[Variational Graph AutoEncoders](https://arxiv.org/pdf/1611.07308.pdf) (Thomas N. Kipf, NeuIPS 2016 Workshop)

[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl) (Irina Higgins, ICLR 2017)

[Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410) (Alexander A. Alemi, ICLR 2017)

[Variational Lossy Autoencoder](https://arxiv.org/pdf/1611.02731.pdf) (Xi Chen, ICLR 2017)

[Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf) (Aaron van den Oord, NeuIPS 2017)

[Adversarially Regularized Autoencoders](http://proceedings.mlr.press/v80/zhao18b/zhao18b.pdf) (Junbo Zhao, ICML 2018)

[Disentangling by Factorising](http://proceedings.mlr.press/v80/kim18b/kim18b.pdf) (Hyunjik Kim, ICML 2018)

[Adversarially Regularized Graph Autoencoder for Graph Embedding](https://www.ijcai.org/proceedings/2018/0362.pdf) (Shirui Pan, IJCAI 2018)

[Isolating Sources of Disentanglement in VAEs](https://arxiv.org/pdf/1802.04942.pdf) (Ricky T. Q. Chen, NeuIPS 2018)

[Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/pdf/1804.00104.pdf) (Emilien Dupont, NeuIPS 2018)

[GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders](https://arxiv.org/pdf/1802.03480.pdf) (Martin Simonovsky, 2018)

[Structured Disentangled Representations](http://proceedings.mlr.press/v89/esmaeili19a/esmaeili19a.pdf) (Babak Esmaeili, AISTATS 2019)

[Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf) (Ali Razavi, NeuIPS 2019)

[Variational Autoencoders and Nonlinear ICA: A Unifying Framework](http://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf) (Ilyes Khemakhem, AISTATS 2020)

[From Variational to Deterministic Autoencoders](https://arxiv.org/abs/1903.12436) (Partha Ghosh, ICLR 2020)

#### Graph Neural Network

[The Graph Neural Network Model](http://persagen.com/files/misc/scarselli2009graph.pdf) (Franco Scarselli, 2009, [note](https://zhuanlan.zhihu.com/p/76290138))

[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) (Thomas N. Kipf, ICLR 2017, [code](https://github.com/tkipf/gcn), [note](https://zhuanlan.zhihu.com/p/35630785))

[Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) (Petar Veličković, ICLR 2018, [code](https://github.com/PetarV-/GAT), [note](https://zhuanlan.zhihu.com/p/34232818))

[A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) (Zonghan Wu, TNNLS 2019)

[Graph Neural Networks for Natural Language Processing: A Survey](https://arxiv.org/pdf/2106.06090.pdf) (Lingfei Wu, 2021)

#### Capsule Network

[Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf) (Geoffrey Hinton, NeuIPS 2017, [code](https://github.com/naturomics/CapsNet-Tensorflow), [note](https://zhuanlan.zhihu.com/p/32156167))

[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) (Geoffrey Hinton, ICLR 2018)

#### Attention Mechanism

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (Dzmitry Bahdanau, ICLR 2015, [note](https://blog.csdn.net/WUTab/article/details/73657905))

[DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/pdf/1709.04696.pdf) (Tao Shen, 2017, [code](https://github.com/taoshen58/DiSAN), [note](https://zhuanlan.zhihu.com/p/36349043))

[Learning What’s Easy: Fully Differentiable Neural Easy-First Taggers](https://www.aclweb.org/anthology/D17-1036.pdf) (Andre F. T. Martins, EMNLP 2017)

[Structured Attention Networks](https://arxiv.org/pdf/1702.00887.pdf) (Yoon Kim, ICLR 2017)

[You May Not Need Attention](https://arxiv.org/pdf/1810.13409.pdf) (Ofir Press, 2018, [code](https://github.com/ofirpress/YouMayNotNeedAttention), [note](https://zhuanlan.zhihu.com/p/48374997))

[An Introductory Survey on Attention Mechanisms in NLP Problems](https://arxiv.org/pdf/1811.05544.pdf) (Dichao Hu, 2018, [note](https://blog.csdn.net/cskywit/article/details/84753293))

##### Memory Network

[Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf) (Alex Graves, 2014, [code](https://github.com/carpedm20/NTM-tensorflow), [note](https://zhuanlan.zhihu.com/p/30383994))

[Memory Networks](https://arxiv.org/pdf/1410.3916v11.pdf) (Jason Weston, ICLR 2015, [note](https://zhuanlan.zhihu.com/p/32257642))

[End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf) (Sainbayar Sukhbaatar, 2015, [note](https://zhuanlan.zhihu.com/p/32257642))

[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://www.thespermwhale.com/jaseweston/ram/papers/paper_21.pdf) (Ankit Kumar, 2015, [note](https://zhuanlan.zhihu.com/p/32257642))

[Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/pdf/1603.01417.pdf) (Caiming Xiong, 2016, [code](https://github.com/ethancaballero/Improved-Dynamic-Memory-Networks-DMN-plus), [note](https://zhuanlan.zhihu.com/p/32257642))

[Gated End-to-End Memory Networks](http://www.aclweb.org/anthology/E/E17/E17-1001.pdf) (Fei Liu, 2016, [note](https://zhuanlan.zhihu.com/p/30722242))

##### Transformer

[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) (Ashish Vaswani, 2017, [code1](https://github.com/jadore801120/attention-is-all-you-need-pytorch), [code2](https://github.com/Kyubyong/transformer), [code3](https://github.com/bojone/attention), [note](https://zhuanlan.zhihu.com/p/48508221))

[Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) (Peter Shaw, 2018, [note](https://www.jianshu.com/p/cb5b2d967e90))

[Input Combination Strategies for Multi-Source Transformer Decoder](https://arxiv.org/pdf/1811.04716.pdf) (Jindrich Libovicky, WMT 2018)

[Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/pdf/1905.09418.pdf) (Elena Voita, ACL 2019)

[Universal Transformer](https://arxiv.org/pdf/1807.03819.pdf) (Mostafa Dehghani, ICLR 2019, [code](https://github.com/andreamad8/Universal-Transformer-Pytorch), [note](https://zhuanlan.zhihu.com/p/44655133))

[Adaptive Attention Span in Transformers](https://arxiv.org/pdf/1905.07799.pdf) (Sainbayar Sukhbaatar, ACL 2019)

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf) (Zihang Dai, ACL 2019, [note](https://zhuanlan.zhihu.com/p/70745925))

[Self-Attention with Structural Position Representations](https://arxiv.org/pdf/1909.00383.pdf) (Xing Wang, EMNLP 2019)

[Tree Transformer: Integrating Tree Structures into Self-Attention](https://arxiv.org/pdf/1909.06639.pdf) (Yau-Shian Wang, EMNLP 2019)

[Star-Transformer](https://arxiv.org/pdf/1902.09113.pdf) (Qipeng Guo, 2019, [note](https://zhuanlan.zhihu.com/p/97888995))

[Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf) (Nikita Kitaev, ICLR 2020)

[How Does Selective Mechanism Improve Self-Attention Networks?](https://arxiv.org/pdf/2005.00979.pdf) (Xinwei Geng, ACL 2020)

[The Unstoppable Rise of Computational Linguistics in Deep Learning](https://arxiv.org/pdf/2005.06420.pdf) (James Henderson, ACL 2020)

[ETC: Encoding Long and Structured Data in Transformers](https://www.aclweb.org/anthology/2020.emnlp-main.19.pdf) (Joshua Ainslie, EMNLP 2020)

[On the Sub-layer Functionalities of Transformer Decoder](https://arxiv.org/pdf/2010.02648.pdf) (Yilin Yang, EMNLP 2020 Findings)

[Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) (Iz Beltagy, 2020, [note](https://zhuanlan.zhihu.com/p/134748587))

[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768.pdf) (Sinong Wang, 2020)

[Multi-Head Attention: Collaborate Instead of Concatenate](https://openreview.net/forum?id=bK-rJMKrOsm) (Jean-Baptiste Cordonnier, 2020, [code](https://github.com/epfml/collaborative-attention), [note](https://medium.com/im%E6%97%A5%E8%A8%98/%E8%AB%96%E6%96%87%E5%88%86%E4%BA%AB-multi-head-attention-collaborate-instead-of-concatenate-196dccff6118))

[Synthesizer: Rethinking Self-Attention for Transformer Models](http://proceedings.mlr.press/v139/tay21a/tay21a.pdf) (Yi Tay, ICML 2021, [note](https://zhuanlan.zhihu.com/p/148054019))

##### Sparse Attention

[From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](http://proceedings.mlr.press/v48/martins16.pdf) (Andre F. T. Martins, ICML 2016, [code](https://github.com/KrisKorrel/sparsemax-pytorch), [note](https://www.cs.utah.edu/~tli/posts/2019/01/blog-post-1/))

[A Regularized Framework for Sparse and Structured Neural Attention](https://arxiv.org/pdf/1705.07704.pdf) (Vlad Niculae, NeuIPS 2017)

[Sparse and Constrained Attention for Neural Machine Translation](https://arxiv.org/pdf/1805.08241.pdf) (Chaitanya Malaviya, ACL 2018)

[Sparse Sequence-to-Sequence Models](https://arxiv.org/pdf/1905.05702.pdf) (Ben Peters, ACL 2019, [note](https://zhuanlan.zhihu.com/p/76607614))

[Adaptively Sparse Transformers](https://arxiv.org/pdf/1909.00015.pdf) (Goncalo M. Correia, EMNLP 2019)

[Efficient Content-Based Sparse Attention with Routing Transformers](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00353/97776/Efficient-Content-Based-Sparse-Attention-with) (Aurko Roy, TACL 2021)

#### Optimization

[Learning representations by back-propagating errors](http://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf) (David E. Rumelhart, 1986)

[On the Momentum Term in Gradient Descent Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf) (Ning Qian, 1999)

[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (John Duchi, JMLR 2011)

[Sequential Model-Based Optimization for General Algorithm Configuration](https://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf) (Frank Hutter, LION 2011)

[ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701.pdf) (Matthew D. Zeiler, 2012, [note](https://zh.d2l.ai/chapter_optimization/adadelta.html))

[ADAM: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf) (Diederik P. Kingma, 2015, [note](https://www.jianshu.com/p/aebcaf8af76e))

[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf) (Sebastian Ruder, 2017, [note](https://zhuanlan.zhihu.com/p/21539419))

#### Weight Initialization

[Understanding the difficulty of training deep feedforward neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf) (Xavier Glorot, JMLR 2010, [note](https://zhuanlan.zhihu.com/p/43840797))

[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) (Kaiming He, ICCV 2015, [note](https://www.cnblogs.com/everyday-haoguo/p/Note-PRelu.html))

#### Loss Function

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) (Florian Schroff, CVPR 2015, [code](https://github.com/davidsandberg/facenet), [note](https://blog.csdn.net/chenriwei2/article/details/45031677))

[A Discriminative Feature Learning Approach for Deep Face Recognition](http://www.eccv2016.org/files/posters/P-3B-20.pdf) (Yandong Wen, ECCV 2016, [code](https://github.com/pangyupo/mxnet_center_loss), [note](https://blog.csdn.net/oJiMoDeYe12345/article/details/78548663))

[An exploration of softmax alternatives belonging to the spherical loss family](https://arxiv.org/pdf/1511.05042.pdf) (Alexandre de Brebisson, ICLR 2016)

[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) (Weiyang Liu, ICML 2016, [code](https://github.com/wy1iu/LargeMargin_Softmax_Loss), [note](https://zhuanlan.zhihu.com/p/45448909))

[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) (Tsung-Yi Lin, ICCV 2017, [code](https://github.com/unsky/focal-loss), [note](https://zhuanlan.zhihu.com/p/49981234))

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf) (Weiyang Liu, ICML 2017, [code](https://github.com/wy1iu/sphereface), [note](https://zhuanlan.zhihu.com/p/45448909))

[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf) (Hao Wang, 2018, [code](https://github.com/yule-li/CosFace), [note](https://zhuanlan.zhihu.com/p/59736735))

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf) (Jiankang Deng, 2018, [code](https://github.com/deepinsight/insightface), [note](https://zhuanlan.zhihu.com/p/76541084))

[Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) (Feng Wang, 2018, [code](https://github.com/Joker316701882/Additive-Margin-Softmax), [note](https://blog.csdn.net/shaoxiaohu1/article/details/79139039))

[On Controllable Sparse Alternatives to Softmax](http://papers.nips.cc/paper/7878-on-controllable-sparse-alternatives-to-softmax.pdf) (Anirban Laha, NIPS 2018)

[DropMax: Adaptive Variational Softmax](http://papers.nips.cc/paper/7371-dropmax-adaptive-variational-softmax.pdf) (Hae Beom Lee, NIPS 2018)

#### Activation Function

[Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) (Vinod Nair, ICML 2010)

[Empirical Evaluation of Rectified Activations in Convolution Network](https://arxiv.org/pdf/1505.00853.pdf) (Bing Xu, 2015)

[Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](https://arxiv.org/pdf/1603.05201v2.pdf) (Wenling Shang, ICML 2016, [note](https://blog.csdn.net/cv_family_z/article/details/52399921))

[Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415.pdf) (Dan Hendrycks, 2016, [code](https://github.com/hendrycks/GELUs), [note](https://zhuanlan.zhihu.com/p/100175788))

[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144) (Eric Jang, ICLR 2017, [note](https://zhuanlan.zhihu.com/p/50065712))

#### Normalization

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://de.arxiv.org/pdf/1502.03167) (Sergey Ioffe, ICML 2015, [note](https://zhuanlan.zhihu.com/p/50444499))

[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) (Jimmy Lei Ba, 2016, [note](https://zhuanlan.zhihu.com/p/54530247))

[Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf) (Tim Salimans, 2016)

[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf) (Dmitry Ulyanov, 2017)

[Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) (Yuxin Wu, 2018, [note](https://zhuanlan.zhihu.com/p/35005794))

#### Reinforcement Learning

[A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf) (Kai Arulkumaran, 2017, [note](https://blog.csdn.net/KyrieHe/article/details/79504481))

[Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf) (Yuxi Li, 2017, [note](https://zhuanlan.zhihu.com/p/31595581))

##### Reinforcement Learning Application

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) (Volodymyr Mnih, 2013, [note](https://zhuanlan.zhihu.com/p/33962867))

[Mastering the Game of Go with Deep Neural Networks and Tree Search](http://www.worlduc.com/FileSystem/1/134755/1585588/ac5b78a1934c49bb93a1f3ad09d96e46.pdf) (David Silver, Nature 2016)

[Mastering the Game of Go without Human Knowledge](https://skatgame.net/mburo/ps/alphago-zero.pdf) (David Silver, Nature 2017, [code](https://github.com/gcp/leela-zero), [note](https://zhuanlan.zhihu.com/p/101669366))

[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf) (David Silver, 2017)

[DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/pdf/2106.06135.pdf) (Daochen Zha, ICML 2021)

#### Contrastive Learning

[Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) (Raia Hadsell, CVPR 2006)

[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf) (Kihyuk Sohn, NeuIPS 2016)

[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) (Aaron van den Oord, 2018)

[Learning Deep Representations by Mutual Information Estimation and Maximization](https://arxiv.org/pdf/1808.06670.pdf) (R Devon Hjelm, ICLR 2019)

[Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/pdf/1906.00910.pdf) (Philip Bachman, 2019, [code](https://github.com/Philip-Bachman/amdim-public))

[Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf) (Yonglong Tian, 2019)

[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf) (Kaiming He, CVPR 2020)

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) (Ting Chen, ICML 2020)

[Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/pdf/2006.09882.pdf) (Mathilde Caron, NeuIPS 2020)

[Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/pdf/2003.04297.pdf) (Xinlei Chen, 2020)

[Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf) (Xinlei Chen, 2020, [note](https://zhuanlan.zhihu.com/p/331678807))

[Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf) (Jean-Bastien Gril, 2020, [code](https://github.com/lucidrains/byol-pytorch))

[An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf) (Xinlei Chen, 2021)

#### Incremental/Continual/Lifelong Learning

[iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) (Sylvestre-Alvise Rebuffi, CVPR 2017)

[Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840) (David Lopez-Paz, NeuIPS 2017)

[Continual Learning with Deep Generative Replay](https://arxiv.org/pdf/1705.08690.pdf) (Hanul Shin, NeuIPS 2017)

[End-to-End Incremental Learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf) (Francisco M. Castro, ECCV 2018)

[Lifelong Machine Learning](https://www.cs.uic.edu/~liub/lifelong-machine-learning-draft.pdf) (Zhiyuan Chen, 2018)

[Large Scale Incremental Learning](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html) (Yue Wu, CVPR 2019)

[Learning a Unified Classifier Incrementally via Rebalancing](http://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) (Saihui Hou, CVPR 2019)

[Continual Lifelong Learning with Neural Networks: A Review](https://www.sciencedirect.com/science/article/pii/S0893608019300231) (German I. Parisi, 2019)

[Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Mnemonics_Training_Multi-Class_Incremental_Learning_Without_Forgetting_CVPR_2020_paper.pdf) (Yaoyao Liu, CVPR 2020)

#### Zero-Shot Learning

[Zero-Shot Learning with Semantic Output Codes](http://www.cs.cmu.edu/afs/cs/project/theo-73/www/papers/zero-shot-learning.pdf) (Mark Palatucci, NeuIPS 2009, [note](https://zhuanlan.zhihu.com/p/34076480))

[Label-Embedding for Attribute-Based Classification](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=94AE59377DBC233D6EE1808F16B505CB?doi=10.1.1.371.9746&rep=rep1&type=pdf) (Zeynep Akata, CVPR 2013, [note](https://blog.csdn.net/hanss2/article/details/80537356))

[Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths](https://arxiv.org/pdf/1703.05002.pdf) (Yanan Li, CVPR 2017, [note](https://zhuanlan.zhihu.com/p/29392845))

#### Few-Shot Learning

[Learning from one example through shared densities on transform](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9021&rep=rep1&type=pdf) (Erik G Miller, CVPR 2000)

[One-Shot Learning of Object Categories](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf) (Li Fei-Fei, PAMI 2006, [note](https://blog.csdn.net/sinat_36594453/article/details/89817314))

[Siamese neural networks for one-shot image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) (Gregory Koch, ICML 2015 Workshop, [note](https://zhuanlan.zhihu.com/p/86283037))

[Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) (Oriol Vinyals, NeuIPS 2016, [note](https://zhuanlan.zhihu.com/p/32101204))

#### Meta Learning

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf) (Chelsea Finn, ICML 2017, [code](https://github.com/cbfinn/maml?utm_source=catalyzex.com), [note](https://zhuanlan.zhihu.com/p/57864886))

[Meta-Learning: A Survey](https://arxiv.org/pdf/1810.03548.pdf) (Joaquin Vanschoren, 2018)

[Meta-Learning Representations for Continual Learning](https://arxiv.org/pdf/1905.12588.pdf) (Khurram Javed, NIPS 2019, [code](https://github.com/khurramjaved96/mrcl?utm_source=catalyzex.com))

[Learning to Continually Learn](https://arxiv.org/pdf/2002.09571.pdf) (Shawn Beaulieu, ECAI 2020, [code](https://github.com/uvm-neurobotics-lab/ANML?utm_source=catalyzex.com))

#### Curriculum Learning

[Curriculum Learning](https://mila.quebec/wp-content/uploads/2019/08/2009_curriculum_icml.pdf) (Yoshua Bengio, ICML 2009, [note](https://zhuanlan.zhihu.com/p/114825029))

[Self-Paced Curriculum Learning](https://ojs.aaai.org/index.php/AAAI/article/view/9608) (Lu Jiang, AAAI 2015)

[Automated Curriculum Learning for Neural Networks](http://proceedings.mlr.press/v70/graves17a/graves17a.pdf) (Alex Graves, ICML 2017)

#### Model Compression and Acceleration

[A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf) (Yu Cheng, 2017)

##### Parameter Pruning and Quantization

[Rethinking the Value of Network Pruning](https://arxiv.org/pdf/1810.05270.pdf) (Zhuang Liu, ICLR 2019, [code](https://github.com/Eric-mingjie/rethinking-network-pruning), [note](https://xmfbit.github.io/2018/10/22/paper-rethinking-the-value-of-network-pruning/))

##### Human-Designed Model

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) (Andrew Howard, 2017)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf) (Mark Sandler, CVPR 2018, [code](https://github.com/pytorch/vision?utm_source=catalyzex.com))

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf) (Xiangyu Zhang, CVPR 2018)

[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf) (Ningning Ma, ECCV 2018)

[GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf) (Kai Han, CVPR 2020)

##### Knowledge Distillation

[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf) (Geoffrey Hinton, NeuIPS 2015, [note](https://xmfbit.github.io/2018/06/07/knowledge-distilling/))

[Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf) (Ilija Radosavovic, CVPR 2020)

[Knowledge Distillation: A Survey](https://link.springer.com/content/pdf/10.1007/s11263-021-01453-z.pdf) (Jianping Gou, IJCV 2021)

#### Neural Architecture Search

[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578.pdf) (Barret Zoph, ICLR 2017, [note](https://zhuanlan.zhihu.com/p/47221948))

[Learning Transferable Architectures for Scalable Image Recognition](http://openaccess.thecvf.com/content_cvpr_2018/html/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.html) (Barret Zoph, CVPR 2018, [note](https://zhuanlan.zhihu.com/p/31655995))

[Progressive Neural Architecture Search](http://openaccess.thecvf.com/content_ECCV_2018/html/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.html) (Chenxi Liu, ECCV 2018, [note](https://blog.csdn.net/dhaiuda/article/details/102599427))

[NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](http://openaccess.thecvf.com/content_ECCV_2018/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.html) (Tien-Ju Yang, ECCV 2018, [code](https://github.com/denru01/netadapt), [note](https://blog.csdn.net/thisiszdy/article/details/90515075))

[Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf) (Hieu Pham, ICML 2018, [note](https://cloud.tencent.com/developer/article/1182704))

[Regularized Evolution for Image Classifier Architecture Search](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4405) (Esteban Real, AAAI 2019, [note](https://blog.csdn.net/dhaiuda/article/details/93337707))

[DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf) (Hanxiao Liu, ICLR 2019, [code](https://github.com/quark0/darts), [note](https://www.cnblogs.com/wangxiaocvpr/p/10556789.html))

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (Han Cai, ICLR 2019, [code](https://github.com/mit-han-lab/proxylessnas), [note](https://zhuanlan.zhihu.com/p/55220311))

[MnasNet: Platform-Aware Neural Architecture Search for Mobile](http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html) (Mingxing Tan, CVPR 2019, [code](https://github.com/mingxingtan/mnasnet), [note](https://zhuanlan.zhihu.com/p/42474017))

[Searching for MobileNetV3](http://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) (Andrew Howard, ICCV 2019, [note](https://blog.nex3z.com/2020/07/27/reading-searching-for-mobilenetv3/))

[Neural Architecture Search: A Survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf) (Thomas Elsken, JMLR 2019, [note](https://zhuanlan.zhihu.com/p/123144164))

[Neural Architecture Design for GPU-Efficient Networks](https://arxiv.org/pdf/2006.14090.pdf) (Ming Lin, 2020, [code](https://github.com/idstcv/GPU-Efficient-Networks), [note](https://zhuanlan.zhihu.com/p/151042330))

#### Evidential Deep Learning and Uncertainty

[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) (Yarin Gal, ICML 2016, [note](https://zhuanlan.zhihu.com/p/82108924))

[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://papers.nips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf) (Yarin Gal, NeuIPS 2016)

[What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf) (Alex Kendall, NeuIPS 2017, [note](https://zhuanlan.zhihu.com/p/98756147))

[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) (Balaji Lakshminarayanan, NeuIPS 2017)

[Evidential Deep Learning to Quantify Classification Uncertainty](https://papers.nips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf) (Murat Sensoy, NeuIPS 2018)

[Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/pdf/1906.02530.pdf) (Yaniv Ovadia, NeuIPS 2019)

### Recommendation System

[Deep Learning based Recommender System: A Survey and New Perspectives](http://de.arxiv.org/pdf/1707.07435v5) (Shuai Zhang, 2017, [code](https://github.com/cheungdaven/DeepRec), [note](https://www.cnblogs.com/z1141000271/p/11399916.html))

[A review on deep learning for recommender systems: challenges and remedies](https://link.springer.com/article/10.1007/s10462-018-9654-y) (Zeynep Batmaz, 2018, [note](https://blog.csdn.net/qq_35771020/article/details/88759986))

#### News Recommendation

[Google News Personalization: Scalable Online Collaborative Filtering](http://wwwconference.org/www2007/papers/paper570.pdf) (Abhinandan Das, WWW 2007, [note](https://blog.csdn.net/jj12345jj198999/article/details/12654531))

[Personalized News Recommendation Based on Click Behavior](http://www.cs.northwestern.edu/~jli156/IUI224-liu.pdf) (Jiahui Liu, 2010, [note](https://www.jianshu.com/p/f3d147fbce3f))

[Personalized Recommendation on Dynamic Content Using Predictive Bilinear Models](http://wwwconference.org/www2009/proceedings/pdf/p691.pdf) (Wei Chu, WWW 2009, [note](https://zhuanlan.zhihu.com/p/75484786))

[A Contextual-Bandit Approach to Personalized News Article Recommendation](http://wwwconference.org/proceedings/www2010/www/p661.pdf) (Lihong Li, WWW 2010, [note](https://zhuanlan.zhihu.com/p/34940176))

[A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf) (Ali Elkahky, WWW 2015, [note](https://zhuanlan.zhihu.com/p/56384078))

#### Click-Through Rate

[Factorization Machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf) (Steffen Rendle, ICDM 2010, [note](https://zhuanlan.zhihu.com/p/50426292))

[Higher-Order Factorization Machine](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf) (Mathieu Blondel, NIPS 2016)

[Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) (Yu-Chin Juan, RecSys 2016, [note](https://blog.csdn.net/Dby_freedom/article/details/84899120))

[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/Proceedings/2017/0239.pdf) (Huifeng Guo, IJCAI 2017, [note](https://blog.csdn.net/Dby_freedom/article/details/85263694))

[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) (Guorui Zhou, KDD 2018, [note](https://www.jianshu.com/p/7af364dcea12))

[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf) (Guorui Zhou, AAAI 2019, [code](https://github.com/mouna99/dien), [note](https://zhuanlan.zhihu.com/p/50758485))

[Representation Learning-Assisted Click-Through Rate Prediction](https://www.ijcai.org/Proceedings/2019/0634.pdf) (Wentao Ouyang, IJCAI 2019, [note](https://zhuanlan.zhihu.com/p/102075293))

[Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction](https://arxiv.org/pdf/1906.03776.pdf) (Wentao Ouyang, KDD 2019, [code](https://github.com/oywtece/dstn), [note](https://www.sohu.com/a/326460892_99979179))

[Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf) (Bin Liu, WWW 2019, [note](https://blog.csdn.net/w55100/article/details/90601310))

[Interpretable Click-Through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/pdf/10.1145/3336191.3371785) (Zeyu Li, WSDM 2020)

[User Behavior Retrieval for Click-Through Rate Prediction](https://arxiv.org/pdf/2005.14171.pdf) (Jiarui Qin, SIGIR 2020)

[Deep Time-Stream Framework for Click-through Rate Prediction by Tracking Interest Evolution](https://arxiv.org/pdf/2001.03025.pdf) (Shu-Ting Shi, AAAI 2020)

[Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://aaai.org/ojs/index.php/AAAI/article/view/5346) (Zequn Lyu, AAAI 2020, [code](https://github.com/lvze92/DMR), [note](https://www.jianshu.com/p/60eed27e06d4))

### Big Data

[The Google File System](http://index-of.es/Misc/pdf/google_file_system.pdf) (Sanjay Ghemawat, SOSP 2003, [note](https://zhuanlan.zhihu.com/p/28155582))

[MapReduce: Simplified Data Processing on Large Clusters](https://homeostasis.scs.carleton.ca/~soma/distos/2008-03-24/mapreduce-osdi04.pdf) (Jeffrey Dean, OSDI 2004, [note](https://juejin.im/post/6844903812784717831))

[Bigtable: A Distributed Storage System for Structured Data](https://fenix.tecnico.ulisboa.pt/downloadFile/845043405442710/10.g-bigtable-osdi06.pdf) (Fay Chang, OSDI 2006, [note](https://zh.wikipedia.org/wiki/Bigtable))

### Tool

[FudanNLP: A Toolkit for Chinese Natural Language Processing](https://www.aclweb.org/anthology/P13-4009) (Xipeng Qiu, ACL 2013, [code](https://github.com/FudanNLP/fnlp))

[LIBSVM: A library for support vector machines](http://www.csie.ntu.edu.tw/~cjlin/libsvm) (Chih-Jen Lin, 2011, [code](https://github.com/cjlin1/libsvm), [note](https://blog.csdn.net/s9434/article/details/75091602))

[HemI: A Toolkit for Illustrating Heatmaps](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0111988&type=printable) (Wankun Deng, 2014, [code](http://hemi.biocuckoo.org/))