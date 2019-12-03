---
layout: index
title: Peiqin Lin's Homepage
description: Peiqin Lin's Homepage
keywords: Peiqin Lin, 林佩勤
comments: true
permalink: /
---

### About Me

My name is Peiqin Lin (林佩勤). Currently, I am a 2nd year M.Eng. student of School of Data and Computer Science at Sun Yat-sen University and am expected to graduate in July 2020. I'm fortunate to specialize in Natural Language Processing (NLP), specifically Sentiment Analysis and Coreference Resolution, under the supervision of [Prof. Meng Yang](http://www.smartllv.com/members.html). **Note: I am seeking for Ph.D position now. If you have any interests, feel free to contact with me.**

### Research Interests

- Sentiment/Emotion Analysis
- Coreference Resolution
- Text Classification
- Deep Learning for Natural Language Processing

### Education

- Sun Yat-sen University (2018 – 2020), M.Eng. in Software Engineering. Supervisor: [Prof. Meng Yang](http://www.smartllv.com/members.html)

- South China Normal University (2014 – 2018). B.Eng. in Software Engineering

### Publications & Manuscripts

#### Publications

1. **Peiqin Lin**; Meng Yang. 2020. Hierarchical Attention Network with Pairwise Loss for Chinese Zero Pronoun Resolution. In AAAI.

- Modeling zero pronouns and candidate antecedents interactively with Hierarchical Attention Mechanism
- Using pairwise loss instead of cross entropy loss used in previous methods
- Taking the constraint of correct-antecedent similarity into account for utilizing the chain information

2. **Peiqin Lin**; Meng Yang; Jianhuang Lai. 2019. Deep Mask Memory Networks with Semantic Dependency and Context Moment for Aspect-based Sentiment Analysis. In IJCAI.

- Integrating semantic parsing information into deep memory network
- Modeling inter-aspect relation for utilizing the information of the nearby aspects
- Designing an auxiliary task to learn the sentiment distribution for the desired aspect

#### Manuscripts

1. **Peiqin Lin**; Meng Yang. 2020. A Shared-Private Model with Joint Learning and Predict-then-Extend Extraction for Targeted Sentiment Analysis. Submitted to ACL.

- A shared-private network to exploit the relation between target extraction and target classification
- An heuristic predict-then-extend algorithm for target extraction

### Projects

- [IAN](https://github.com/lpq29743/IAN) (70+ stars): TensorFlow implementation for "Interactive Attention Networks for Aspect-Level Sentiment Classification"
- [RAM](https://github.com/lpq29743/RAM) (50+ stars): TensorFlow implementation for "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"
- text_classification: implementation of text classification models, including traditional ML methods and DL methods (in Pytorch)

<!-- ### Social

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
  {% endfor %} -->