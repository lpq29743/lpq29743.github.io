---
layout: page
title: About
description: Peiqin Lin's CV
keywords: Peiqin Lin, 林佩勤
comments: true
menu: about
permalink: /
---

Peiqin Lin

Email: linpq3@mail2.sysu.edu.cn

### Research Interests

- Sentiment/Emotion Analysis
- Coreference Resolution
- Text Classification
- Deep Learning for Natural Language Processing

### Education

- Sun Yat-sen University (2018 – 2020), M.E. in Software Engineering. Supervisor: [Prof. Meng Yang](http://www.smartllv.com/members.html)

- South China Normal University (2014 – 2018). B.E. in Software Engineering

### Publications

**Deep Mask Memory Networks with Semantic Dependency and Context Moment for Aspect-based Sentiment Analysis.** First author, accepted by IJCAI 2019.

- Integrating semantic parsing information into deep memory network
- Modeling inter-aspect relation for utilizing the information of the nearby aspects
- Designing an auxiliary task to learn the sentiment distribution for the desired aspect

**A Shared-Private Model with Joint Learning and Predict-then-Extend Extraction for Targeted Sentiment Analysis.** First author, submitted to AAAI 2020.

- A shared-private network to exploit the relation between target extraction and target classification
- An heuristic predict-then-extend algorithm for target extraction

**Hierarchical Attention Network with Pairwise Loss for Chinese Zero Pronoun Resolution.** First author, submitted to AAAI 2020.

- Modeling zero pronouns and candidate antecedents interactively with Hierarchical Attention Mechanism
- Using pairwise loss instead of cross entropy loss used in previous methods
- Taking the constraint of correct-antecedent similarity into account for utilizing the chain information

### Projects

- [IAN](https://github.com/lpq29743/IAN) (70+ stars): TensorFlow implementation for "Interactive Attention Networks for Aspect-Level Sentiment Classification"
- [RAM](https://github.com/lpq29743/RAM) (50+ stars): TensorFlow implementation for "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"
- text_classification: implementation of text classification models, including traditional ML methods and DL methods (in Pytorch)

### Social

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}