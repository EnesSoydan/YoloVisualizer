---
title: Partial Is Better Than All: Revisiting Fine-tuning Strategy for Few-shot Learning
url: http://arxiv.org/abs/2102.03983v1
topic: training_guide
source_type: arxiv
fetched_at: 2026-04-05T12:02:57.347949+00:00
word_count: 264
---

# Partial Is Better Than All: Revisiting Fine-tuning Strategy for Few-shot Learning

**Authors:** Zhiqiang Shen, Zechun Liu, Jie Qin, Marios Savvides, Kwang-Ting Cheng
**Published:** 2021-02-08
**arXiv ID:** http://arxiv.org/abs/2102.03983v1
**Categories:** cs.CV, cs.AI, cs.LG

## Abstract

The goal of few-shot learning is to learn a classifier that can recognize unseen classes from limited support data with labels. A common practice for this task is to train a model on the base set first and then transfer to novel classes through fine-tuning (Here fine-tuning procedure is defined as transferring knowledge from base to novel data, i.e. learning to transfer in few-shot scenario.) or meta-learning. However, as the base classes have no overlap to the novel set, simply transferring whole knowledge from base data is not an optimal solution since some knowledge in the base model may be biased or even harmful to the novel class. In this paper, we propose to transfer partial knowledge by freezing or fine-tuning particular layer(s) in the base model. Specifically, layers will be imposed different learning rates if they are chosen to be fine-tuned, to control the extent of preserved transferability. To determine which layers to be recast and what values of learning rates for them, we introduce an evolutionary search based method that is efficient to simultaneously locate the target layers and determine their individual learning rates. We conduct extensive experiments on CUB and mini-ImageNet to demonstrate the effectiveness of our proposed method. It achieves the state-of-the-art performance on both meta-learning and non-meta based frameworks. Furthermore, we extend our method to the conventional pre-training + fine-tuning paradigm and obtain consistent improvement.
