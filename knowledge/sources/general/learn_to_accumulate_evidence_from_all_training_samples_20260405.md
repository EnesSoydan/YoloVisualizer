---
title: Learn to Accumulate Evidence from All Training Samples: Theory and Practice
url: http://arxiv.org/abs/2306.11113v2
topic: general
source_type: arxiv
fetched_at: 2026-04-05T12:03:44.286159+00:00
word_count: 200
---

# Learn to Accumulate Evidence from All Training Samples: Theory and Practice

**Authors:** Deep Pandey, Qi Yu
**Published:** 2023-06-19
**arXiv ID:** http://arxiv.org/abs/2306.11113v2
**Categories:** cs.LG, cs.AI, cs.CV

## Abstract

Evidential deep learning, built upon belief theory and subjective logic, offers a principled and computationally efficient way to turn a deterministic neural network uncertainty-aware. The resultant evidential models can quantify fine-grained uncertainty using the learned evidence. To ensure theoretically sound evidential models, the evidence needs to be non-negative, which requires special activation functions for model training and inference. This constraint often leads to inferior predictive performance compared to standard softmax models, making it challenging to extend them to many large-scale datasets. To unveil the real cause of this undesired behavior, we theoretically investigate evidential models and identify a fundamental limitation that explains the inferior performance: existing evidential activation functions create zero evidence regions, which prevent the model to learn from training samples falling into such regions. A deeper analysis of evidential activation functions based on our theoretical underpinning inspires the design of a novel regularizer that effectively alleviates this fundamental limitation. Extensive experiments over many challenging real-world datasets and settings confirm our theoretical findings and demonstrate the effectiveness of our proposed approach.
