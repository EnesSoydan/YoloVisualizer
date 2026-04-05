---
title: Using Focal Loss to Fight Shallow Heuristics: An Empirical Analysis of Modulated Cross-Entropy in Natural Language Inference
url: http://arxiv.org/abs/2211.13331v1
topic: training_guide
source_type: arxiv
fetched_at: 2026-04-05T12:03:34.636112+00:00
word_count: 160
---

# Using Focal Loss to Fight Shallow Heuristics: An Empirical Analysis of Modulated Cross-Entropy in Natural Language Inference

**Authors:** Frano Rajič, Ivan Stresec, Axel Marmet, Tim Poštuvan
**Published:** 2022-11-23
**arXiv ID:** http://arxiv.org/abs/2211.13331v1
**Categories:** cs.CL, cs.LG

## Abstract

There is no such thing as a perfect dataset. In some datasets, deep neural networks discover underlying heuristics that allow them to take shortcuts in the learning process, resulting in poor generalization capability. Instead of using standard cross-entropy, we explore whether a modulated version of cross-entropy called focal loss can constrain the model so as not to use heuristics and improve generalization performance. Our experiments in natural language inference show that focal loss has a regularizing impact on the learning process, increasing accuracy on out-of-distribution data, but slightly decreasing performance on in-distribution data. Despite the improved out-of-distribution performance, we demonstrate the shortcomings of focal loss and its inferiority in comparison to the performance of methods such as unbiased focal loss and self-debiasing ensembles.
