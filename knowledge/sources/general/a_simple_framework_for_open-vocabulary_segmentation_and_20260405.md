---
title: A Simple Framework for Open-Vocabulary Segmentation and Detection
url: http://arxiv.org/abs/2303.08131v3
topic: general
source_type: arxiv
fetched_at: 2026-04-05T12:04:28.657465+00:00
word_count: 305
---

# A Simple Framework for Open-Vocabulary Segmentation and Detection

**Authors:** Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao et al.
**Published:** 2023-03-14
**arXiv ID:** http://arxiv.org/abs/2303.08131v3
**Categories:** cs.CV

## Abstract

We present OpenSeeD, a simple Open-vocabulary Segmentation and Detection framework that jointly learns from different segmentation and detection datasets. To bridge the gap of vocabulary and annotation granularity, we first introduce a pre-trained text encoder to encode all the visual concepts in two tasks and learn a common semantic space for them. This gives us reasonably good results compared with the counterparts trained on segmentation task only. To further reconcile them, we locate two discrepancies: $i$) task discrepancy -- segmentation requires extracting masks for both foreground objects and background stuff, while detection merely cares about the former; $ii$) data discrepancy -- box and mask annotations are with different spatial granularity, and thus not directly interchangeable. To address these issues, we propose a decoupled decoding to reduce the interference between foreground/background and a conditioned mask decoding to assist in generating masks for given boxes. To this end, we develop a simple encoder-decoder model encompassing all three techniques and train it jointly on COCO and Objects365. After pre-training, our model exhibits competitive or stronger zero-shot transferability for both segmentation and detection. Specifically, OpenSeeD beats the state-of-the-art method for open-vocabulary instance and panoptic segmentation across 5 datasets, and outperforms previous work for open-vocabulary detection on LVIS and ODinW under similar settings. When transferred to specific tasks, our model achieves new SoTA for panoptic segmentation on COCO and ADE20K, and instance segmentation on ADE20K and Cityscapes.
  Finally, we note that OpenSeeD is the first to explore the potential of joint training on segmentation and detection, and hope it can be received as a strong baseline for developing a single model for both tasks in open world.
