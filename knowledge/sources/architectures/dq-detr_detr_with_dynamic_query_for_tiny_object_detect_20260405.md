---
title: DQ-DETR: DETR with Dynamic Query for Tiny Object Detection
url: http://arxiv.org/abs/2404.03507v6
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:58:49.249155+00:00
word_count: 189
---

# DQ-DETR: DETR with Dynamic Query for Tiny Object Detection

**Authors:** Yi-Xin Huang, Hou-I Liu, Hong-Han Shuai, Wen-Huang Cheng
**Published:** 2024-04-04
**arXiv ID:** http://arxiv.org/abs/2404.03507v6
**Categories:** cs.CV

## Abstract

Despite previous DETR-like methods having performed successfully in generic object detection, tiny object detection is still a challenging task for them since the positional information of object queries is not customized for detecting tiny objects, whose scale is extraordinarily smaller than general objects. Also, DETR-like methods using a fixed number of queries make them unsuitable for aerial datasets, which only contain tiny objects, and the numbers of instances are imbalanced between different images. Thus, we present a simple yet effective model, named DQ-DETR, which consists of three different components: categorical counting module, counting-guided feature enhancement, and dynamic query selection to solve the above-mentioned problems. DQ-DETR uses the prediction and density maps from the categorical counting module to dynamically adjust the number of object queries and improve the positional information of queries. Our model DQ-DETR outperforms previous CNN-based and DETR-like methods, achieving state-of-the-art mAP 30.2% on the AI-TOD-V2 dataset, which mostly consists of tiny objects. Our code will be available at https://github.com/hoiliu-0801/DQ-DETR.
