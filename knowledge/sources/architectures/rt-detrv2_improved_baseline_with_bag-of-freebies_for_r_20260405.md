---
title: RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer
url: http://arxiv.org/abs/2407.17140v1
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:59:39.928164+00:00
word_count: 168
---

# RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer

**Authors:** Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu
**Published:** 2024-07-24
**arXiv ID:** http://arxiv.org/abs/2407.17140v1
**Categories:** cs.CV

## Abstract

In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale feature extraction by the decoder. To enhance practicality, we propose an optional discrete sampling operator to replace the grid_sample operator that is specific to RT-DETR compared to YOLOs. This removes the deployment constraints typically associated with DETRs. For the training strategy, we propose dynamic data augmentation and scale-adaptive hyperparameters customization to improve performance without loss of speed. Source code and pre-trained models will be available at https://github.com/lyuwenyu/RT-DETR.
