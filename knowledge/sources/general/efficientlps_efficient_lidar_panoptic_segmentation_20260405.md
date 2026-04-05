---
title: EfficientLPS: Efficient LiDAR Panoptic Segmentation
url: http://arxiv.org/abs/2102.08009v3
topic: general
source_type: arxiv
fetched_at: 2026-04-05T12:04:04.532942+00:00
word_count: 216
---

# EfficientLPS: Efficient LiDAR Panoptic Segmentation

**Authors:** Kshitij Sirohi, Rohit Mohan, Daniel Büscher, Wolfram Burgard, Abhinav Valada
**Published:** 2021-02-16
**arXiv ID:** http://arxiv.org/abs/2102.08009v3
**Categories:** cs.CV, cs.LG, cs.RO

## Abstract

Panoptic segmentation of point clouds is a crucial task that enables autonomous vehicles to comprehend their vicinity using their highly accurate and reliable LiDAR sensors. Existing top-down approaches tackle this problem by either combining independent task-specific networks or translating methods from the image domain ignoring the intricacies of LiDAR data and thus often resulting in sub-optimal performance. In this paper, we present the novel top-down Efficient LiDAR Panoptic Segmentation (EfficientLPS) architecture that addresses multiple challenges in segmenting LiDAR point clouds including distance-dependent sparsity, severe occlusions, large scale-variations, and re-projection errors. EfficientLPS comprises of a novel shared backbone that encodes with strengthened geometric transformation modeling capacity and aggregates semantically rich range-aware multi-scale features. It incorporates new scale-invariant semantic and instance segmentation heads along with the panoptic fusion module which is supervised by our proposed panoptic periphery loss function. Additionally, we formulate a regularized pseudo labeling framework to further improve the performance of EfficientLPS by training on unlabelled data. We benchmark our proposed model on two large-scale LiDAR datasets: nuScenes, for which we also provide ground truth annotations, and SemanticKITTI. Notably, EfficientLPS sets the new state-of-the-art on both these datasets.
