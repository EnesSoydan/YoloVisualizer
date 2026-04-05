---
title: EfficientDet: Scalable and Efficient Object Detection
url: http://arxiv.org/abs/1911.09070v7
topic: architectures
source_type: arxiv
fetched_at: 2026-04-04T19:21:24.798626+00:00
word_count: 179
---

# EfficientDet: Scalable and Efficient Object Detection

**Authors:** Mingxing Tan, Ruoming Pang, Quoc V. Le
**Published:** 2019-11-20
**arXiv ID:** http://arxiv.org/abs/1911.09070v7
**Categories:** cs.CV, cs.LG, eess.IV

## Abstract

Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with single model and single-scale, our EfficientDet-D7 achieves state-of-the-art 55.1 AP on COCO test-dev with 77M parameters and 410B FLOPs, being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Code is available at https://github.com/google/automl/tree/master/efficientdet.
