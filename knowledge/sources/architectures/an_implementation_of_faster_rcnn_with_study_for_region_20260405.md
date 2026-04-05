---
title: An Implementation of Faster RCNN with Study for Region Sampling
url: http://arxiv.org/abs/1702.02138v2
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:59:02.726224+00:00
word_count: 112
---

# An Implementation of Faster RCNN with Study for Region Sampling

**Authors:** Xinlei Chen, Abhinav Gupta
**Published:** 2017-02-07
**arXiv ID:** http://arxiv.org/abs/1702.02138v2
**Categories:** cs.CV

## Abstract

We adapted the join-training scheme of Faster RCNN framework from Caffe to TensorFlow as a baseline implementation for object detection. Our code is made publicly available. This report documents the simplifications made to the original pipeline, with justifications from ablation analysis on both PASCAL VOC 2007 and COCO 2014. We further investigated the role of non-maximal suppression (NMS) in selecting regions-of-interest (RoIs) for region classification, and found that a biased sampling toward small regions helps performance and can achieve on-par mAP to NMS-based sampling when converged sufficiently.
