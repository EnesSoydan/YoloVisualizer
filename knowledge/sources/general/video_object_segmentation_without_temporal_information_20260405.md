---
title: Video Object Segmentation Without Temporal Information
url: http://arxiv.org/abs/1709.06031v2
topic: general
source_type: arxiv
fetched_at: 2026-04-05T12:03:56.837903+00:00
word_count: 240
---

# Video Object Segmentation Without Temporal Information

**Authors:** Kevis-Kokitsi Maninis, Sergi Caelles, Yuhua Chen, Jordi Pont-Tuset, Laura Leal-Taixé, Daniel Cremers et al.
**Published:** 2017-09-18
**arXiv ID:** http://arxiv.org/abs/1709.06031v2
**Categories:** cs.CV

## Abstract

Video Object Segmentation, and video processing in general, has been historically dominated by methods that rely on the temporal consistency and redundancy in consecutive video frames. When the temporal smoothness is suddenly broken, such as when an object is occluded, or some frames are missing in a sequence, the result of these methods can deteriorate significantly or they may not even produce any result at all. This paper explores the orthogonal approach of processing each frame independently, i.e disregarding the temporal information. In particular, it tackles the task of semi-supervised video object segmentation: the separation of an object from the background in a video, given its mask in the first frame. We present Semantic One-Shot Video Object Segmentation (OSVOS-S), based on a fully-convolutional neural network architecture that is able to successively transfer generic semantic information, learned on ImageNet, to the task of foreground segmentation, and finally to learning the appearance of a single annotated object of the test sequence (hence one shot). We show that instance level semantic information, when combined effectively, can dramatically improve the results of our previous method, OSVOS. We perform experiments on two recent video segmentation databases, which show that OSVOS-S is both the fastest and most accurate method in the state of the art.
