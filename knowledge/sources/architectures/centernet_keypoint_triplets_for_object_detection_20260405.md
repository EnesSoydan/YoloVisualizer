---
title: CenterNet: Keypoint Triplets for Object Detection
url: http://arxiv.org/abs/1904.08189v3
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:59:51.165617+00:00
word_count: 187
---

# CenterNet: Keypoint Triplets for Object Detection

**Authors:** Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian
**Published:** 2019-04-17
**arXiv ID:** http://arxiv.org/abs/1904.08189v3
**Categories:** cs.CV

## Abstract

In object detection, keypoint-based approaches often suffer a large number of incorrect object bounding boxes, arguably due to the lack of an additional look into the cropped regions. This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs. We build our framework upon a representative one-stage keypoint-based detector named CornerNet. Our approach, named CenterNet, detects each object as a triplet, rather than a pair, of keypoints, which improves both precision and recall. Accordingly, we design two customized modules named cascade corner pooling and center pooling, which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions, respectively. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%, which outperforms all existing one-stage detectors by at least 4.9%. Meanwhile, with a faster inference speed, CenterNet demonstrates quite comparable performance to the top-ranked two-stage detectors. Code is available at https://github.com/Duankaiwen/CenterNet.
