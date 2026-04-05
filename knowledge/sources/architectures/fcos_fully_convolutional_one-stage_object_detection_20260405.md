---
title: FCOS: Fully Convolutional One-Stage Object Detection
url: http://arxiv.org/abs/1904.01355v5
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T12:00:05.466492+00:00
word_count: 200
---

# FCOS: Fully Convolutional One-Stage Object Detection

**Authors:** Zhi Tian, Chunhua Shen, Hao Chen, Tong He
**Published:** 2019-04-02
**arXiv ID:** http://arxiv.org/abs/1904.01355v5
**Categories:** cs.CV

## Abstract

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks. Code is available at:Code is available at: https://tinyurl.com/FCOSv1
