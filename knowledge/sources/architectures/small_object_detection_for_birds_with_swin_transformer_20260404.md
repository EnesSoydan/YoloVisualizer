---
title: Small Object Detection for Birds with Swin Transformer
url: http://arxiv.org/abs/2511.22310v1
topic: architectures
source_type: arxiv
fetched_at: 2026-04-04T19:22:20.208883+00:00
word_count: 226
---

# Small Object Detection for Birds with Swin Transformer

**Authors:** Da Huo, Marc A. Kastner, Tingwei Liu, Yasutomo Kawanishi, Takatsugu Hirayama, Takahiro Komamizu et al.
**Published:** 2025-11-27
**arXiv ID:** http://arxiv.org/abs/2511.22310v1
**Categories:** cs.CV

## Abstract

Object detection is the task of detecting objects in an image. In this task, the detection of small objects is particularly difficult. Other than the small size, it is also accompanied by difficulties due to blur, occlusion, and so on. Current small object detection methods are tailored to small and dense situations, such as pedestrians in a crowd or far objects in remote sensing scenarios. However, when the target object is small and sparse, there is a lack of objects available for training, making it more difficult to learn effective features. In this paper, we propose a specialized method for detecting a specific category of small objects; birds. Particularly, we improve the features learned by the neck; the sub-network between the backbone and the prediction head, to learn more effective features with a hierarchical design. We employ Swin Transformer to upsample the image features. Moreover, we change the shifted window size for adapting to small objects. Experiments show that the proposed Swin Transformer-based neck combined with CenterNet can lead to good performance by changing the window sizes. We further find that smaller window sizes (default 2) benefit mAPs for small object detection.
