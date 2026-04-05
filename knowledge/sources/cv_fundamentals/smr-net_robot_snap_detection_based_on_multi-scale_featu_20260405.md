---
title: SMR-Net:Robot Snap Detection Based on Multi-Scale Features and Self-Attention Network
url: http://arxiv.org/abs/2603.01036v1
topic: cv_fundamentals
source_type: arxiv
fetched_at: 2026-04-05T12:01:30.209994+00:00
word_count: 215
---

# SMR-Net:Robot Snap Detection Based on Multi-Scale Features and Self-Attention Network

**Authors:** Kuanxu Hou
**Published:** 2026-03-01
**arXiv ID:** http://arxiv.org/abs/2603.01036v1
**Categories:** cs.CV, cs.RO

## Abstract

In robot automated assembly, snap assembly precision and efficiency directly determine overall production quality. As a core prerequisite, snap detection and localization critically affect subsequent assembly success. Traditional visual methods suffer from poor robustness and large localization errors when handling complex scenarios (e.g., transparent or low-contrast snaps), failing to meet high-precision assembly demands. To address this, this paper designs a dedicated sensor and proposes SMR-Net, an self-attention-based multi-scale object detection algorithm, to synergistically enhance detection and localization performance. SMR-Net adopts an attention-enhanced multi-scale feature fusion architecture: raw sensor data is encoded via an attention-embedded feature extractor to strengthen key snap features and suppress noise; three multi-scale feature maps are processed in parallel with standard and dilated convolution for dimension unification while preserving resolution; an adaptive reweighting network dynamically assigns weights to fused features, generating fine representations integrating details and global semantics. Experimental results on Type A and Type B snap datasets show SMR-Net outperforms traditional Faster R-CNN significantly: Intersection over Union (IoU) improves by 6.52% and 5.8%, and mean Average Precision (mAP) increases by 2.8% and 1.5% respectively. This fully demonstrates the method's superiority in complex snap detection and localization tasks.
