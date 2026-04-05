---
title: PVAFN: Point-Voxel Attention Fusion Network with Multi-Pooling Enhancing for 3D Object Detection
url: http://arxiv.org/abs/2408.14600v1
topic: general
source_type: arxiv
fetched_at: 2026-04-04T19:25:05.534393+00:00
word_count: 222
---

# PVAFN: Point-Voxel Attention Fusion Network with Multi-Pooling Enhancing for 3D Object Detection

**Authors:** Yidi Li, Jiahao Wen, Bin Ren, Wenhao Li, Zhenhuan Xu, Hao Guo et al.
**Published:** 2024-08-26
**arXiv ID:** http://arxiv.org/abs/2408.14600v1
**Categories:** cs.CV

## Abstract

The integration of point and voxel representations is becoming more common in LiDAR-based 3D object detection. However, this combination often struggles with capturing semantic information effectively. Moreover, relying solely on point features within regions of interest can lead to information loss and limitations in local feature representation. To tackle these challenges, we propose a novel two-stage 3D object detector, called Point-Voxel Attention Fusion Network (PVAFN). PVAFN leverages an attention mechanism to improve multi-modal feature fusion during the feature extraction phase. In the refinement stage, it utilizes a multi-pooling strategy to integrate both multi-scale and region-specific information effectively. The point-voxel attention mechanism adaptively combines point cloud and voxel-based Bird's-Eye-View (BEV) features, resulting in richer object representations that help to reduce false detections. Additionally, a multi-pooling enhancement module is introduced to boost the model's perception capabilities. This module employs cluster pooling and pyramid pooling techniques to efficiently capture key geometric details and fine-grained shape structures, thereby enhancing the integration of local and global features. Extensive experiments on the KITTI and Waymo datasets demonstrate that the proposed PVAFN achieves competitive performance. The code and models will be available.
