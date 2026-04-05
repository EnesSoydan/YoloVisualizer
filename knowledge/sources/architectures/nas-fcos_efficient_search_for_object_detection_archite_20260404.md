---
title: NAS-FCOS: Efficient Search for Object Detection Architectures
url: http://arxiv.org/abs/2110.12423v1
topic: architectures
source_type: arxiv
fetched_at: 2026-04-04T19:21:52.783764+00:00
word_count: 235
---

# NAS-FCOS: Efficient Search for Object Detection Architectures

**Authors:** Ning Wang, Yang Gao, Hao Chen, Peng Wang, Zhi Tian, Chunhua Shen et al.
**Published:** 2021-10-24
**arXiv ID:** http://arxiv.org/abs/2110.12423v1
**Categories:** cs.CV

## Abstract

Neural Architecture Search (NAS) has shown great potential in effectively reducing manual effort in network design by automatically discovering optimal architectures. What is noteworthy is that as of now, object detection is less touched by NAS algorithms despite its significant importance in computer vision. To the best of our knowledge, most of the recent NAS studies on object detection tasks fail to satisfactorily strike a balance between performance and efficiency of the resulting models, let alone the excessive amount of computational resources cost by those algorithms. Here we propose an efficient method to obtain better object detectors by searching for the feature pyramid network (FPN) as well as the prediction head of a simple anchor-free object detector, namely, FCOS [36], using a tailored reinforcement learning paradigm. With carefully designed search space, search algorithms, and strategies for evaluating network quality, we are able to find top-performing detection architectures within 4 days using 8 V100 GPUs. The discovered architectures surpass state-of-the-art object detection models (such as Faster R-CNN, Retina-Net and, FCOS) by 1.0% to 5.4% points in AP on the COCO dataset, with comparable computation complexity and memory footprint, demonstrating the efficacy of the proposed NAS method for object detection. Code is available at https://github.com/Lausannen/NAS-FCOS.
