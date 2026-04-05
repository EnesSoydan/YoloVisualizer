---
title: Deep Video Prior for Video Consistency and Propagation
url: http://arxiv.org/abs/2201.11632v1
topic: general
source_type: arxiv
fetched_at: 2026-04-04T19:24:25.256255+00:00
word_count: 215
---

# Deep Video Prior for Video Consistency and Propagation

**Authors:** Chenyang Lei, Yazhou Xing, Hao Ouyang, Qifeng Chen
**Published:** 2022-01-27
**arXiv ID:** http://arxiv.org/abs/2201.11632v1
**Categories:** cs.CV, cs.AI

## Abstract

Applying an image processing algorithm independently to each video frame often leads to temporal inconsistency in the resulting video. To address this issue, we present a novel and general approach for blind video temporal consistency. Our method is only trained on a pair of original and processed videos directly instead of a large dataset. Unlike most previous methods that enforce temporal consistency with optical flow, we show that temporal consistency can be achieved by training a convolutional neural network on a video with Deep Video Prior (DVP). Moreover, a carefully designed iteratively reweighted training strategy is proposed to address the challenging multimodal inconsistency problem. We demonstrate the effectiveness of our approach on 7 computer vision tasks on videos. Extensive quantitative and perceptual experiments show that our approach obtains superior performance than state-of-the-art methods on blind video temporal consistency. We further extend DVP to video propagation and demonstrate its effectiveness in propagating three different types of information (color, artistic style, and object segmentation). A progressive propagation strategy with pseudo labels is also proposed to enhance DVP's performance on video propagation. Our source codes are publicly available at https://github.com/ChenyangLEI/deep-video-prior.
