---
title: TerraGen: A Unified Multi-Task Layout Generation Framework for Remote Sensing Data Augmentation
url: http://arxiv.org/abs/2510.21391v1
topic: training_guide
source_type: arxiv
fetched_at: 2026-04-05T12:02:47.341131+00:00
word_count: 210
---

# TerraGen: A Unified Multi-Task Layout Generation Framework for Remote Sensing Data Augmentation

**Authors:** Datao Tang, Hao Wang, Yudeng Xin, Hui Qiao, Dongsheng Jiang, Yin Li et al.
**Published:** 2025-10-24
**arXiv ID:** http://arxiv.org/abs/2510.21391v1
**Categories:** cs.CV

## Abstract

Remote sensing vision tasks require extensive labeled data across multiple, interconnected domains. However, current generative data augmentation frameworks are task-isolated, i.e., each vision task requires training an independent generative model, and ignores the modeling of geographical information and spatial constraints. To address these issues, we propose \textbf{TerraGen}, a unified layout-to-image generation framework that enables flexible, spatially controllable synthesis of remote sensing imagery for various high-level vision tasks, e.g., detection, segmentation, and extraction. Specifically, TerraGen introduces a geographic-spatial layout encoder that unifies bounding box and segmentation mask inputs, combined with a multi-scale injection scheme and mask-weighted loss to explicitly encode spatial constraints, from global structures to fine details. Also, we construct the first large-scale multi-task remote sensing layout generation dataset containing 45k images and establish a standardized evaluation protocol for this task. Experimental results show that our TerraGen can achieve the best generation image quality across diverse tasks. Additionally, TerraGen can be used as a universal data-augmentation generator, enhancing downstream task performance significantly and demonstrating robust cross-task generalisation in both full-data and few-shot scenarios.
