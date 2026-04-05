---
title: FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection
url: http://arxiv.org/abs/2104.10956v3
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T12:00:08.467640+00:00
word_count: 237
---

# FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection

**Authors:** Tai Wang, Xinge Zhu, Jiangmiao Pang, Dahua Lin
**Published:** 2021-04-22
**arXiv ID:** http://arxiv.org/abs/2104.10956v3
**Categories:** cs.CV, cs.AI, cs.RO

## Abstract

Monocular 3D object detection is an important task for autonomous driving considering its advantage of low cost. It is much more challenging than conventional 2D cases due to its inherent ill-posed property, which is mainly reflected in the lack of depth information. Recent progress on 2D detection offers opportunities to better solving this problem. However, it is non-trivial to make a general adapted 2D detector work in this 3D task. In this paper, we study this problem with a practice built on a fully convolutional single-stage detector and propose a general framework FCOS3D. Specifically, we first transform the commonly defined 7-DoF 3D targets to the image domain and decouple them as 2D and 3D attributes. Then the objects are distributed to different feature levels with consideration of their 2D scales and assigned only according to the projected 3D-center for the training procedure. Furthermore, the center-ness is redefined with a 2D Gaussian distribution based on the 3D-center to fit the 3D target formulation. All of these make this framework simple yet effective, getting rid of any 2D detection or 2D-3D correspondence priors. Our solution achieves 1st place out of all the vision-only methods in the nuScenes 3D detection challenge of NeurIPS 2020. Code and models are released at https://github.com/open-mmlab/mmdetection3d.
