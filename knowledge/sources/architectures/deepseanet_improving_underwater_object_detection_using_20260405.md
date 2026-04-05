---
title: DeepSeaNet: Improving Underwater Object Detection using EfficientDet
url: http://arxiv.org/abs/2306.06075v2
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:59:17.079086+00:00
word_count: 265
---

# DeepSeaNet: Improving Underwater Object Detection using EfficientDet

**Authors:** Sanyam Jain
**Published:** 2023-05-26
**arXiv ID:** http://arxiv.org/abs/2306.06075v2
**Categories:** cs.CV, cs.LG

## Abstract

Marine animals and deep underwater objects are difficult to recognize and monitor for safety of aquatic life. There is an increasing challenge when the water is saline with granular particles and impurities. In such natural adversarial environment, traditional approaches like CNN start to fail and are expensive to compute. This project involves implementing and evaluating various object detection models, including EfficientDet, YOLOv5, YOLOv8, and Detectron2, on an existing annotated underwater dataset, called the Brackish-Dataset. The dataset comprises annotated image sequences of fish, crabs, starfish, and other aquatic animals captured in Limfjorden water with limited visibility. The aim of this research project is to study the efficiency of newer models on the same dataset and contrast them with the previous results based on accuracy and inference time. Firstly, I compare the results of YOLOv3 (31.10% mean Average Precision (mAP)), YOLOv4 (83.72% mAP), YOLOv5 (97.6%), YOLOv8 (98.20%), EfficientDet (98.56% mAP) and Detectron2 (95.20% mAP) on the same dataset. Secondly, I provide a modified BiSkFPN mechanism (BiFPN neck with skip connections) to perform complex feature fusion in adversarial noise which makes modified EfficientDet robust to perturbations. Third, analyzed the effect on accuracy of EfficientDet (98.63% mAP) and YOLOv5 by adversarial learning (98.04% mAP). Last, I provide class activation map based explanations (CAM) for the two models to promote Explainability in black box models. Overall, the results indicate that modified EfficientDet achieved higher accuracy with five-fold cross validation than the other models with 88.54% IoU of feature maps.
