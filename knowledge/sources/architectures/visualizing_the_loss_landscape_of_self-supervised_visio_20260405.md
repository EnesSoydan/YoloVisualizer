---
title: Visualizing the loss landscape of Self-supervised Vision Transformer
url: http://arxiv.org/abs/2405.18042v1
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T12:00:32.889201+00:00
word_count: 263
---

# Visualizing the loss landscape of Self-supervised Vision Transformer

**Authors:** Youngwan Lee, Jeffrey Ryan Willette, Jonghee Kim, Sung Ju Hwang
**Published:** 2024-05-28
**arXiv ID:** http://arxiv.org/abs/2405.18042v1
**Categories:** cs.CV, cs.LG

## Abstract

The Masked autoencoder (MAE) has drawn attention as a representative self-supervised approach for masked image modeling with vision transformers. However, even though MAE shows better generalization capability than fully supervised training from scratch, the reason why has not been explored. In another line of work, the Reconstruction Consistent Masked Auto Encoder (RC-MAE), has been proposed which adopts a self-distillation scheme in the form of an exponential moving average (EMA) teacher into MAE, and it has been shown that the EMA-teacher performs a conditional gradient correction during optimization. To further investigate the reason for better generalization of the self-supervised ViT when trained by MAE (MAE-ViT) and the effect of the gradient correction of RC-MAE from the perspective of optimization, we visualize the loss landscapes of the self-supervised vision transformer by both MAE and RC-MAE and compare them with the supervised ViT (Sup-ViT). Unlike previous loss landscape visualizations of neural networks based on classification task loss, we visualize the loss landscape of ViT by computing pre-training task loss. Through the lens of loss landscapes, we find two interesting observations: (1) MAE-ViT has a smoother and wider overall loss curvature than Sup-ViT. (2) The EMA-teacher allows MAE to widen the region of convexity in both pretraining and linear probing, leading to quicker convergence. To the best of our knowledge, this work is the first to investigate the self-supervised ViT through the lens of the loss landscape.
