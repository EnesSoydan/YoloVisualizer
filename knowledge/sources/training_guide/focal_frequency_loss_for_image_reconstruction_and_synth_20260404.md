---
title: Focal Frequency Loss for Image Reconstruction and Synthesis
url: http://arxiv.org/abs/2012.12821v3
topic: training_guide
source_type: arxiv
fetched_at: 2026-04-04T19:24:05.077671+00:00
word_count: 169
---

# Focal Frequency Loss for Image Reconstruction and Synthesis

**Authors:** Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy
**Published:** 2020-12-23
**arXiv ID:** http://arxiv.org/abs/2012.12821v3
**Categories:** cs.CV, cs.LG, eess.IV

## Abstract

Image reconstruction and synthesis have witnessed remarkable progress thanks to the development of generative models. Nonetheless, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we show that narrowing gaps in the frequency domain can ameliorate image reconstruction and synthesis quality further. We propose a novel focal frequency loss, which allows a model to adaptively focus on frequency components that are hard to synthesize by down-weighting the easy ones. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent bias of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve popular models, such as VAE, pix2pix, and SPADE, in both perceptual quality and quantitative performance. We further show its potential on StyleGAN2.
