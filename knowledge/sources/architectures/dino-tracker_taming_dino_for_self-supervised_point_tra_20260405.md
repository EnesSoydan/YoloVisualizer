---
title: DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video
url: http://arxiv.org/abs/2403.14548v2
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T12:00:34.390091+00:00
word_count: 155
---

# DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video

**Authors:** Narek Tumanyan, Assaf Singer, Shai Bagon, Tali Dekel
**Published:** 2024-03-21
**arXiv ID:** http://arxiv.org/abs/2403.14548v2
**Categories:** cs.CV

## Abstract

We present DINO-Tracker -- a new framework for long-term dense tracking in video. The pillar of our approach is combining test-time training on a single video, with the powerful localized semantic features learned by a pre-trained DINO-ViT model. Specifically, our framework simultaneously adopts DINO's features to fit to the motion observations of the test video, while training a tracker that directly leverages the refined features. The entire framework is trained end-to-end using a combination of self-supervised losses, and regularization that allows us to retain and benefit from DINO's semantic prior. Extensive evaluation demonstrates that our method achieves state-of-the-art results on known benchmarks. DINO-tracker significantly outperforms self-supervised methods and is competitive with state-of-the-art supervised trackers, while outperforming them in challenging cases of tracking under long-term occlusions.
