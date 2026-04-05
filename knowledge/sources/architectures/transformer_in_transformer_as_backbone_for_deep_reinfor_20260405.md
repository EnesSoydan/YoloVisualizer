---
title: Transformer in Transformer as Backbone for Deep Reinforcement Learning
url: http://arxiv.org/abs/2212.14538v2
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T12:00:50.259774+00:00
word_count: 192
---

# Transformer in Transformer as Backbone for Deep Reinforcement Learning

**Authors:** Hangyu Mao, Rui Zhao, Hao Chen, Jianye Hao, Yiqun Chen, Dong Li et al.
**Published:** 2022-12-30
**arXiv ID:** http://arxiv.org/abs/2212.14538v2
**Categories:** cs.LG, cs.AI, cs.RO

## Abstract

Designing better deep networks and better reinforcement learning (RL) algorithms are both important for deep RL. This work focuses on the former. Previous methods build the network with several modules like CNN, LSTM and Attention. Recent methods combine the Transformer with these modules for better performance. However, it requires tedious optimization skills to train a network composed of mixed modules, making these methods inconvenient to be used in practice. In this paper, we propose to design \emph{pure Transformer-based networks} for deep RL, aiming at providing off-the-shelf backbones for both the online and offline settings. Specifically, the Transformer in Transformer (TIT) backbone is proposed, which cascades two Transformers in a very natural way: the inner one is used to process a single observation, while the outer one is responsible for processing the observation history; combining both is expected to extract spatial-temporal representations for good decision-making. Experiments show that TIT can achieve satisfactory performance in different settings consistently.
