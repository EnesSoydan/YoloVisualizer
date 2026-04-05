---
title: Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation
url: http://arxiv.org/abs/2002.00741v1
topic: cv_fundamentals
source_type: arxiv
fetched_at: 2026-04-04T19:22:40.264803+00:00
word_count: 201
---

# Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation

**Authors:** Jibang Wu, Renqin Cai, Hongning Wang
**Published:** 2020-01-29
**arXiv ID:** http://arxiv.org/abs/2002.00741v1
**Categories:** cs.IR, cs.CL, cs.LG

## Abstract

Predicting users' preferences based on their sequential behaviors in history is challenging and crucial for modern recommender systems. Most existing sequential recommendation algorithms focus on transitional structure among the sequential actions, but largely ignore the temporal and context information, when modeling the influence of a historical event to current prediction.
  In this paper, we argue that the influence from the past events on a user's current action should vary over the course of time and under different context. Thus, we propose a Contextualized Temporal Attention Mechanism that learns to weigh historical actions' influence on not only what action it is, but also when and how the action took place. More specifically, to dynamically calibrate the relative input dependence from the self-attention mechanism, we deploy multiple parameterized kernel functions to learn various temporal dynamics, and then use the context information to determine which of these reweighing kernels to follow for each input. In empirical evaluations on two large public recommendation datasets, our model consistently outperformed an extensive set of state-of-the-art sequential recommendation methods.
