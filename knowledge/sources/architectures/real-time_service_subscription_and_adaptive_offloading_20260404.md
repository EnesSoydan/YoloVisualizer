---
title: Real-Time Service Subscription and Adaptive Offloading Control in Vehicular Edge Computing
url: http://arxiv.org/abs/2512.14002v1
topic: architectures
source_type: arxiv
fetched_at: 2026-04-04T19:21:32.690243+00:00
word_count: 240
---

# Real-Time Service Subscription and Adaptive Offloading Control in Vehicular Edge Computing

**Authors:** Chuanchao Gao, Arvind Easwaran
**Published:** 2025-12-16
**arXiv ID:** http://arxiv.org/abs/2512.14002v1
**Categories:** cs.DC, cs.DM

## Abstract

Vehicular Edge Computing (VEC) has emerged as a promising paradigm for enhancing the computational efficiency and service quality in intelligent transportation systems by enabling vehicles to wirelessly offload computation-intensive tasks to nearby Roadside Units. However, efficient task offloading and resource allocation for time-critical applications in VEC remain challenging due to constrained network bandwidth and computational resources, stringent task deadlines, and rapidly changing network conditions. To address these challenges, we formulate a Deadline-Constrained Task Offloading and Resource Allocation Problem (DOAP), denoted as $\mathbf{P}$, in VEC with both bandwidth and computational resource constraints, aiming to maximize the total vehicle utility. To solve $\mathbf{P}$, we propose $\mathtt{SARound}$, an approximation algorithm based on Linear Program rounding and local-ratio techniques, that improves the best-known approximation ratio for DOAP from $\frac{1}{6}$ to $\frac{1}{4}$. Additionally, we design an online service subscription and offloading control framework to address the challenges of short task deadlines and rapidly changing wireless network conditions. To validate our approach, we develop a comprehensive VEC simulator, VecSim, using the open-source simulation libraries OMNeT++ and Simu5G. VecSim integrates our designed framework to manage the full life-cycle of real-time vehicular tasks. Experimental results, based on profiled object detection applications and real-world taxi trace data, show that $\mathtt{SARound}$ consistently outperforms state-of-the-art baselines under varying network conditions while maintaining runtime efficiency.
