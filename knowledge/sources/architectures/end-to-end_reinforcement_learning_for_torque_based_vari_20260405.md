---
title: End-to-End Reinforcement Learning for Torque Based Variable Height Hopping
url: http://arxiv.org/abs/2307.16676v2
topic: architectures
source_type: arxiv
fetched_at: 2026-04-05T11:58:47.748601+00:00
word_count: 183
---

# End-to-End Reinforcement Learning for Torque Based Variable Height Hopping

**Authors:** Raghav Soni, Daniel Harnack, Hannah Isermann, Sotaro Fushimi, Shivesh Kumar, Frank Kirchner
**Published:** 2023-07-31
**arXiv ID:** http://arxiv.org/abs/2307.16676v2
**Categories:** cs.RO, cs.LG, eess.SY

## Abstract

Legged locomotion is arguably the most suited and versatile mode to deal with natural or unstructured terrains. Intensive research into dynamic walking and running controllers has recently yielded great advances, both in the optimal control and reinforcement learning (RL) literature. Hopping is a challenging dynamic task involving a flight phase and has the potential to increase the traversability of legged robots. Model based control for hopping typically relies on accurate detection of different jump phases, such as lift-off or touch down, and using different controllers for each phase. In this paper, we present a end-to-end RL based torque controller that learns to implicitly detect the relevant jump phases, removing the need to provide manual heuristics for state detection. We also extend a method for simulation to reality transfer of the learned controller to contact rich dynamic tasks, resulting in successful deployment on the robot after training without parameter tuning.
