---
layout: post
title: 'Collaborative Static and Dynamic Vision-Language Streams for Spatio-Temporal Video Grounding CVPR2023🙁'
subtitle: '面向时空视频定位的协同静态和动态视觉语言流'
date: 2025-10-23
author: Sun
cover: 'https://pic1.imgdb.cn/item/68f885ca3203f7be008d09b8.png'
tags: 论文阅读
---

> [Collaborative Static and Dynamic Vision-Language Streams for Spatio-Temporal Video Grounding]([https://arxiv.org/abs/2501.07972](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Collaborative_Static_and_Dynamic_Vision-Language_Streams_for_Spatio-Temporal_Video_Grounding_CVPR_2023_paper.pdf))

> ❌❌[未开源代码]
> 
> 📌作者单位
> 
> 1. 中山大学
> 2. 腾讯
> 3. 广东省信息安全技术重点实验室
> 4. 机器智能与先进计算教育部重点实验室

# 1.文章针对痛点

这篇文章考虑的问题是现有的方法通常分离了时间流和空间流，二者信息没有互相交流互补。文章认为时间流是动态流，可以让模型关注动作特征；空间流是静态流，可以让模型关注静态特征。

# 2.主要贡献

针对上面的问题，文章提出了静态视觉语言流和动态视觉语言流。其中，<mark>静态视觉语言流学习根据静态视觉线索（如外观）关注某些候选区域，而动态视觉语言流则学习理解动态视觉线索（如文本查询中描述的动作）</mark>。之后，让二者在协作块中交流互补，通过使用静态流中学习到的关注区域来引导动态流仅关注候选对象的运动。并且我们将动态流中学习到的文本运动匹配信息转移到静态流中，以帮助其进一步检查和确定目标对象并预测更一致的管子。通过上述跨流协作块，静态和动态视觉语言流都可以从另一个流中学习相互的信息。

原文总结贡献如下：

1. 我们开发了一个有效的框架，其中包含两个并行流来模拟静态-动态视觉-语言依赖关系，以实现完整的跨模式理解；
2. 我们在两个流之间提出了一种新颖的跨流协作块，以相互交换信息并实现目标对象的协作推理；
3. 我们的整体框架在 *HC-STVG*  和 *VidSTG* 数据集上实现了最先进的性能。

# 3.实现流程

文章提出模型的架构图如下所示。

![模型架构图](https://pic1.imgdb.cn/item/68f888a63203f7be008d249e.png)

# 4.实现细节

* **静态流**：参考DETR方法构建，示意图如下。
* **动态流**：在每一层中，首先对动态视觉特征和语言特征进行模态内自注意力。对于视觉特征，为了降低计算成本，按照 *TimesFormer* 将时空注意力分割成单独的注意力。

$$
\begin{aligned}
Q_{v}^{(h,w)}=W_{v}^{q}\mathcal{H}_{v}^{(h,w)},K_{v}=W_{v}^{k}\overline{\mathcal{H}}_{v},V_{v}=W_{v}^{v}\overline{\mathcal{H}}_{v}, \\
Q_{l}=W_{l}^{v}\mathcal{H}_{l},K_{l}=W_{l}^{k}\mathcal{H}_{l},V_{l}=W_{l}^{v}\mathcal{H}_{l}, \\
 \widetilde{\mathcal{H}}_{v}^{(h,w)}=\mathcal{H}_{v}^{(h,w)}+\mathrm{Attention}(Q_{v}^{(h,w)},K_{l},V_{l}), \\
\widetilde{\mathcal{H}}_{l}=\mathcal{H}_{l}+\mathrm{Attention}(Q_{l},K_{v},V_{v}), \\
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V,
\end{aligned}
$$

![示意图](https://pic1.imgdb.cn/item/68f88b563203f7be008d3cac.png)

* **交互模块**：如下图所示，在解码器每层之后让静态流和动态流进行交互。
  ![交叉模块示意图](https://pic1.imgdb.cn/item/68f88ba13203f7be008d3ff9.png)

# 5.模型性能

![模型性能1](https://pic1.imgdb.cn/item/68f88be23203f7be008d438b.png)

# 6.改进/挑战/问题/想法

* **想法**：

