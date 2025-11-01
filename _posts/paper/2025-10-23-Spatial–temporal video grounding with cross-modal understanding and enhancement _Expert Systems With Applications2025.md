---
layout: post
title: 'Spatial–temporal video grounding with cross-modal understanding and enhancement _Expert Systems With Applications2025🙁'
subtitle: '具有跨模态理解和增强的时空视频定位'
date: 2025-10-25
author: Sun
cover: 'https://pic1.imgdb.cn/item/68fb2cbc3203f7be00951098.png'
tags: 论文阅读
---

> [CSpatial–temporal video grounding with cross-modal understanding and enhancement ](https://www.sciencedirect.com/science/article/abs/pii/S0957417425002726)

> ❌❌[未开源代码]
> 
> 📌作者单位
> 
> 1. 湖南大学
> 2. 山东建筑大学

# 1.文章针对痛点

这篇文章认为先前的方法都会受到局限的跨模态交互和不充分特征表示的影响，缺乏时空跨模态交互学习和多模态特征增强。

# 2.主要贡献

针对上面的问题，文章提出了一种端到端的架构*CUTE*。首先，该模型**融合视觉和文本特征**，并将它们输入到*transformer* 编码器中进行跨模态信息交互，该编码器在级联架构中学习时空线索。此后，跨模态时空特征被输入到*transformer* 解码器中以输出 *STVG* 结果。此外，引入了**时空对比学习方法**，能够从时间和空间维度增强视觉和文本模态的表征学习结果。

原文总结贡献如下：

1. 我们设计了一种更灵活、更有效的跨模态交互学习方法，可以更好地提取时空多模态相关性。
2. 我们的方法不限于单模态特征学习，它通过时空对比学习过滤掉不相关或噪声特征来增强多模态特征。
3. 对两个数据集进行了全面的实验，展示了我们方法的逻辑合理性和有效性。

# 3.实现流程

文章提出模型的架构图如下所示。

![模型架构图](https://pic1.imgdb.cn/item/68fb2eb63203f7be00951dc4.png)

# 4.实现细节

* **跨模态交互模块**：该部分分为跨模态空间交互模块和跨模态时间交互模块，示意图如下。文章考虑到空间定位不需要细粒度时间信息，所以将时间和空间编码分开（该点我认为还待考虑，不过文章并没有证明这个猜想）。这个部分整体而言不是很难，空间交互模块是将视觉特征与文本特征*cat* 后，进行自注意力操作；时间交互模块，改变*cat* 方式，同样也是进行自注意力操作。![跨模态交互模块示意图](https://pic1.imgdb.cn/item/68fb2f7b3203f7be00952331.png)
* **解码器**：解码器参考*NIPS STCAT*。
* **时空对比学习**：这一部分就是正常的对比学习。空间对比学习是将真实时间片段内的作为正样本，片段外作为负样本；时间对比学习目标是最大化*GT* 帧之间的语义相似性，同时最小化*GT* 帧和非*GT* 帧之间的相似性。

# 5.模型性能

这个实验对比数据，不知道是做了实验设置上改变还是什么改变，STCAT的精度与原论文精度不一致。

![模型性能1](https://pic1.imgdb.cn/item/68fc60353203f7be009ddb87.png)

# 6.改进/挑战/问题/想法

* **想法**：

