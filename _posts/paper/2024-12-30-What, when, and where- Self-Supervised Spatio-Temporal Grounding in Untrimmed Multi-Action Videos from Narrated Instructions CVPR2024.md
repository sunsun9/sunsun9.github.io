---
layout: post
title: 'What, when, and where? Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions CVPR2024'
subtitle: 'What, when, and where? 根据解释说明在未剪裁的多动作视频中进行自监督时空定位训练'
date: 2024-12-28
author: Sun
cover: 'https://pic1.imgdb.cn/item/677202e7d0e0a243d4ec5bac.png'
tags: 论文阅读
---

> [What, when, and where? Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_What_When_and_Where_Self-Supervised_Spatio-Temporal_Grounding_in_Untrimmed_Multi-Action_CVPR_2024_paper.html)
> 1哥伦比亚大学, 2法兰克福歌德大学, 3波恩大学, 4麻省理工学院 CSAIL, 5Quality Match GmbH, 6IBM Research AI, 7麻省理工学院-IBM Watson AI 实验室

# 1.文章针对痛点

首先明确这篇文章关注的是自监督时空定位任务。

文章提到说现有的模型经常是在人工注释的句子和bounding box的监督环境下训练的，这种模式限制了模型泛化到训练数据分布之外的场景。

# 2.主要贡献

为了解决这个问题，文章提出可以利用多模态自监督学习，也就是可以使用“免费”的数据源(没有标签)；例如，来自大规模指导视频中的视频和自动语音识别，来在没有人工注释的情况下学习特征。

* 因此，文章提出了一个定位方法，<mark>使用基于*ASR* 转录的视频文本对</mark>，学习无文本事件的空间特征和事件范围（有点伪标签的意思）。具体来说，利用两种不同的视觉表征，一种是基于所有帧信息的全局特征表示来定义事件的时间范围；另一种是基于单帧的局部表征，学习空间定位。

> 但是这种方法会存在一个问题, 就是*ASR* 转录的得到的文本, 可能存在一定的噪声, 可能有的文本是一种幻觉(即描述的内容并没有在视频中出现); 或者是描述的事件并不是准确对齐的, 可能是分散在多个帧中.

* 为了解决上述的这个衍生问题, <mark>文章提出了一种选择帧的方法, 仅捕捉对训练有用的帧.</mark> 具体来说, 就是寻找那些匹配对应文本词汇的帧, 提出了一种*Sinkhorn* 最优传输的选择策略.
* 此外, <mark>文章还提出了一种新的基准*Grounding YouTube* .</mark> 该基准与其他基准测试有两点不同：首先，通过使用多个中心点注释，它专注于动作本身的建立，而不是通常标记的交互人或物体；其次，视频中多个动作的密集注释使模型能够对长而逼真的未剪辑视频中的动作建立进行基准测试，而现有的基准测试通常是预先剪辑的. 该基准测试提供了针对 512 种不同事件类型的查询和超过 *5K* 个时空注释. ![比较图](https://pic.imgdb.cn/item/676f91d9d0e0a243d4ebe0da.png)

文章对贡献的总结如下:

1. 提出了一个基于弱对齐多模态监督的未修剪视频时空基础框架，无需人工注释，采用全局和局部表征学习相结合的方式学习动作的时空范围。
2. 为了促进这项任务，提出了一种基于 *Sinkhorn-Knopp* 最优传输的帧选择策略，该策略可提高获取的学习样本的质量，从而实现更有效的监督。
3. 提供了新的基准和注释来评估现实世界多动作教学视频数据上的这一具有挑战性的问题。

# 3.实现流程

还是先看一下整体的架构图![整体的架构图](https://pic1.imgdb.cn/item/6770e890d0e0a243d4ec11ed.png)
按照这篇文章的描述，整个模型可以分为4个过程。

1. 提取特征：和其他所有方法一样，第一步都是先提取视觉和文本特征；这里需要特别注意的是，这个提供的文本特征是*ASR* 转录的文本。
2. 对齐视觉文本特征：这个就是为了对齐视觉和文本模态。
3. 根据前面提到的*Sinkhorn-Knopp* 最优传输的帧选择策略，选择与文本更相关的特征用于后面的步骤。
4. 生成预测的事件管道和分类标签：这个全局表征学习是为了提取时间管道，局部表征学习是为了提取空间管道。

# 4.实现细节

* **分配矩阵计算**：后面的跨模态最佳传输机制需要应用分配矩阵Q上，而Q是投影后的多模态特征的相似度，可以简单的表示成（1）。也就是这一步只是计算文本和视频模态之间的相似度。

$$
P=g(\mathcal{S})\bigotimes f(\mathcal{V})^{\top}\in\mathbb{R}^{K\times U}(1)
$$

* **帧选择**：首先需要进行这一步的目的是，因为得到的文本描述是通过ASR转录的，其中包含的信息是多于真实的任务表述的，也就是说上一步匹配的高相似度的文本和视觉对可能是不需要的。这里提到的是使用*Sinkhorn-Knopp*  算法实现的。
* **用于空间定位的局部表征学习**：为了捕捉多模态交互之间的更细粒度的信息，文章设计了上图(d)的模块，这个模块和其他论文的方法差不多。只不过这里用了对比损失，表示如（2）(其中的k≠l)。

$$
-\frac{1}{B}\sum_{l=1}^{B}\left[\left(\log\frac{e^{\bar{V}_{l}\cdot\bar{S}_{l}-\delta}}{e^{\bar{V}_{l}\cdot\bar{S}_{l}-\delta}+\sum_{k=1}^{B}e^{\bar{V}_{k}^{imp}\cdot\bar{S}_{l}}}\right)+\left(\log\frac{e^{\bar{V}_{l}\cdot\bar{S}_{l}-\delta}}{e^{\bar{V}_{l}\cdot\bar{S}_{l}-\delta}+\sum_{k=1}^{B}e^{\bar{V}_{l}\cdot\bar{S}_{k}^{imp}}}\right)\right](2)
$$

* 用于实践的全局表征学习：这个部分和局部表征学习一样，也是使用了对比学习，损失和上一步的损失公式一样。
  这两步的过程大概可以表示为下图：![过程图](https://pic1.imgdb.cn/item/6770ef5cd0e0a243d4ec13ea.png)

# 5.模型性能

这篇文章的工作是提出了利用全局表征和局部表征分别捕捉时间管道和空间管道，此外，还提出了一个数据集用于自监督时空动作定位任务。

所以在实验部分大部分都是在各种数据集上与*SOTA* 方法比较。

下面展示文章的实验部分：
![比较1](https://pic1.imgdb.cn/item/677200d6d0e0a243d4ec5ad0.png)
![比较2](https://pic1.imgdb.cn/item/67720119d0e0a243d4ec5aee.png)
![比较3](https://pic1.imgdb.cn/item/6772013ad0e0a243d4ec5afb.png)

消融实验部分：
![消融实验1](https://pic1.imgdb.cn/item/67720168d0e0a243d4ec5b10.png)
![消融实验2](https://pic1.imgdb.cn/item/6772018ad0e0a243d4ec5b20.png)

# 6.改进/挑战/问题

对于这篇文章我觉得可学习的部分就是这个任务，即使用自监督的时空动作定位；其次，就是提到了一个帧选择内容，最近在很多文章中都看到了帧选择这个模块，说明这个模块肯定是能够有效改善模型的性能。

* 改进：对比学习基本都是需要构建正负样本对的，这里视频模态的正样本对和负样本对都是文本模态，猜测应该是基于伪标签分配的正负样本对，感觉未来的话，是不是该可以基于伪标签这个方面进行改进。或者很多对比学习都是需要构建良好的正负样本对，感觉也可以从这个地方进行改进，但是针对正负样本对这个问题，文章没有提到怎么构建的。

