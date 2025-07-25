---
layout: post
title: 'UniSTD: Towards Unified Spatio-Temporal Learning across Diverse Disciplines CVPR2025😊'
subtitle: 'UniSTD: 面向跨多种学科的统一时空学习'
date: 2025-07-01
author: Sun
cover: 'https://pic1.imgdb.cn/item/6861ece058cb8da5c87e7e0d.png'
tags: 论文阅读
---

> [UniSTD: Towards Unified Spatio-Temporal Learning across Diverse Disciplines](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_UniSTD_Towards_Unified_Spatio-Temporal_Learning_across_Diverse_Disciplines_CVPR_2025_paper.pdf)

> 💐💐[提供代码（但是仓库里还没有更新代码）](https://github.com/1hunters/UniSTD)
> 
> 📌作者单位
> 
> 1. 香港中文大学 MMLab
> 2. 上海人工智能实验室
> 3. 上海交通大学
> 4. 香港中文大学的另外一个单位

# 1.文章针对痛点

这篇文章想做的是构建一个跨学科、跨多任务的统一时空学习模型。现有的一些方法都是基本依赖具体任务架构，而这样会限制模型在跨任务上的泛化性和可扩展性；如果将模型直接应用到其他任务上会导致性能不稳定，从而限制他们的泛化能力并不可避免的产生昂贵的计算和存储花费。

# 2.主要贡献

所以针对上面的这个问题，文章提出可以构建一个统一的时空学习模型。具体来说，文章将时空学习转化为一个两阶段优化问题。<mark>文章使用了一个标准的*transformer* 模型来构建一个泛化结构，并且也表明这个选择允许模型利用嵌入在大规模预训练数据集的昂贵知识。在第二阶段（这是文章的主要重点）中，使用特定于任务的时空数据集对单个模型进行联合培训，以将特定于领域的知识嵌入模型中，以适合时空任务并提高模型的适应性。</mark>

尽管有良好的前景，但由于各种学科的复杂特性（例如，天气预报与交通控制），对单个模型进行联合培训非常具有挑战性，很容易触发学科之间的冲突并导致次级融合的冲突。为了以最低的训练成本来解决此问题，文章提出了一种排名自适应的专家（*MOE* ）机制，该机制基于任务属性和相互依赖性，动态优化低级适配器等级，同时根据输入特性选择性激活适配器。为了减轻等级优化的复杂性，文章使用基于辅助矩阵的方法对问题进行了重新制定，从而降低了复杂性并实现了细颗粒的等级调整。通过结合离散等级值的持续放松，从而实现了完全的可不同性，通过最小的计算开销促进了有效的优化。此外，对于最初对2D数据进行培训的*imbue* 模型，没有引入大量计算开销，我们设计了一个轻巧的时间模块，该模块结合了零元素化投影*MLP* 层。这种设计消除了对*transformer* 的计算密集型FFN层进行微调的需求，从而在保持效率的同时增强了时间建模功能。


原文总结贡献如下：

1. 我们使用在任务无关数据集（例如*OpenClip-vit*，*ImagEnetVit* ）上预测的标准*transformer* 引入了一个统一的时空建模框架，并支持有关各种时空任务的专业培训，可确保一致性的性能，交叉任务，跨任务，以及对最小值的跨性别可伸缩性的延伸性。
2. 我们通过等级自适应*MOE* 和轻巧的时间模块分解时空建模，从而可以有效地表示空间和时间依赖性。
3. 在跨越四个学科和十个任务的大规模基准的支持下，*UniSTD* 在缩放任务的数量而没有性能降低的情况下表现出令人印象深刻的表现，与当前方法相比，提高了18.8 *PSNR* 的改进


# 3.实现流程

下图展示了文章提出方法的大致流程图。文章使用的视频编码器和文本编码器都是预训练的。

![模型架构图](https://pic1.imgdb.cn/item/68621a1958cb8da5c87ebc14.png)

# 4.实现细节

* **编码器&解码器** ：文章采用了时空表征学习的标准设计的编码器，即整合了一系列的*2DConv-GroupNorm-SiLU* 层，从而逐步对输入的空间维度进行下采样。解码器则是编码器的逆过程。
* 这篇文章的重点就是排名自适应的专家混合机制。主要是为了使用不同任务的权重更新方式，这一部分从原文来看，文章通过了精心设计，并且使用了很多数学知识。可以参考原文


# 5.模型性能


# 6.改进/挑战/问题/想法

* **想法**：这篇文章是为了构建一个通用模型，主要就是设计了一个混合专家机制，这个机制设计还是使用了很多数学知识，所以这一部分不是特别理解。

