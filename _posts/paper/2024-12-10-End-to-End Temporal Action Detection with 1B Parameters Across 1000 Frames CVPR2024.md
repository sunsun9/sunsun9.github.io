---
layout: post
title: 'End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames CVPR2024😊'
subtitle: '跨1000帧、具有1B参数量的端到端时序动作检测'
date: 2024-12-10
author: Sun
cover: 'https://pic.imgdb.cn/item/6757b1bdd0e0a243d4e0d7fa.png'
tags: 论文阅读
---


> 💐💐[提供代码](https://github.com/sming256/OpenTAD/tree/main/configs/adatad)
>
> 📌作者单位
>
> 1. King Abdullah University of Science and Technology (KAUST) 
> 2. 4Paradigm Inc

# 1.文章针对痛点

这篇文章提出主要是为了解决目前*TAD* 任务下的端到端模型的局限性，由于内存瓶颈，现有的端到端模型大多是规模有限或者数据量有限，但是作者认为这会明显限制*TAD* 的性能。

其次，作者提到现有的端到端*TAD* 方法使用计算密集的全微调，而这种方式在迁移学习过程中，会带来灾难性遗忘或过拟合的问题。

# 2.主要贡献

为了解决上述问题，作者采用了结合端到端训练和模型扩展优势的方法，提出了对于TAD的适应器微调的方法，*AdaTAD*。

具体来说，作者采用了以下的几种策略;

* 采用了一种更节省内存的帧表示模式。因为作者发现目前基于特征的方法在片段表示上过于冗余。
* 采用了参数高效的微调技术。作者提到通过这种技术可以最大限度地减少内存使用量，并减轻迁移学习中的过拟合。在这一部分，本文提出了**时间信息适应器*TIA***，作者提到这个适应器被注入主干层之间，是微调期间唯一可学习的组件，它集成了深度卷积以聚合来自相邻帧的信息上下文。
* 为了进一步提高内存效率，提出了一种更轻量的方法。通过将*TIA* 放入主干网络旁边而不是主干网络内部。

原文将文章贡献总结为了以下几部分：

1. 为 *TAD* 引入了一个高效的端到端框架，将模型大小扩大到 10 亿个参数，将输入数据扩大到 1,536 帧。随着规模扩大，实现了模型的持续的性能改进，这揭示了规模扩大对 *TAD* 的重要性。
2. 提出了一种新颖的时间信息适配器，以减少内存并聚合 TAD 的时间上下文。这些适配器的不同变体旨在在性能和内存成本之间进行权衡。据我们所知，我们是第一个将适配器机制引入 TAD 的人。
3. 本文方法在四个 TAD 数据集上实现了最先进的性能。值得注意的是，这是第一个端到端方法，其性能大大优于以前的基于特征的方法。

与其他的基于特征的方法以及其他的端到端模型对比![模型对比](https://pic.imgdb.cn/item/675522a3d0e0a243d4dfdbdf.png)

# 3.实现流程

这篇文章虽然说是端到端的模型，也是基于大模型进行微调的。时序动作检测的基本流程就是前面一个视频提取器和编码器的*backbone*，后面使用一个检测头。这篇文章就是微调前面的*backbone*。

我觉得就是仿照目前的大模型微调的方法，加入了Adapter进行微调，显而易见使用Adapter会大大减少需要训练的参数，所以这篇文章也是利用它减少内存利用，从而可以让backbone更复杂或者输入的数据更多。

文章提出了两种样式，下图展示了本文的模型架构图。**(a)**展示的传统的离线方法，也就是不训练视频特征提取和编码器；**(b)**全微调的方法，整个模型都会进行训练，但是对于大模型来说，需要训练的参数量太大了；**(c)**展示的本文提出了标准方法*AdaTAD*，这个方法是在*backbone* 的每一个基本块之后都加一个*TIA*，虽然参数量相较于全微调是减少的，但是反向传播时依然需要经过整个大模型网络；**(d)**展示的精简版的*AdaTAD*，这个版本将*TIA* 放到了模型外，这样在进行反向传播的时候，不需要经过整个大模型网络。![模型架构图](https://pic.imgdb.cn/item/675695f1d0e0a243d4e03fdc.png)

# 4.实现细节

这篇文章确实比较简单，就是加了一个适配器。文章提到是常用的适配器没有关注时间信息，所以这篇文章的适配器加了一个时间深度卷积层来关注相邻帧的时间上下文信息。提出的TIA架构图如下，公式表示如下。

$$
\overline{\boldsymbol{x}} = \boldsymbol{\sigma}\left(\boldsymbol{W}_{\mathrm{down}}^{\top} \cdot \boldsymbol{x}\right), \\
\hat{\boldsymbol{x}} = \boldsymbol{W}_{\mathrm{mid}}^{\top} \cdot \mathbf{D W C o n v}_{k}(\overline{\boldsymbol{x}})+\overline{\boldsymbol{x}}, \\
\boldsymbol{x}^{\prime} = \alpha \cdot \boldsymbol{W}_{\mathrm{up}}^{\top} \cdot \hat{\boldsymbol{x}}+\boldsymbol{x}
$$

![TIA](https://pic.imgdb.cn/item/67569794d0e0a243d4e04065.png)

# 5.模型性能

根据文章描述，*AdaTAD* 在使用一些扩展策略后，性能可以超过基于特征的*SOTA* 方法（原文描述的是第一个性能超过基于特征方法的端到端方法）。
与其他*SOTA* 方法：![SOTA方法比较](https://pic.imgdb.cn/item/6757adf1d0e0a243d4e0d347.png)

*Table4* 展示了*Adapter* 的消融实验，验证其有效性。根据实验结果可以看到，使用*Adapter* 确实大大减少了训练参数，并且减少内存使用量，可以让更多的内存用于扩展上。*Table7* 展示了*Adapter* 的不同架构对性能的影响。![Adapter的消融实验](https://pic.imgdb.cn/item/6757aef3d0e0a243d4e0d4c5.png)
![Adapter的不同架构实验](https://pic.imgdb.cn/item/6757b02ed0e0a243d4e0d573.png)

*Table5* 和*Table6* 分别展示了扩展数据和扩展模型对性能的影响。![扩展的消融实验](https://pic.imgdb.cn/item/6757afb5d0e0a243d4e0d509.png)

# 6.改进/挑战/问题

* 改进：这篇文章既然是在大模型上微调，感觉也可以从其他的微调方式下手，现在其他任务基于大模型的微调有很多微调方式。
* 这篇文章整体上来说思想是很简单的，就是加了一个适配器而已，感觉就是CLIP-Adapter差不多的想法和架构，至于扩展这个问题，因为我认为想要提升模型性能，无非就是数据多一点或者模型更复杂一点。所以这篇文章我觉得整体确实很简单，可能像文章说的那样是第一个在TAD上使用Adapter的文章吧。

