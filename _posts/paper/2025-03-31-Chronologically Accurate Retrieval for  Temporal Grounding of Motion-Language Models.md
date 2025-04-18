---
layout: post
title: 'Chronologically Accurate Retrieval for  Temporal Grounding of Motion-Language Models ECCV 2024'
subtitle: '用于时间定位动作语言模型的按时间顺序的准确时刻检索'
date: 2025-03-31
author: Sun
cover: 'https://pic1.imgdb.cn/item/67e8a7f20ba3d5a1d7e69e16.png'
tags: 论文阅读
---

> [Chronologically Accurate Retrieval for  Temporal Grounding of Motion-Language Models](http://link.springer.com/chapter/10.1007/978-3-031-73636-0_19)

> ❌❌不提供代码
> 
> 📌作者单位
> 1.LY Corporation日本

# 1.文章针对痛点

这篇文章关注的应该是视频动作与文本的对齐，虽然题目说的是时刻检索、时间定位。但是感觉重点关注的还是这两种模态的对齐。

文章认为现有的方法并没有充分考虑时间元素，并且文章认为现有没有方法可以准确评估动作和文本的时间对齐是否实现。

# 2.主要贡献

所以针对上面的关于动作和文本的对齐问题，文章首先引入了一个**简单的测试，称为*Chronologically Accurate Retrieval (CAR)***。<mark>这个测试是希望获取模型识别事件顺序的能力</mark>。具体来说，文章将动作描述分解维事件，并且调整描述的顺序从而形成事件顺序不准确的描述，然后测试模型是否能力检索时间顺序正确的描述。文章提到，***CAR* 揭示了现有的模型基本不能理解动作和描述的时间关系**。

其次，为了解决对齐问题，文章进一步提出了一个简单的**策略来细化动作-语言模型**。具体来说，文章通过**使用时间顺序负样本来增强现有的对比学习架构**。

原文总结贡献如下:

1. 我们建议通过 “时间精确检索（*CAR* ）”来评估运动和语言之间的时间一致性。*CAR* 是一种新颖的测试方法，用于测量模型如何准确地区分原始运动描述和给定运动的时间错误描述。
2. 我们通过 *CAR* 发现，即使在运动语言模型中引入更大的语言模型，当前的运动语言模型也无法完全理解运动和语言的时间成分。
3. 我们提出了一个简单的解决方案来实现语言和运动之间更好的时间对应，即使用洗牌事件描述作为负文本样本，通过对比学习来训练和完善运动语言模型。由此产生的运动语言模型在文本运动检索和文本运动生成方面都取得了很高的性能。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67e8ca700ba3d5a1d7e6a577.png)

这是文章关于时间准确性检索测试的架构图，关于文章提出方法的架构图，文章没有明确展示。

# 4.实现细节

* **在动作-语言潜在空间中的时间元素** ：这个文章重点其实不是构建一个新的模型方法，而是为了验证很多模型方法不能够准确理解时间关系（尽管模型在任务表现上良好）。1️⃣首先，文章先提到了文本-动作检索，也就是检索出与文本相关的动作片段，文章使用近期提出了一个新的模型*TMR* 进行了实验。这个模型是遵循着*CLIP* 的原则，采用对比学习的方式，构建一系列正负样本对。损失公式如下1所示。2️⃣文章在上面的设置下进行了时间准确性检索测试实验。首先就是前面提到了，使用大语言模型对于原来的文本描述进行了分解，并打乱了顺序，如下图所示。3️⃣针对实验结果文章进行了分析，计算了*CAR*，计算公式如下2所示。$f(z_i^T,z_i^M)$是相当于是正样本，也就是正确的描述与动作；$f(z_i^C,z_i^M)$是负样本，也就是打乱顺序的描述与动作；最后$g$表示如果前者的分数大，则结果维1；反之，为0。![生成描述](https://pic1.imgdb.cn/item/67e8cc730ba3d5a1d7e6a5ab.png)

$$
\mathcal{L}=-\frac{1}{2N}\sum_{i}\left(\log\frac{\exp S_{ii}/\tau}{\sum_{j}\exp S_{ij}/\tau}+\log\frac{\exp S_{ii}/\tau}{\sum_{j}\exp S_{ji}/\tau}\right),	\quad(1)
$$

$$
CAR=\frac{1}{K}\sum_i^Kg\left(f(z_i^T,z_i^M),f(z_i^C,z_i^M)\right),	\quad(2)
$$

* **具有事件顺序调整的对比学习**：这个模块也就是文章针对上面的问题，提出的改进模型，下图展示了文章的模型架构。但是这个模块也没有什么特别的地方，也就是文本和动作视频的对比学习。需要注意的是，文本负样本不仅仅包含非动作描述的正确文本，也包含打乱顺序的文本描述；其次，对比学习在开始之前，进行了筛选，保留多事件动作视频。模型的损失如下公式1 2 3所示。![文章的模型架构](https://pic1.imgdb.cn/item/67e9fd120ba3d5a1d7e7383f.png)

$$
\mathcal{L}=\mathcal{L}_{t2m}+\mathcal{L}_{m2t},	\quad(1)
$$

$$
\mathcal{L}_{t2m}=-\frac{1}{N}\sum_{i}^{N}\log\frac{\exp\tilde{S}_{ii}/\tau}{\sum_{j}^{N}\exp\tilde{S}_{ij}/\tau},	\quad(2)
$$

$$
\mathcal{L}_{m2t}=-\frac{1}{N}\sum_{i}^{N}\log\frac{\exp\tilde{S}_{ii}/\tau}{\sum_{j}^{(N+K)}\exp\tilde{S}_{ji}/\tau}, \quad(3)
$$

# 5.模型性能

文章的实验部分不仅仅包含文本-图像的实验，也进行了图像-文本的实验。

实验性能效果对比图。![效果对比图1](https://pic1.imgdb.cn/item/67e9fe380ba3d5a1d7e73968.png)
![效果对比图2](https://pic1.imgdb.cn/item/67e9fe550ba3d5a1d7e73974.png)

# 6.改进/挑战/问题/想法

* **想法**：我觉得这篇文章主要提到的问题就是模型是否真的检测到了文本顺序与动作顺序的关系，这个其实在一般的时间动作检测中，并不会涉及到文本；其次，在时刻检索这种任务中，可能会涉及到这个问题。但是对于文章提出的这个问题，作为负样本的内容也就是模型希望远离的，但是在实际应用中，我不认为内容描述无误，仅仅是顺序有所调整的文本就视为负样本，我觉得这种反而也是应该检测出来的；实际应用中，可能有的描述就是语无伦次，顺序颠倒，但是描述内容是无误的，这个时候其实仍然需要检测出来。

