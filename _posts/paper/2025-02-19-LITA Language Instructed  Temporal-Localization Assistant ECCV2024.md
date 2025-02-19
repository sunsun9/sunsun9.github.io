---
layout: post
title: 'LITA: Language Instructed  Temporal-Localization Assistant ECCV2024'
subtitle: 'LITA：语言指导的时间定位助手'
date: 2025-02-19
author: Sun
cover: 'https://pic1.imgdb.cn/item/67b41fc3d0e0a243d4006c00.png'
tags: 论文阅读
---

> [DDG-Net: Discriminability-Driven Graph Network for Weakly-supervised Temporal Action Localization](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08090.pdf)
> 
> 1北京邮政与电信大学，中国；
> 2自动化研究所，中国科学院，中国；
> 3人工智能与机器人技术中心，HKISI_CAS，香港，中国；
> 4号中国科学院，UCAS，中国

# 1.文章针对痛点

这篇文章关注的是在*Video LLMs* 中时间定位的应用。

文章提到在使用现有的*Video LLMs* 时，当提示包含“*When?* "相关问题时，现有的模型往往不能准确回答时间定为或者回复不相关的信息。文章认为之所以会存在这样的问题，是有以下三方面原因：

* **时间表示**：现有的模型经常使用纯文本来表示时间戳，例如，01:22或122s；但是文章认为给定一个帧序列的集合，正确的时间戳应该依赖于帧率，而帧率又是模型无法获得的信息。
* **架构**：现有的*Video LLMs* 架构没有足够的时间分辨率来准确插入时间信息。例如，一些模型从整个视频中统一采样8帧，而8帧是不足以准确定位时间的。
* **数据**：现有的*Video LLMs* 在使用数据时，很容易忽视时间定位，在训练时，可能仅仅使用部分数据来进行微调，并且这些时间戳的准确性也没有进行验证。

# 2.主要贡献

为了解决上面提到的问题，文章提出了一个新的方法，即语言指导的时间定位助手*LITA*。

并针对上面提到的问题，进行了针对性的解决方案：

* **时间表示**：**文章引入*time tokens* 来表示相关时间戳**，允许*Video LLMs* 更好的交流关于时间的信息。具体来说，就是使用时间的相关表示，例如，视频的第一个10%，而不是像前面提到的准确的纯文本时间表示。文章采取将给定视频分成*T* 个相等长度的时间块，之后使用*T* 个*time token* 来表示相关时间定位。
* **架构**：引入了*SlowFast tokens* 以更好的时间分辨率来捕获时间信息，从而能够准确定位时间。具体来说，文章采用**密集采样视频帧**，受到*SlowFast*  架构的启发，**文章采用了两种类型的*tokens*，分别是*fast tokens* 和*slow tokens***；前者以高时间分辨率生成，提供时间信息，但数量较少；后者以低时间分辨率生成，提供空间信息，数量多。
* **数据**：在*LITA* 中，重点强调时间定位数据。
* **新任务-推理时间定位*Reasoning Temporal Localization (RTL)***：这个任务要求模型仅利用世界知识和时间推理得到问题的答案。这个新任务文章给了一个例子，如下：![一个例子](https://pic1.imgdb.cn/item/67b428b0d0e0a243d4006fcc.png)

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67b4441dd0e0a243d4007906.png)

这个架构的工作流程就是首先**视频通过编码器和投影层得到视频视觉特征**，之后按照前面提到的将所有**视频特征分为*T* 块**，**输入给*SlowFast Token Pooling* 块得到*Fast Tokens* 和*Slow Tokens***，之后**与文本*Tokens* 连接输入到LLM模块中**，得到最后的问题答案。

# 4.实现细节

* ***Time Tokens***：文章将视频分为*T* 块，使用*T* 个特定的*time tokens* 来表示时间戳。在涉及准确时间戳与time token转换上，遵循下图的描述;文章也提及虽然这种方式会引入离散错误，但是这种方式简化了时间表示。![涉及准确时间戳与time token转换](https://pic1.imgdb.cn/item/67b446e9d0e0a243d4007b50.png)
* ***SlowFast Visual Tokens***：如果不采用这种方式的话，计算量会非常大。因此模型会**首先密集采样得到*fast tokens***，也就是在前面得到的*T* 块中，将每一个块取平均值得到一个*fast token*，按照这种方式，*T* 块就是对应*T* 个*fast tokens*。**之后，模型进行稀疏采样得到*slow tokens***，这个方式是将原来的*T* 块的每一块，按照*s* 的下采样率进行平均池化（即*s×s* 大小进行平均池化），这样的话每个块就是$$\frac{M}{s^2} $$，因此，在保证最后有*M* 个*slow tokens* 的情况下，模型还可以采样$${s^2}$$个块。![简图](https://pic1.imgdb.cn/item/67b44c6bd0e0a243d4007d59.jpg)

# 5.模型性能

因为这篇文章还做了 很多其他的任务，所以在时间定位这个子任务上，并不是使用的标准时间定位任务所用的两个数据集，而是使用的*ActivityNet Captions*数据集，这个效果*mIOU* 最好接好就是28.6左右。

# 6.改进/挑战/问题/想法

* 想法：这篇文章比较是做了一个视频大语言模型，这也算是现在的热点任务；其次，我觉得提出的时间*tokens* 这个改进可以尝试利用。



