---
layout: post
title: 'VideoGrounding-DINO: Towards Open-Vocabulary Spatio-Temporal Video
Grounding CVPR2024'
subtitle: '视频定位-DINO：面向开放词汇的时空视频定位'
date: 2024-12-18
author: Sun
cover: 'https://pic.imgdb.cn/item/676233ddd0e0a243d4e599b3.png'
tags: 论文阅读
---

> [视频定位-DINO：面向开放词汇的时空视频定位](https://openaccess.thecvf.com/content/CVPR2024/html/Wasim_VideoGrounding-DINO_Towards_Open-Vocabulary_Spatio-Temporal_Video_Grounding_CVPR_2024_paper.html)
> 
> 1穆罕默德·本·扎耶德人工智能大学 2澳大利亚国立大学 3加州大学默塞德分校 4谷歌研究中心 5林雪平大学 ¨(翻译软件译，可能有误)

# 1.文章针对痛点

首先关于视频定位任务（video grounding)，包括基于文本的视频定位、基于事件的视频定位和基于音频的视频定位，这篇文章研究的是基于文本的视频定位，即定位视频中与输入文本查询相关的时空部分。

文章针对的痛点可以从以下两个方面讲述：

* **监督式的闭集任务对模型训练有限制**。这种传统的基于闭集的方法是利用预定义的词汇和精心注释的数据，在特定数据集上进行训练；这种方式容易导致模型有限泛化到训练数据集分布之外的数据，且现有数据集具有相对小规模的数据和有限的样本种类的特点，阻碍模型有效适应未知场景。
* **开放集词汇任务数据集具有局限性**。想要训练一个高效的开放词汇视频定位模型，需要足够多的具有丰富自然语言表示和相应时空定位注释的数据集（这样才能让模型学习到更加普适的视觉文本表示，从而解决分布外问题）。但是现有的数据集规模十分有限，因此，<mark>本文作者提出怎么在数据规模有限的情况下，仍然能够提高模型在闭集和开放词汇场景下的性能。</mark>

# 2.主要贡献

为了解决上面的问题，作者在专注空间定位的基础模型上受到了启发，这些模型在大量的图像文本数据集上训练，实现对给定目标分布样本的有效泛化。因此，文章提出<mark>利用这种预训练的表示来增强他们的视频定位模型。</mark>具体来说，提出了一种类似*DETR* 架构的时空视频定位模型，并通过时间聚合增强；文章也提到，空间模块使用基础图像模型预训练的表示进行初始化，图像和本文特征提取器冻结，而特定视频时空自适应通过可学习的*adapter* 块建模。文章指出，**这种方法旨在保留基础模型的细微表征，增强我们的模型有效推广到新样本的能力**。

<mark>总的来说，就是在视频的空间定位上是参考基于图像的空间定位模型。</mark>

原文将文章贡献总结为了以下几部分：

1. 首先，在开放词汇设置中以零样本方式在 *HCSTVG V1* 和 *YouCook-Interactions* 基准上评估时空视频定位模型。模型表现分别比最先进的方法 *TubeDETR*  和 *STCAT* 高出 *4.26 m vIoU* 和 1.83% 的准确率。
2. 通过将空间定位模型的优势与互补的视频专用适配器相结合，文章方法在四个基准（即 *VidSTG*（声明性）、*VidSTG*（疑问性）、*HC-STVG V1* 和 *HC-STVG V2* ）的封闭集设置中始终优于之前的最新技术。

与目前SOTA方法性能对比![模型对比](https://pic.imgdb.cn/item/675e6df5d0e0a243d4e41976.png)

# 3.实现流程

首先明确这篇文章的任务，模型的输入是文本和一段视频序列，模型的输出是检测目标的*bounding box* $\left(x_{i}^{t}, y_{i}^{t}, w_{i}^{t}, h_{i}^{t}\right)$和时间间隔$\left(t_{s}, t_{e}\right)$；简单来说，这个任务就是结合时序检测和空间定位。

文章提出模型的总体架构如下所示，需要注意的是，根据文章描述，跨模态时空编码器这个模块下面的视觉-文本交叉注意力和文本-视觉交叉注意力我认为应该是一个并行结构，而不是图中画的串行的结构；其次，图中的跨模态时空解码器的视觉和文本交叉注意力上面的键和值的输入写反了，在视觉交叉注意力这个地方按照文章输入应该是$F_{v}^{M}$，对应地，文本交叉注意力这个地方按照文章输入应该是$F_{p}^{M}$。![总体架构](https://pic.imgdb.cn/item/675fc376d0e0a243d4e48297.png)

根据总体结构图，文章模型设计的实现流程大致如下（根据文章所述，文章所提方法是基于*DINO* 实现的，*DINO* 是一种基于*DETR* 的目标检测*SOTA* 方法；同时借鉴*GLIP* 和*Grounding DINO* 的文本对齐和定位概念）：

1. 使用backbone视觉编码器（*Swin Transformer* ）和文本编码器（*BERT* ）提取初始特征，注意在这一步，和其他基于DETR的检测器方法一样，图像特征使用多个视觉编码器从多尺度提取；
2. 在跨模态时空编码器建模帧间和帧内特征，并且学习跨模态时空关系；
3. 第二步得到的已经丰富跨模态特征的结果，被用于初始化每帧的查询，就是图中的$\left \{ Q_{t}^{0}  \right \}_{t=1}^{T}$；
4. 这个查询然后被解码来预测每帧的*bounding boxes* 和时间定位开始/结束帧。
5. 最后经过*BBox Head* 和*Temporal Head* 得到最后的结果，这两个模块是三层的多层感知机。

# 4.实现细节

* **跨模态时空编码器，*Cross-Modality Spatio-Temporal Encoder***：文章认为前面编码器得到的视觉和文本特征既不包含跨模态信息，也没有建模跨帧的时间关系，因此提出了该模块来解决这两个问题。
  **1）** 在每一层的开始，会对视觉特征$F_{v}$使用一个沿着时间维度的多头自注意力和沿着空间维度的多头自注意力，从而建模帧内关系和跨帧的时序关系，解决了第二个问题；类似地，对$F_{p}$使用多头自注意力。这一步可以用下面的公式表示，其中$F_ {v}^ {m-1}$是上一层的输出。
  
  $$
  F_ {v}^ {m\prime} = DA_ {spatial}^ {m} ( MHSA_ {temporal}^ {m} ( F_ {v}^ {m-1} )),
  $$
  
  $$
  F_ {p}^ {m\prime} = MHSA_ {p}^ {m} ( F_ {p}^ {m-1} ),
  $$
  
  **2）** 使用1）的输出，计算视觉-文本联合注意力$Attn_ {joint}^ {m}$，其计算公式如下；最后经过前馈神经网络FNN，得到最后的输出。从这里的公式来看，我认为结合下面公式1&2或者1&3才算一个完整的交叉注意力计算，所以感觉这个地方的公式和架构图描述的并不是一样的。
  
  $$
  \mathrm{Attn}_{\mathrm{joint}}^\mathrm{m}=\left(\frac{proj_{q,v}^m(F_v^{m\prime})proj_{q,p}^m(F_p^{m\prime})^T}{\sqrt{d^k}}\right),
  $$
  
  $$
  F_{v}^{m}=\mathrm{FFN}_{v}^{m}(\mathrm{softmax}(\mathrm{Attn}_{\mathrm{joint}}^{\mathrm{m}})proj_{p}^{m}(F_{p}^{m\prime}))),
  $$
  
  $$
  F_{p}^{m}=\mathrm{FFN}_{p}^{m}(\mathrm{softmax}(\mathrm{Attn}_{\mathrm{joint}}^{\mathrm{m}}{}^{T})proj_{v}^{m}(F_{v}^{m\prime}))),
  $$
* **语言引导的查询选择，*Language-Guided Query Selection***：这个模块的设计目的是，选择与输入文本更相关的特征作为解码器查询输入，从而实现有效的视觉文本融合。文章描述的是在查询的位置编码部分加入正弦时序位置，作者认为这可以增加帧序列重要的上下文信息，从而提高时序相关和定位。简单来说，<mark>该模块通过选定的索引和动态锚框的组合来初始化解码器查询。</mark>查询的内容部分在训练过程中是可学习的，而位置部分则使用动态锚框进行计算，这些锚框是通过编码器输出初始化的。文章还在查询的位置信息部分添加了一个正弦时序位置编码。
* **跨模态时空解码器，*Cross-Modality Spatio-Temporal Decoder***：与跨模态编码器类似，该模块也具有*N* 层。这个部分也比较简单，结合架构图和公式就能理解，从公式描述来看，和结构图存在一定不一致。
  
  $$
  Q_{t}^{n\prime}=\mathrm{MHSA}_{spatial}^n(\mathrm{MHSA}_{temporal}^n(Q_t^{n-1})),
  $$
  
  $$
  Q_{t}^{n}=\mathrm{FFN}^n(\mathrm{CA}_p^n(\mathrm{CA}_v^n(Q_t^{n\prime},F_v^M),F_p^M)),
  $$
  
  $$
  \mathrm{CA}_{v}^{n}(Q_{t}^{n},F_{v}^{M})=\left(\frac{proj_{q,v}^n(Q_t^{n\prime})proj_{k,v}^n(F_v^M)^T}{\sqrt{d^k}}proj_v^n(F_v^M)^T\right),
  $$

$$
\mathrm{CA}_{p}^{n}(\mathrm{CA}_{v}^{n},F_{p}^{M})=\left(\frac{proj_{q,p}^n(\mathrm{CA}_v^n)proj_{k,p}^n(F_p^M)^T}{\sqrt{d^k}}proj_p^n(F_p^M)^T\right)
$$

* **损失函数**：根据架构图，损失函数包含两部分，分别是空间和时间上的损失计算。公式中的$\mathcal{L}_{GIoU}$是广义交并比，是*IoU* 的扩展，考虑了没有重叠时的空间关系，

$$
\mathcal{L}_{spatial}=\lambda_{L_{1}}\mathcal{L}_{L_{1}}(\hat{B},B)+\lambda_{GIoU}\mathcal{L}_{GIoU}(\hat{B},B).
$$

$$
\mathcal{L}_{temporal}=\mathcal{L}_{KL}^s(\hat{\pi_s},\pi_s)+\mathcal{L}_{KL}^e(\hat{\pi_e},\pi_e),
$$

# 5.模型性能

文章说明在训练和推理阶段，采样了128帧；并且改变分辨率大小为448×448，架构图中的M和N都设置大小为6，在训练阶段*batch size* 设置为8，*learning rate* 设置为$ 1e^ {-4}$。

文章在进行实验的时候分为了开放词汇实验和闭集监督实验。

* **开放词汇实验**：文章方法在*VidSTG* 数据集上进行训练，在*HC-STVG V1* 和*YouCook-Interactions* 数据集上评估；其中，前者的数据分布与训练数据集存在的差异较小，后者与训练数据集存在的差异更大。实验结果如下![开放词汇实验结果](https://pic.imgdb.cn/item/67611436d0e0a243d4e5166e.png)
* **闭集监督评估**：这种就是正常的设计，有训练集、验证集和测试集。*VidSTG* 数据集实验结果：![VidSTG数据集实验结果](https://pic.imgdb.cn/item/6761148dd0e0a243d4e5168f.png)*HC-STVG V1*和*YouCook-Interactions* 数据集实验结果：![HC-STVG V1和YouCook-Interactions数据集实验结果](https://pic.imgdb.cn/item/676114d2d0e0a243d4e516b3.png)
* 消融实验：最后文章也进行了消融实验，实验设置如下：![消融实验](https://pic.imgdb.cn/item/67623258d0e0a243d4e59972.png)

# 6.改进/挑战/问题

总的来说，这篇文章的主要创新点就是将目标检测方向的方法和空间定位方向的方法迁移到了时空定位任务上。根据文章的描述，跨模态时空编码器参考的GLIP；语言引导的查询选择模块参考DETR/DINO；跨模态时空解码器没有提到参考哪个方法，但是大差不差应该也差不多参考的（后来去阅读了Grounding DINO，发现也是参考的这篇文章）。

* 改进：根据之前看过的文章，再改进的话，可以尝试修改提取特征的编码器部分。
* 问题：在消融实验上，文章提到了对编码器和解码器的空间模块进行了微调，但是不太清楚这个微调是怎么设置的，文章并没有提到，并且原文没有提供代码。

