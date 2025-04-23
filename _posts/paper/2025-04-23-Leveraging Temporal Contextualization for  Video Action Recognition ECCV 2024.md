---
layout: post
title: 'Leveraging Temporal Contextualization for  Video Action Recognition ECCV 2024😊'
subtitle: '利用时间情景用于视频动作识别'
date: 2025-04-23
author: Sun
cover: 'https://pic1.imgdb.cn/item/6807042f58cb8da5c8bdcfd0.png'
tags: 论文阅读
---

> [Leveraging Temporal Contextualization for  Video Action Recognition](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03099.pdf)

> 💐💐[提供代码](https://github.com/naver-ai/tc-clip)
> 
> 📌作者单位
> 
> 1. 首尔国立大学电气与计算机工程系
> 2. 首尔国立大学智能与感知人工智能研究所
> 3. NAVER人工智能实验室

# 1.文章针对痛点

这篇文章关注的问题是视频动作识别，虽然叫做视频动作识别，但是从论文来看，仍然是处理文本和视频，获取与文本相关的片段。

文章认为现有的方法不能在视频特征学习过程中建模时间信息，认为这种限制源自*tokens* 在时间轴上交互限制。文章列举了现有方法的一些问题，如下图。

为确保*patch tokens* 在时空域中的全局交互，一种可能的选择是在编码过程中将所有帧中的每个*patch tokens* 作为参考。遗憾的是，在使用短的图像-文本对进行预训练的 *VLM* 中，对时间上的全局交互进行这样直接的扩展会面临外推法的挑战。

![问题](https://pic1.imgdb.cn/item/6807067e58cb8da5c8bddb2b.png)

# 2.主要贡献

所以针对上面的内容，文章提出了一种扩展*CLIP* 到视频的新范式，*Temporally Contextualized CLIP (TC-CLIP)*，通过利用先进的时间分析编码整体视频信息。

具体来说，文章的 “时间语境化”（*Temporal Contextualization，TC* ）管道将全局动作线索归纳为一小组标记，称为 “语境标记”，供编码过程中参考。这些上下文标记可作为注意力操作的附加键值对，大概可作为传达视频级上下文的时间桥梁。

此外，视频条件提示（*VP* ）模块会根据来自视觉编码器的上下文标记生成实例级文本提示。*VP* 模块包括交叉注意操作，采用可学习文本提示作为查询，上下文标记作为键和值，将视频实例表示注入视频条件文本提示。这一策略弥补了动作识别数据集中文本语义的不足。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：
![模型架构图2](https://pic1.imgdb.cn/item/680721b458cb8da5c8be5c23.png)

文章提出的这个架构包括三个步骤： 1) 在每个帧中选择信息标记；2) 跨时空维度的上下文总结；3) 将上下文注入后续层中的所有标记。

# 4.实现细节

* ***Informative token selection***：这个模块就是前面提到的三个步骤的第一步，用于选择包含有效信息的*token*。由于视频中存在许多冗余*token*，因此要提取所需的时序信息，使用所有*token* 可能不是最佳选择。因此利用自注意力操作获得的每帧注意力分数来选择信息量大的种子标记。具体来说，利用公式1来计算注意力分数，其中$\mathbf{q}_{\mathrm{cls}}=\mathbf{z}_{\mathrm{t},0}\mathbf{W}_q$，$\mathbf{K}_{\mathbf{z}_t}=\mathbf{z}_t\mathbf{W}_k$。
  
  $$
  \mathbf{a}(\mathbf{z}_t)=\mathrm{Softmax}\left(\frac{\mathbf{q}_\mathrm{cls}\mathbf{K}_{\mathbf{z}_t}^\mathsf{T}}{\sqrt d}\right),	\quad(1)
  $$
* ***Temporal context summarization***：这个部分是根据各帧的相关性将种子标记连接起来，并识别出一系列上下文标记。具体来说，首先收集所有帧的种子标记$\left\{\hat{\mathbf{z}}_{t, i}\right\}_{(t, i) \in \mathcal{S}}$，其中，${\hat{\mathbf{z}}_{t, i}}$是${\mathbf{z}}_{t, i}$经过自注意力操作得到的。之后，按照下面公式1得到上下文*tokens*，这个操作就是聚类和合并操作。最后将得到的输出输入送入一个前馈神经网络层得到最后的时间上下文总结。
  
  $$
  \hat{\mathbf{s}}=\phi\left(\left\{\hat{\mathbf{z}}_{t, i}\right\}_{(t, i) \in \mathcal{S}}\right)	\quad(1)
  $$
* ***Temporal context infusion***：按照下公式1进行交叉注意力操作融合上下文信息。最后文章按照公式2 3构建了逐层的*TC* 管道。

$$
\mathrm{Attention}_{\mathrm{TC}}(\mathbf{z}_t,\mathbf{s})=\mathrm{Softmax}\left(\frac{\mathrm{Q}_{\mathbf{z}_t}\left[\mathrm{K}_{\mathbf{z}_t}|\mathrm{K}_\mathbf{s}\right]^\mathsf{T}}{\sqrt{d}}+\mathbf{B}\right)\left[\mathrm{V}_{\mathbf{z}_t}|\mathrm{V}_\mathbf{s}\right],	\quad(1)
$$

$$
\mathbf{B}_{ij}=
\begin{cases}
b_{\mathrm{local}} & \mathrm{~if~}j\leq N+1 \\
b_{\mathrm{global}} & \text{ otherwise,} & & & 
\end{cases}	\quad(2)
$$

$$
\begin{aligned}
 & \hat{\mathbf{z}}_t^l=
\begin{cases}
\mathrm{MHSA}(\mathrm{LN}(\mathbf{z}_t^{l-1}))+\mathbf{z}_t^{l-1} & \mathrm{if~}l=1 \\
\mathrm{MHSA}_\mathrm{TC}(\mathrm{LN}(\mathbf{z}_t^{l-1}),\mathrm{LN}(\mathbf{s}^{l-1}))+\mathbf{z}_t^{l-1} & \text{otherwise,} & 
\end{cases} \\
 & \mathbf{z}_t^l=\mathrm{FFN}(\mathrm{LN}(\hat{\mathbf{z}}_t^l))+\hat{\mathbf{z}}_t^l, \\
 & s^{l}=\mathrm{FFN}(\mathrm{LN}(\hat{\mathbf{s}}^l))+\hat{\mathbf{s}}^l,
\end{aligned}	\quad(3)
$$

* ***Video-conditional Prompting***：这个模块没太看懂，文章提供了该模块的架构图如下。![模块架构图](https://pic1.imgdb.cn/item/6808550c58cb8da5c8c3c84a.png)

# 5.模型性能

文章最后使用到的损失公式如下所示：

$$
\mathcal{L}(\mathcal{P}, \mathcal{G})=\lambda_{\text {cls }} \mathcal{L}_{\text {cls }}(\mathcal{P}, \mathcal{G})+\lambda_{\text {dice }} \mathcal{L}_{\text {dice }}(\mathcal{P}, \mathcal{G})+\lambda_{\text {bce }} \mathcal{L}_{\text {bce }}(\mathcal{P}, \mathcal{G})
$$

实验性能效果对比图：![效果对比图1](https://pic1.imgdb.cn/item/680854a758cb8da5c8c3c37b.png)
![效果对比图2](https://pic1.imgdb.cn/item/6808554558cb8da5c8c3cb0f.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章没太看懂，但是大概意思就是不能选择所有的*tokens*，所以文章就是一个选择*tokens*，总结*tokens* 的过程。

