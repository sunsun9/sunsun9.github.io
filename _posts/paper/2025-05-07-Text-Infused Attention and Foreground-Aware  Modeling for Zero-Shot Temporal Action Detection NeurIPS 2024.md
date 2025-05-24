---
layout: post
title: 'Text-Infused Attention and Foreground-Aware  Modeling for Zero-Shot Temporal Action Detection NeurIPS 2024😊'
subtitle: '面向零样本时序动作检测的注入文本注意和前景感知建模'
date: 2025-05-07
author: Sun
cover: 'https://pic1.imgdb.cn/item/6819756558cb8da5c8dea832.png'
tags: 论文阅读
---

> [Text-Infused Attention and Foreground-Aware  Modeling for Zero-Shot Temporal Action Detection](https://proceedings.neurips.cc/paper_files/paper/2024/file/13250eb13871b3c2c0a0667b54bad165-Paper-Conference.pdf)

> 💐💐[提供代码](https://github.com/YearangLee/Ti-FAD)
> 
> 📌作者单位
> 
> 1. 韩国高丽大学人工智能系

# 1.文章针对痛点

这篇文章关注的问题是零样本的时序动作检测。之所以关注这个任务，是文章认为传统的时序动作检测方法在注释长视频上是耗时的并且昂贵的，所以在真实场景下是不适用的。

而目前现有的大部分零样本时序动作检测方法通常应用基于前景的方法，也就是从视觉特征中初始化生成一些前景候选提案，之后与文本特征融合。但是文章认为这中方法仅仅融合文本与前景的视觉特征，所以会限**制视觉-文本模态全部信息的利用**。

所以文章提出了一种跨模态的方法，虽然文章使用的*baseline* 得到的效果是优于先前的方法的，<mark>但是文章发现了一个新的问题，观察到了一种普遍动作偏差问题，也就是分类分数趋向于捕捉*ground truth* 中的更普遍的子动作。</mark>

以上这两个问题，文章使用了相应的图示。**上面一行图**是用于辅助理解第一个问题，也就是先前的方法在利用视觉-文本信息上存在不充分问题，所以文章提出了第一行右边的跨模态方法。**下面一行图**就是文章给出的辅助理解第二个问题的图示，可以看到在处理未见类别时，模型更加关注*running* 这个动作，但是这个动作并不是识别*ground truth*动作的关键子动作，从而导致模型在识别未见类别动作时出错。![图示](https://pic1.imgdb.cn/item/681977d758cb8da5c8deb405.png)

# 2.主要贡献

所以针对上面的这些问题，文章提出了一个新的模型，即注入文本注意和前景感知的动作检测方法（*Ti-FAD* ）。这个方法能够为动作实例的准确识别捕捉判别性子动作。文章提出的这个方法包含两个重要的模块，分**别是文本注入交叉注意力和前景感知头**，前者能够让模型关注与文本描述最相关的判别性子动作，后者能够让模型从背景中准确识别动作片段。

原文总结贡献如下：

1. 我们构建了一个新颖的跨模态基线，在整个时间动作检测过程中整合了文本和视觉特征。
2. 我们提出了 *Ti-FAD*，它结合了 *TiCA* 和前景感知头，专注于具有区分性的子动作，尤其是未见动作类别。
3. 我们在 *THUMOS14* 和 *ActivityNet v1.3* 上进行了广泛的实验，结果表明我们的 *Ti-FAD* 在相当大的程度上优于最先进的方法。

# 3.实现流程

下图分别展示了跨模态基线架构图和文章提出的新的方法的架构图。可以看到在文章提出的新方法的架构图中是包含了文章所使用的跨模态基线。至于跨模态这部分就不在详细介绍了，就是先使用的多头自注意力，后面使用了多头交叉注意力，最后经过FNN处理，从而实现更新视觉和文本特征。后面详细介绍文章提出的新的架构核心模块。
![对比图](https://pic1.imgdb.cn/item/6819a7e458cb8da5c8df5e6a.png)
![模型架构图](https://pic1.imgdb.cn/item/6819a82758cb8da5c8df5ff4.png)

# 4.实现细节

* ***Text-infused Cross Attention (TiCA)***：该模块提出的目的是鼓励模型关注判别性子动作，该模块的流程主要包含以下几个部分：(1) 使用视频和文本特征生成分类得分图，从而表示与文本相关的视觉部分；(2) 选择前 k 个点以提取辨别点；(3) 根据动态阈值通过合并相邻索引重新定义索引；(4) 根据重新定义的索引生成高斯核，以生成突出注意力掩码（*SAM* ）；(5) 将生成的突出注意力掩码应用于交叉注意力。
  具体来说，1️⃣计算分类得分图。该步骤是通过计算*MHSA* 输出的两个中间特征的相似度获得的，如下公式1所示；得到的$$P_{s}=\left\{p_{t}\right\}_{t=1}^{T}$$提供了每个时间索引的分数，表明动作发生的可能性。2️⃣选择*top-k* 个索引来生成高斯内核。文章定义了$$I_{topk}$$作为$$P_{s}$$中与*top-k* 相关索引的中心位置候选，如下公式2所示。3️⃣重新定义索引。文章是基于一个动态距离阈值来更新索引，如下公式3所示，其中，$$\bar{\theta(T)}=\theta_{base}\cdot T/T_{init}$$。4️⃣之后，生成*SAM* 的掩码$$S_{mask}$$，如下公式4所示，其中，$$F(t)=\sum_{i\in I}G(t;i,\sigma_{i})$$。5️⃣最后直接将这个掩码用于多头交叉注意力中，如下公式5所示。
  
  $$
  P_{s}=\max _{C}\left({F^{\prime}}_{v i d}^{(l)} F_{t x t}^{\prime(l)}{ }^{\top}\right) .	\quad(1)
  $$
  
  $$
  I_{topk}=\{i\mid i\in\text{indices of top-}K\mathrm{~in~}P_s\}, 	\quad(2)
  $$
  
  $$
  I=\left\{
  \begin{array}
  {cc}\frac{i+j}{2} & \mathrm{if}|i-j|<\theta(T) \\
  i & \mathrm{otherwise}
  \end{array}\right.,\quad\forall(i,j)\in I_{topk}.	\quad(3)
  $$

$$
S_{mask}=\frac{F(t)}{\max_t(F(t))}\in\mathbb{R}^T.	\quad(4)
$$

$$
F^{\prime\prime}{}_{vid}^{(l)}=\mathrm{TiCA}(F^{\prime}{}_{vid}^{(l)},F^{\prime}{}_{txt}^{(l)})=\mathrm{Softmax}\left(\frac{\mathrm{Q}(F^{\prime\prime}{}_{vid}^{(l)})\mathrm{K}(F^{\prime}{}_{txt}^{(l)})^{\top}}{\sqrt{d}}\cdot(S_{mask}\otimes\mathbf{1}_{C})\right)\mathrm{V}(F^{\prime}{}_{txt}^{(l)}),(5)
$$

* ***Foreground-Aware Head*** ：该模块的目的就是抑制不相关的背景片段，同时增强模型对前景片段的关注。针对前景感知的衡量，文章设置了两个目标分数，分别是$$A_{\text {soft }}^{f g}$$和$$A_{\text {hard }}^{f g}$$，相应地，也设计了两个前景感知头，分别是*soft  foreground-aware head (S-FAD)* 和*hard foreground-aware head (H-FAD)*。1️⃣对于*S-FAD*，该感知头的目的是计算当前时间步与*ground-truth* 动作片段中心的归一化距离，具体如下公式1所示.2️⃣而*H-FAD* 更加直接，是一个二分类预测，也就是如果认为是前景片段就设为1，相反，则设为0。公式如下2所示。该部分的损失函数如下3所示。

$$
Y_{soft}=\min\left(\phi(l;0,\sigma),\phi(r;0,\sigma)\right),l=|c-s|\mathrm{and}r=|c-e|,	\quad(1)
$$

$$
Y_{hard}=\mathbb{1}_{[s\leq i\leq e]}.	\quad(2)
$$

$$
\mathcal{L}_{fg}=\alpha\mathcal{L}_{soft}(A_{soft}^{fg},Y_{soft})+\beta\mathcal{L}_{hard}(A_{hard}^{fg},Y_{hard}),	\quad(3)
$$

# 5.模型性能

实验性能效果对比图：![效果对比图1](https://pic1.imgdb.cn/item/681ac05558cb8da5c8e1a4bf.png)


# 6.改进/挑战/问题/想法

* **想法**：这篇文章其实提到的跨模态架构，这个其实并不是什么有新意的东西；重要的部分就是提到的*TiCA* 这个模块，也就是捕捉判别性动作。解决的这个问题也是很常见的问题，也就是在零样本的设置下，模型其实还是更倾向于识别在训练数据集中普遍出现的子动作，这样就会导致分类错误；所以最重要的还是找到与文本相关的判别性关键的动作。

