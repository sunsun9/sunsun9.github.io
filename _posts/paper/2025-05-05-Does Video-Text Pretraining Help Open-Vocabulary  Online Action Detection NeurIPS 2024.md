---
layout: post
title: 'Does Video-Text Pretraining Help Open-Vocabulary  Online Action Detection? NeurIPS 2024'
subtitle: '视频-文本预训练是否可以帮助开放词汇设置下的在线动作检测'
date: 2025-05-05
author: Sun
cover: 'https://pic1.imgdb.cn/item/6816d8bc58cb8da5c8da38f8.png'
tags: 论文阅读
---

> [Does Video-Text Pretraining Help Open-Vocabulary  Online Action Detection?](https://proceedings.neurips.cc/paper_files/paper/2024/file/5598cf1b2905a26ddb863e6705588327-Paper-Conference.pdf)

> 💐💐[提供代码](https://github.com/OpenGVLab/OV-OAD)
> 
> 📌作者单位
> 
> 1. 同济大学
> 2. 上海人工智能实验室
> 3. 南京大学
> 4. 复旦大学

# 1.文章针对痛点

这篇文章关注的问题是在线动作检测，并且是在开放集和零样本设置下的，也就是关注的开放词汇零样本的在线动作检测。简单来说，该任务并不是不训练，而是仅在文本监督下训练，另外需要能够识别之前没见过的类别。

文章之所以提出关注这个问题，首先是目前大部分动作检测方法是离线的，并且是在闭集设置下分类和定位长未剪裁视频中的动作（一些预定义的类别）；但是这种设置会限制模型动作检测的能力，并且是需要所有动作类别的手工注释。

在文章所提出的这种任务设置下，也有一些问题。首先在线动作检测使用的滑动窗口帧采样经常会导致背景帧占比大，而这与*VLM* 训练所需的低背景信息假设是矛盾的；其次，*OAD*（在线动作检测）不能在训练阶段未来获取帧，导致在同一个*batch* 中很难采样所有类别标签，这不利于图像文本对比损失的优化。

# 2.主要贡献

所以针对上面的这些问题，文章提出引入以目标为中心的解码器单元到基于*Transformer* 的模型中，能够在文本监督下自动聚合具有相似语义的帧。

文章提出在视频-文本数据集上进行预训练，不使用任何帧级别的手工注释信息。并<mark>提出了三种代理任务，分别是当前帧-文本嵌入空间对齐，背景帧掩码预测以及多标签视频-文本嵌入空间对齐。第一项任务可使模型优先处理邻近帧的判别信息。第二项任务是成功检测自然视频中的复杂背景帧。第三个任务是减轻网络视频中字幕噪声的影响。</mark>

原文总结贡献如下：

1. 我们研究了如何利用预先训练好的视觉语言模型在未剪辑视频中进行零镜头在线动作检测这一关键问题。
2. 我们引入了一种新颖的视频-文本双编码器架构，即 *OV-OAD*，来执行开放词汇在线动作检测。在下游数据集上的实验表明，我们的模型成功地学习了相似视频帧集群，并以零镜头的方式将它们转移到多个动作语义词汇表中。
3. 据我们所知，我们的工作是首次在不依赖任何精确的帧尺度标签的情况下，探索从文本监督到在线动作检测任务的零点转移。而且，我们还为这一新设置建立了稳健的基线。

# 3.实现流程

下图分别展示了闭集与文章所提出的开放集的区别以及文章的架构图：
![对比图](https://pic1.imgdb.cn/item/68170b4f58cb8da5c8da9c66.png)
![模型架构图](https://pic1.imgdb.cn/item/68170b8a58cb8da5c8da9c6a.png)
根据文章的描述，文章所提出模型架构包含两部分，分别是视觉编码器和文本编码器（文章使用的是*CLIP* 的文本编码器），上面所示的架构图，大部分内容都是视觉编码器的内容。

视觉编码器包含两个部分，分别是*distant neighboring-frame transformer block* 和*action clustering block*；前者在是架构图浅灰的区域，后者是架构图中的浅黄区域。

# 4.实现细节

* ***distant neighboring-frame transformer block*（远近帧transformer块）**：该模块是视觉编码器的一部分，也就是处理视频帧，得到当前帧的邻居帧和远距离帧的结合版。具体来说，利用相邻帧作为查询，从遥远过去的帧中提取信息。主要由四层标准变压器解码器单元组成。视频中的每一帧都添加了一维绝对位置编码，独立应用于过去帧和邻近帧。只有相邻帧才加入了方向注意掩码，确保信息单向流向当前帧。直观地说，最后得到的当前帧嵌入$$\mathcal{V}_{N}^{t^{\prime}}[-1]$$具有更多的空间信息，将与作为原始视频片段表示的相应文本对齐。
* ***Action Clustering*（动作聚类块）** ：该模块也是视觉编码器的一部分。动作聚类模块将帧组合成组，并以数据驱动的方式将这些组与人类可理解的类别进行对齐，只受视频文本对的监督。它由三个步骤组成，分别是帧到组的绑定（将具有相似语义的静态帧分配到一个组中）、组到动作的映射（计算组嵌入与动作标签之间的余弦相似度），以及背景掩码预测（通过计算帧到组相似性矩阵 A 的统计数据来预测一组二进制掩码）。<mark>在训练阶段，</mark>文章设计了一个二元预测任务，将帧聚类为背景、非背景以及分组。参考下列公式（1），其中$$\mathcal{M}_{B}$$是背景掩码。<mark>在评估阶段，</mark>视频片段的动作预测得分即 $$ \mathcal{P}_{AC}$$的计算公式如下2。
  
  $$
  \mathcal{A}:=\operatorname{softmax}\left(\frac{K @ Q^{\top}}{\sqrt{d}}\right), \quad \mathcal{M}_{B}:=\mathcal{A} \cdot \mathcal{G} \cdot \mathcal{T}^{\top} \quad（1）
  $$
  
  $$
  \mathcal{P}_{AC}:=\mathrm{one-hot}(\underset{k}{\operatorname*{\operatorname*{\arg\max}}}(\mathcal{A}))\cdot\mathcal{G}\cdot\mathcal{T}_{val}^\top.	\quad(2)
  $$
* ***Object-Centric Decoder Unit*** ：该模块是前面动作聚类模块的一个单元。具体来说，为了对帧进行分组，文章提出了一个以对象为中心（*OC* ）的解码器单元，该单元在交叉注意计算过程中采用了以查询对象为中心的插槽注意机制。它的工作原理与之前提出的 *GroupViT* 和 *OVSegmentor* 方法类似，后者是专门为语义分割任务设计的。该部分用文章提供的公式表示如下，但是我觉得这个符号与文章架构图使用的符号稍微有些区别，不知道这个模块数据流到底怎么流动的。

$$
\mathcal{G}^{\prime}:=\operatorname{softmax}\left(\frac{\mathcal{G} \cdot \mathcal{G}^{\top}}{\sqrt{d}}\right) \cdot \mathcal{G}, \\ \operatorname{Slot} \operatorname{Attn}\left(\theta\left(\mathcal{G}^{\prime}\right), \mathcal{V}_{N}^{t^{\prime}}\right):=\left[\operatorname{softmax}\left(\frac{\mathcal{V}_{N}^{t^{\prime}} \cdot \theta\left(\mathcal{G}^{\prime}\right)^{\top}}{\sqrt{d}}\right)\right]^{\top} \cdot \mathcal{V}_{N}^{t^{\prime}},
$$
# 5.模型性能

文章的对比损失计算如下公式所示，也是按照前面提到的三个代理任务产生了三部分损失。

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{contras}}+\alpha\mathcal{L}_{\mathrm{current}}+\beta\mathcal{L}_{\mathrm{mask}},
$$

$$
\mathcal{L}_{\mathrm{current}}=\mathcal{L}_{\mathrm{ITC}}(z^I,z^T)=-\frac{1}{2}\left(log\frac{\exp(z_i^I\cdot z_i^T/\tau)}{\sum_j^B\exp(z_i^I\cdot z_j^T/\tau)}+log\frac{\exp(z_i^T\cdot z_i^I/\tau)}{\sum_j^B\exp(z_i^T\cdot z_j^I/\tau)}\right),
$$

$$
\mathcal{L}_{contras}=\mathcal{L}_{ITC}(z^V,z_0^T)-\frac{1}{2}\left(log\frac{\sum_m^Mexp(z_i^I\cdot z_i^{T_m}/\tau)}{\sum_m^M\sum_j^Bexp(z_i^I\cdot z_j^{T_m}/\tau)}+\frac{1}{M}\sum_{m=1}^Mlog\frac{exp(z_i^{T_m}\cdot z_i^I/\tau)}{\sum_j^Bexp(z_i^{T_m}\cdot z_j^I/\tau)}\right),
$$

$$
\mathcal{L}_{mask}=\mathcal{L}_{dice}(\mathcal{M}_{B},\mathcal{M}_{GT})
$$

实验性能效果对比图：![效果对比图1](https://pic1.imgdb.cn/item/6818230858cb8da5c8dc796c.png)

# 6.改进/挑战/问题/想法

* **想法**：感觉这篇文章的这种设计还是挺复杂的，并且设计的层其实都挺深的。*batch_size* 还256，这根本跑不起来，这种模型根本不适合，太大了，数据量也挺大的。另外，这篇文章使用的是*CLIP*，在文章的描述中，没有怎么看到对时序信息的处理，不知道对时序信息是怎么处理的；如果没有相应的处理话，未来可以考虑加入时序信息的处理。

