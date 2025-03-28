---
layout: post
title: 'Few-Shot Common Action Localization  via Cross-Attentional Fusion of Context and Temporal Dynamics ICCV 2023'
subtitle: '通过上下文和时间的动态交叉注意力融合实现少样本普通动作定位'
date: 2025-03-23
author: Sun
cover: 'https://pic1.imgdb.cn/item/67de1d1f88c538a9b5c32a06.png'
tags: 论文阅读
---

> [Few-Shot Common Action Localization  via Cross-Attentional Fusion of Context and Temporal Dynamics](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Few-Shot_Common_Action_Localization_via_Cross-Attentional_Fusion_of_Context_and_ICCV_2023_paper.pdf)
> ❌❌不提供代码
> 
> 📌作者单位
> 1.Qualcomm AI Research-高通人工智能研究部门

# 1.文章针对痛点

这篇文章关注的是少样本时间定位问题。少样本的话解决的问题就是数据集的标注需要大量的人力和时间来注释。

因此，这篇文章的工作是**基于少量的描述普通动作实例的剪裁支持视频**，来时间定位长未剪裁查询视频中的动作实例。文章认为这个任务最重要的就是查询视频和支持视频之间的对齐，而为了**更好的对齐**，文章将这个问题分成了两点：<mark>在查询视频上下文下，重新校准支持视频特征；以及增强校准后支持视频特征的多样性和兼容性。</mark>

但是现有的方法大多是关注上面两点的第一点，并且经常是将多个支持视频作为整个视频来处理。但是文章认为虽然支持视频表示的是同一个普通动作，但是他们的上下文是不同的；因此，容易导致过度抑制支持视频的有用片段。

# 2.主要贡献

所以针对上面的问题，文章认为第一点应该是单独处理每个支持视频。而针对第二点，文章从多时间粒度关注时间多样性；文章尝试从不同的时间粒度协作融合支持视频特征，同时也考虑不同支持视频的兼容性。

对于时间对齐，文章提出了三阶段的交叉注意力机制。**第一阶段**是查询到支持上下文交叉注意力，该阶段会架构支持视频转化到查询视频的上下文中，通过和查询提案特征的交叉联系。**第二阶段**是多样性交叉注意力，低时间粒度特征汇总每个支持视频的片段组，并且加入了细粒度片段级特征（没太明白什么意思）。**最后一个阶段**是支持到查询的上下文交叉注意力，查询视频特征将会被同时添加到所有的生成视频特征中（也没太懂）。

此外，文章也提出了一个相关分类器。

原文总结贡献如下:

1. 我们建议采用三阶段*CA* 来通过支持视频增强查询视频的表示，反之亦然。
2. 我们单独处理每个支持视频，以提高其在前两个阶段的辨别能力。
3. 我们开发了一个关系分类器，包括一个动作分类器和一个辅助关系模块。后者仅在训练期间需要。
4. 对两个基准数据集进行了广泛的实验分析，我们在其中实现了*SOTA* 性能。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67de264d88c538a9b5c32d31.png)

原文也展示了一个简要版的架构图：![简要版的架构图](https://pic1.imgdb.cn/item/67de263f88c538a9b5c32d25.png)
整体来说，就是先使用预训练的backbone来提取查询视频和支持视频特征，之后针对查询视频会生成多个动作提案，而对于支持视频，也会先将每个视频分成多个片段；之后再对齐查询视频和支持视频。

# 4.实现细节

* **三阶段交叉注意力机制** ：三阶段大概是什么内容前面已经阐述了，下面就详细介绍三阶段分别怎么实现的。1️⃣*QtoS Context-CA* ：在这一阶段主要是考虑查询视频的上下文完成对支持视频的增强。具体来说，文章将每一个支持视频都与查询视频计算可学习权重矩阵，并计算交叉相关性，可用公式1表示；之后对得到的$$ \Lambda_{Q\to S}^{(l)}$$逐行求*softmax* 得到$$A_{Q\to S}^{(l)}$$。最后进行公式2的转化，得到该阶段最终的增强特征。2️⃣*Dynamics-CA*：这个模块的作用主要是在上一步的输出上，可以获取时间多粒度信息。具体来说首先是对上一步的每一个输出信息$$X_{S}^{(l)}$$使用了一维时间卷积，从而获得多时间粒度信息，如公式3所示。之后计算多粒度时间信息特征与上一步输出信息之间的交叉关系，与第一部类似，如公式4所示。最后得到动态的支持视频特征，如下公式5。3️⃣*StoQ Context-CA*：这个也是和前面类似，先计算交叉注意力，之后就是相乘残差操作如下公式6，最后得到查询提案$$\tilde{X}_Q$$。
  
  $$
  \Lambda_{Q\to S}^{(l)}=X_{Q}^{T}W_{Q\to S}X_{S}^{(l)} \quad(1)
  $$
  
  $$
  \tilde{X}_{S}^{(l)}=X_{Q}A_{Q\to S}^{(l)}+X_{S}^{(l)}. \quad(2)
  $$
  
  $$
  \tilde{X}_{S^{\prime}}^{(l)}=\tilde{X}_S^{(l)}\odot\mathbf{w}_k \quad(3)
  $$
  
  $$
  \Lambda_{S^{\prime}\to S}^{(l)}=\tilde{X}_{S^{\prime}}^{T}W_{S^{\prime}\to S}\tilde{X}_{S}^{(l)} \quad(4)
  $$
  
  $$
  \tilde{Y}_{S}^{(l)}=\tilde{X}_{S^{\prime}}A_{S^{\prime}\to S}^{(l)}+\tilde{X}_{S}^{\prime(l)},  \quad(5)
  $$
  
  $$
  \tilde{X}_Q=\tilde{Y}_SA_{S\to Q}+X_Q  \quad(6)
  $$
* **相关分类器**：这个整体没怎么看明白，大概就是分类器分为了两个部分，一个是*Auxiliary relational module*，以及一个动作分类器。


# 5.模型性能

实验性能效果对比图，在少样本的情况下，感觉这个性能不错了。![效果对比图1](https://pic1.imgdb.cn/item/67e2113d0ba3d5a1d7e2d21c.png)


# 6.改进/挑战/问题/想法

* **想法**：我感觉少样本这个效果还是很好的，少样本设置下的，还能进行相对准确的时间定位。但是我感觉我对文章为什么会有这样的效果还不是很明白。
* **问题**：这篇文章其实也没怎么看懂，我觉得这种架构图都画的很抽象，怎么看不懂呢 啊啊啊啊。还有一个问题就是一般来说，模型都是在*ActivityNet1.3* 表现较差，而在*THUMOS14* 上表现较好，但是这里刚好反过来，但是不知道是什么原因（文章提到的是，后者的动作实例更短）

