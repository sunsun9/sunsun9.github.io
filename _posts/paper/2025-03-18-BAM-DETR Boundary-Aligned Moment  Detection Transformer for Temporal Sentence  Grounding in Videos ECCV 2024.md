---
layout: post
title: 'BAM-DETR: Boundary-Aligned Moment  Detection Transformer for Temporal Sentence  Grounding in Videos ECCV 2024😊'
subtitle: 'BAM-DETR：用于视频时间句子定位的边界对齐时刻定位Transformer'
date: 2025-03-18
author: Sun
cover: 'https://pic1.imgdb.cn/item/67d7abaa88c538a9b5bfb91f.png'
tags: 论文阅读
---

> [BAM-DETR: Boundary-Aligned Moment  Detection Transformer for Temporal Sentence  Grounding in Videos](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00212.pdf)
> 💐💐提供代码
> 
> 📌作者单位
> 1.仁荷大学人工智能系
> 2.延世大学计算机科学系

# 1.文章针对痛点

这篇文章关注的是视频时间句子定位任务。

现有的方法通常采用*DETR-like* 方法，这种方法通常是检测视频中心和长度来定位视频片段。但是实际上，视频片段的中心可能不是最能代表所给句子的表示。因此，**这种模糊性给模型准确定位中心的能力带来了挑战，和错误对齐的中心，从而导致低质量的预测**。（也就是说，模型找到的中心可能是与给定句子的相关程度较高，但是这个中心并不是*ground-truth* 片段的中心）文章针对这个问题给出了示例，如下图；（a）表示的这个示例就是上面括号里所说的意思。![示例](https://pic1.imgdb.cn/item/67d7ad5588c538a9b5bfb9c0.png)

此外，文章还发现了**传统评分方法的问题**，其中二元分类（或匹配）分数用于提案排名。文章认为这回导致定位任务的次优结果，因为分数矩可能与给定句子具有很高的匹配分数，但是定位不一定是准确的。

# 2.主要贡献

因此，为了解决上面的问题，文章首先为时刻提出了一种新的面向边界的公式，也就是每个时刻表示为一个三元组，包含锚点和锚点与边界的距离，即（$$p$$, $$d_s$$, $$d_e$$），这种不对称性可以有效解决上述的预测中心不是*ground-truth* 片段中心的问题。

基于这种时刻建模，文章提出了一种新的架构，具有专用的解码器设计，称为*Boundary-Aligned Mo ment Detection Transformer (BAM-DETR)*。模型采用**双路径解码管道来预测锚点和边界**，也就是利用两种不同类型的查询分别实现锚点和边界细化，前者聚合全局信息，后者使用提出的边界焦点注意集中边界的稀疏局部邻域。

为了解决上述的分数排名问题，文章**提出了基于定位质量来排序提案**。也就是通过放弃分类分数角色，修改基于查询模型的经典匹配函数和训练目标为面向定位。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67d7ec1d88c538a9b5bfeb96.png)

这个采用的就是*DETR-like* 结构，也就是编码器-解码器的架构。

# 4.实现细节

* **多模态编码器** ：这个**多模态编码器采用的是文本到视频的编码器，包含交叉注意力模块和自注意力模块**。文本和视频分别经过单模态编码器编码后，投影到同一个共享空间，再输入到多模态编码器中。1️⃣**交叉注意力模块**的公式如下1 2所示，最后经过整个多头交叉注意力块后，得到的多模态特征表示为$$\tilde{V}$$。2️⃣**自注意力块**用于增强特征，允许片段内交互。公式也是类似于交叉注意力模式，只不过输入可能有所不同。3️⃣后面还提出了一个**显著性引导**，这个部分没太看明白，文章说是为了让模型更好的理解视频和文本之间的语义关系，但是它的实现方式也没有具体说明，只是说使用了一个显著性预测器，给了一个损失，这个损失我也没理解为什么这个设置。

$$
\begin{aligned}
 & \mathcal{V}^{\prime}=\mathrm{softmax}\left(\frac{\mathbf{Q}_{\mathcal{V}}\mathbf{K}_{\mathcal{T}}^{\top}}{\sqrt{D}}\right)\mathbf{V}_{\mathcal{T}}+\mathcal{V}, \quad(1)\\
 & \mathcal{V}^{\prime\prime}=\mathrm{FFN}(\mathcal{V}^{\prime})+\mathcal{V}^{\prime},\quad(2)
\end{aligned}
$$

* **双路径解码器**：因为文章提出了一种新的编码片段的方式，不同于常用的预测中心的方式，称为面向边界的编码。所以文章提出了两种路径并行进行，**分别是锚点更新路径和边界更新路径**。1️⃣锚点更新路径：锚点也就是模型找到的认为与文本语义相关度高的时刻。这条路径的目标就是适应锚点位置，方便边界可以基于此预测。具体来说，锚点更新路径包含自注意力层、交叉注意力层和前馈神经网络。**在自注意力层**，也就是将锚点查询正常投影，从而实现自注意力，公式如下1，需要说明的是，因为锚点查询本省不具有位置编码，因此文章在查询基础上增加了位置编码。**在交叉注意力层**，该层的目的是为了聚合多模态特征，也就是编码器最后输出的特征。这个公式如下2，这里需要特别说明的是，使用的是连接操作而不是加运算；公式中的$$p^l$$表示的是解码器上一层输出预测结果中的预测锚点位置。2️⃣边界更新路径：这个过程文章展示了一个架构图，如下图。根据图所示的流程，<mark>先是</mark>对编码器的输出特征，使用一维卷积操作，也就是在原来的基础上进一步叠加了一个激活函数变化得到$$\hat{V}_s$$，之后再与编码特征进行连接，就得到*locality-enhanced memory features*；在一维卷积这个地方提出了一个损失，如下3，但是这个损失的目的没太明白，其中$$\hat{g}^s=\max(\sigma(\hat{\mathcal{V}}_s))$$ 。<mark>之后</mark>，基于前面的预测锚点位置，以开始边界预测为例，利用新预测的锚点和上一层解码得到的距离，计算出开始时刻的位置，即$$p^{(l+1)}-d_s^l$$ 。使用可变性注意力预测新的开始边界，如下4（在此之后，还要使用一个全连接层）；结束边界位置预测类似。

$$
\tilde{\mathbf{C}}_p^l=\mathrm{softmax}\left(\frac{\left(\mathbf{Q}_{\mathbf{C}_p^l}+\mathbf{P}_{\mathbf{A}^l}\right)\left(\mathbf{K}_{\mathbf{C}_p^l}+\mathbf{P}_{\mathbf{A}^l}\right)^\top}{\sqrt{D}}\right)\mathbf{V}_{\mathbf{C}_p^l}+\mathbf{C}_p^l.\quad(1)
$$

$$
\hat{\mathbf{C}}_p^l=\mathrm{softmax}\Big(\frac{\left(\mathbf{Q}_{\tilde{\mathbf{C}}_p^l}\parallel\mathbf{P}_{\mathbf{p}^l}\right)\left(\mathbf{K}_{\hat{\mathcal{V}}}\parallel\mathbf{P}_{\hat{\mathcal{V}}}\right)\Big)^\top}{\sqrt{2D}}\Big)\mathbf{V}_{\hat{\mathcal{V}}}+\tilde{\mathbf{C}}_p^l. \quad(2)
$$

$$
\mathcal{L}_{\mathrm{regul}}^s=-\frac{1}{N_v}\sum_{i=1}^{N_v}\left(g_i^s\log(\hat{g}_i^s)+(1-g_i^s)\mathrm{log}(1-\hat{g}_i^s)\right),\quad(3)
$$

$$
\hat{\mathbf{C}}_s^l=\sum_{k=1}^K\left[\mathbf{w}_k^s\cdot\hat{\mathcal{V}}_s^{\prime}[\mathbf{p}^{(l+1)}-\mathbf{d}_s^l+\mathbf{o}_k^s]\right]+\mathbf{C}_s^l,\quad(4)
$$

![架构图](https://pic1.imgdb.cn/item/67d8f66d88c538a9b5c01e1a.png)

* **基于质量的分数**：这个就是说原来的方法都是基于分类的分数，但是文章认为这种方式对于定位会不准确，所以提出了基于定位质量的分数，计算公式如下1，在这一部分提出了一个损失函数，如下2。

$$
\begin{aligned}
\mathbf{q}=\sigma(\mathrm{MLP}([\mathbf{C}_p\parallel\mathbf{C}_s\parallel\mathbf{C}_e]))
\end{aligned} \quad(1)
$$

$$
\mathcal{L}_{\mathrm{qual}}=\sum_{m=1}^M\left|q_m-\max_{\forall n}(\frac{|\hat{\varphi}_m\cap\varphi_n|}{|\hat{\varphi}_m\cup\varphi_n|})\right|, \quad(2)
$$

# 5.模型性能

实验性能效果对比图，在弱监督的情况下，感觉这个性能其实也还行相比较全监督的话。![效果对比图1](https://pic1.imgdb.cn/item/67d8f6a588c538a9b5c01e20.png)
![效果对比图2](https://pic1.imgdb.cn/item/67d8f6ee88c538a9b5c01e26.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章总体来说就是提出了一种新的编码范式，之后就是围绕这种编码方式进行预测等等。也可以采用以下这种方式吧，至于文章的解码部分提到的一些损失，确实没理解这个损失的作用是什么

