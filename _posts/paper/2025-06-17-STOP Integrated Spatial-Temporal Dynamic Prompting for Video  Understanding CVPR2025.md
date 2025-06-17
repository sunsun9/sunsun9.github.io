---
layout: post
title: 'STOP: Integrated Spatial-Temporal Dynamic Prompting for Video  Understanding CVPR2025😊'
subtitle: 'STOP: 面向视频理解的时空综合动态提示'
date: 2025-06-17
author: Sun
cover: 'https://pic1.imgdb.cn/item/684f871d58cb8da5c84e7032.png'
tags: 论文阅读
---

> [STOP: Integrated Spatial-Temporal Dynamic Prompting for Video  Understanding](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_STOP_Integrated_Spatial-Temporal_Dynamic_Prompting_for_Video_Understanding_CVPR_2025_paper.html)

> 💐💐[提供代码](https://github.com/zhoujiahuan1991/CVPR2025-STOP)
> 
> 📌作者单位
> 
> 1. 北京大学
> 2. 中国人民大学
> 3. 华中科技大学

# 1.文章针对痛点

这篇文章关注的是怎么帮助视频-语言模型更好地理解视频内容。目前由于像CLIP这样的预训练视觉-语言模型在图像领域的各种下游任务上良好的零样本泛化能力，研究人员也希望能够将这种能力应用到视频领域。但是相比于图像，要在视频领域应用这种范式是一个十分具有挑战性的任务。<mark>因为，要获取有标记的大视频数据集是十分困难的，其次训练在这种大视频数据集上训练计算成本是非常高的。</mark>

所以，为了解决上面提到的这种训练范式不可用的问题，不少研究人员转向在视频数据上训练大规模预训练的视频语言模型，期望能够让它们以较低成本解决下游任务。在这些解决方法中，大部分方法都是利用视觉提示或者视觉提示学习来解决；但是它们往往是针对整个视频的单个提示信息，<mark>这种模式容易忽视具有重要时间动态的关键帧，以及每帧内的判别性区域变化；从而导致视觉语言模型关注的视频帧和帧内区域的不准确性，进一步阻碍了模型理解视频内容。</mark>

# 2.主要贡献

所以针对上面的这个问题，文章提出了时空综合动态提示方法，Spatial TempOral dynamic Prompting(STOP)。具体而言，这个方法包含两个互补的模块，分别是帧内空间提示和帧间时间提示。

帧内空间提示采用轻量级三维卷积网络来捕捉视频不同区域的时间动态。通过将这些信息与帧内注意力权重相结合，可以在视频中识别出既包含单帧主要对象又包含动态时间信息的辨别区域。因此，文章设计了一个内部提示器，为这些区域生成空间提示，引导模型关注具有显著时间变化的区域，从而增强其捕捉视频数据中细粒度关键信息的能力。

帧间时间提示利用帧内空间提示识别视频帧内分辨区域的能力，在此基础上，进一步计算这些区域在帧间的变化程度。对于具有显著时间动态变化的关键帧，就使用轻量级提示器动态生成帧间提示，并插入这些提示以提供两帧之间的细粒度信息。

原文总结贡献如下：

1. 针对预先训练的视觉语言模型难以捕捉视频中的时间信息这一难题，我们提出了 STOP 方法。首先，我们设计了帧内空间提示，以突出视频帧中的判别区域，有效引导模型关注动态变化的区域。
2. 此外，我们还计算视频帧间的动态变化，并动态生成帧间时间提示。这些提示插入动态变化显著的帧之间，提供了精细的时间信息，便于模型关注和理解视频中的关键帧。
3. 在视频动作识别和视频文本检索的各种基准上进行的广泛实验表明，我们提出的 STOP 方法始终显著优于现有的视频提示方法

# 3.实现流程

下图展示了文章提出方法的大致流程图。文章使用的视频编码器和文本编码器都是预训练的。

![模型架构图](https://pic1.imgdb.cn/item/684fa62058cb8da5c84eebab.png)

# 4.实现细节

* ***Integrated Spatial-Temporal Dynamic Prompt ing*** ：整个模型的重点就是这一部分，这个是包含两个模块，分别是帧内空间提示和帧间时间提示。1️⃣帧内空间提示模块首先识别辨别区域$$r_{i}\in\mathbb{R}^{N_{p}}$$的位置，然后通过轻量级提示器 $$\mathcal{P}^{s}$$为帧$$F_i$$生成帧内空间提示 $${p_i^s}$$，生成帧内空间提示可以用以下公式1表示。2️⃣帧间时间提示模块参考下面。

$$
p_i^s=\mathcal{P}^s(h_{i-1},h_i,h_{i+1}), 	\quad(1)
$$

* **帧内空间提示** ：该模块的架构图在第三部分展示了。具体来说，前面提取的*Image token* 会分别送入一个注意力模块（公式1）和一个3D卷积模块（公式2），前者是为了捕捉不同*patch* 的差异，后者是为了捕捉时间动态。之后，concat这两部分内容（公式3），并根据公式4判断得到最终的判别性区域。最后，通过设计的轻量化提示器得到特征（公式5）。

$$
A_i=\operatorname{Attn}(\boldsymbol{h}_{cls},\boldsymbol{h}_i), \quad(1)
$$

$$
[\tilde{\boldsymbol{h}}_1,\tilde{\boldsymbol{h}}_2,\cdots,\tilde{\boldsymbol{h}}_{N_F}]=\mathcal{N}^s([\boldsymbol{h}_1,\boldsymbol{h}_2,\cdots,\boldsymbol{h}_{N_F}])\\
M_{i,j}=\frac{1}{d_v}\sum_k\tilde{\boldsymbol{h}}_{i,j,k}^2,	\quad(2)
$$

$$
W_i^s=\alpha A_i+(1-\alpha)M_i, \quad(3)
$$

$$
\boldsymbol{r}_{i,j}=
\begin{cases}
1, & \mathrm{if}W_{i,j}^s\text{ is the top }N_s\text{ largest values of }W_i^s \\
0, & \mathrm{otherwise} & 
\end{cases},	\quad(4)
$$

$$
h_{i,j}^s=h_{i,j}+r_{i,j}\cdot{p_{i,j}^s}. 	\quad(5)
$$

* **帧间时间提示** ：帧内空间提示可以突出帧内的分辨区域，在此基础上，进一步引入帧间时间提示，以沿时间维度识别关键帧。如前面所述，首先使用三维卷积层来获取相邻帧之间的动态变化（公式1 2 3），其中为了让模型更加关注主要对象而非背景的变化，在公式3中会为帧内空间提示识别出的分辨区域赋予更高的权重。之后，基于帧时间变化 $$W_i^t$$的大小，确定在帧与帧之间插入的提示标记$$N_i^t$$ 的数量（公式4）。最后得到最终的特征（公式5）。最后，使用对比学习进行文本-视频匹配（公式6）。

$$
\Delta h_i^s=h_i^s-h_{i-1}^s,	\quad(1)
$$

$$
[\tilde{\boldsymbol{h}}_1^s,\cdots,\tilde{\boldsymbol{h}}_{N_F-1}^s]=\mathcal{N}^t([\Delta\boldsymbol{h}_1^s,\cdots,\Delta\boldsymbol{h}_{N_F-1}^s)]),	\quad(2)
$$

$$
W_i^t=\frac{1}{N_p\cdot d_v}\sum_j((1+\beta\cdot\boldsymbol{r}_{i,j})\sum_k(\tilde{\boldsymbol{h}}_{i,j,k}^s)^2),	\quad(3)
$$

$$
N_i^t=\lceil\eta\cdot W_i^t\rceil,	\quad(4)
$$

$$
[\boldsymbol{p}_1^t,\cdots,\boldsymbol{p}_{N_F}^t]=\mathcal{P}^t([\Delta\boldsymbol{h}_1^s,\cdots,\Delta\boldsymbol{h}_{N_F-1}^s]),\quad(5)
$$

$$
c(s,v)=\frac{s\cdot v}{\|s\|\|v\|},	\quad(6)
$$

# 5.模型性能

性能如下

1. 动作识别任务实验：

![性能图1](https://pic1.imgdb.cn/item/685151e058cb8da5c8558fc4.png)
2. 文本视频检索任务：

![性能图2](https://pic1.imgdb.cn/item/6851521f58cb8da5c855900c.png)
![性能图3](https://pic1.imgdb.cn/item/6851529458cb8da5c855907d.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章简单来说就是通过时空标记让模型关注时间和空间更重要的部分，也就是更加有针对性一点分析重要的部分，不关注不重要的部分。

