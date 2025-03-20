---
layout: post
title: 'UniMD: Towards Unifying Moment Retrieval and  Temporal Action Detection ECCV 2024😊'
subtitle: 'UniMD: 致力于统一时刻检索和时序动作检测'
date: 2025-03-20
author: Sun
cover: 'https://pic1.imgdb.cn/item/67da454788c538a9b5c062a3.png'
tags: 论文阅读
---

> [UniMD: Towards Unifying Moment Retrieval and  Temporal Action Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06283.pdf)
> 💐💐提供代码
> 
> 📌作者单位
> 1.美团

# 1.文章针对痛点

这篇文章关注的是将时刻检索任务（*MR* ）与时序动作检测任务（*TAD* ）统一起来。

文章认为这两个任务之间又很强的关联性，但是现有的方法一般是将这两个任务作为独立的任务解决。

而文章在研究这两个任务的共有数据集，发现在**他们之间有潜在的互利性**：（1）来自*MR* 的时间可以表达多个动作之间的关系和顺序，因此可以建立动作之间的依赖性；（2）来自*TAD* 的动作可以作为一个完整事件的分解，为*MR* 任务提供更细致的监督；（3）*TAD* 和*MR* 的整合可以增强训练实例的数量。

# 2.主要贡献

为了研究探索这两个任务之间的潜在协同作用，文章提出了一个新的任务，叫做**时刻检测*Moment Detection (MD)***，目的是同时解决这两个任务。针对这个新的任务，文章提出了一种**统一任务架构，*Unified network for Moment Detection (UniMD)***，具有一个任务融合学习方法，可以增强两个任务的性能。

其中，<mark>针对任务统一架构</mark>，文章认为任务整合需要解决输入不一致性和动作范围差异两个问题。因此，文章提出了**建立统一的输入和输出接口**。

<mark>针对任务融合学习</mark>，文章检测并讨论了预训练和共同训练在任务融合学习上的影响。文章引入了**两种共训练方法：协同任务采样和选择任务采样**。也就是前者是有限考虑包含两个任务的视频采样，后者是选择某一种任务视频采样。

文章就是从上述的角度来探索两个任务之间的协同作用，最后通过实验验证确实可以互利。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67da4b7588c538a9b5c06591.png)

这个就是常规的编码-解码的架构，新任务的问题定义如下公式所示。这里需要注意的是在编码阶段，文本和视频模态并没有进行融合。

$$
\{(s,e,c_{\mathrm{md}})\}=f_{\mathrm{md}}(X,\{event\}),
$$

# 4.实现细节

* **文本编码器** ：文本编码器使用的是CLIP训练好的文本编码器，这里需要特别说明的就是针对TAD任务，怎么生成文本查询。这个就是按照 CLIP所需的prompt形式生成的，“*a video of  [action name]* ”。
* **视觉编码器**：这个也比较简单，文章想要使用**多尺度视频特征**，因此使用了特征金字塔模型来提取特征。但是这里需要特别说明的是，文章并没有使用常规的transformer的特征金字塔模型，而是使用了**基于卷积来生成特征金字塔**。因为文章认为在处理这两个任务时，不需要依赖长时序关系，使用卷积提取邻居信息即可。（这个卷积使用的是CVPR2022的文章[A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)）
* **解码器**：解码部分也没有像基于transformer的方法，设计很简单，依旧是**利用的卷积**，就是**包含依赖查询的分类头和依赖查询的聚合头**。1️⃣依赖查询的分类头：这个从结构图种可以清楚的看到是怎么运行的，包含什么部分，就是在两个模态融合的时候，使用了内能积；过程可以用下公式1表示。2️⃣依赖查询的聚合头：这部分的设计和分类头的设计类似，但是在处理文本特征的时候稍显不同，因为聚合头是需要给出事件边界的，所以文章使用了*MLP* 和*Resize* 模块对文本特征进行了处理，处理公式如下2所示，最后得到的文本呢特征的形状是$$\begin{aligned}w_{reg}\in\mathbb{R}^{K\times K\times CH\times2}\end{aligned}$$。该部分整体的处理如下公式3，注：文本特征是作为卷积核使用的。
  解码器整体的损失函数如下公式4。

$$
\begin{aligned}
\begin{Bmatrix}
conf^t\mid t\in[1,T]
\end{Bmatrix} & 
\begin{aligned}
=\Phi_{\mathrm{cls}}\left(G,Z\right)
\end{aligned} \\
 & =\Phi_{\mathrm{sigmoid}}(\Phi_{\mathrm{scale}}(G\odot\Phi_{\mathrm{conv}}(Z))))),
\end{aligned} \quad(1)
$$

$$
w_{\mathrm{reg}}=\Phi_{\mathrm{resize}}(\Phi_{\mathrm{mlp}}(G)),  \quad(2)
$$

$$
\begin{aligned}
\begin{Bmatrix}
(d_\mathrm{s}^t,d_\mathrm{e}^t)\mid t\in[1,T])
\end{Bmatrix} & 
\begin{aligned}
=\Phi_{\mathrm{reg}}(G,Z)
\end{aligned} \\
 & =\Phi_{\mathrm{relu}}(\Phi_{\mathrm{scale}}(\Phi_{\mathrm{q-conv}}(\Phi_{\mathrm{conv}}(Z)))),
\end{aligned}\quad(3)
$$

$$
\mathcal{L}_{\mathrm{task}}=\sum_t\sum_i(\mathcal{L}_{\mathrm{cls}}+\delta_{t,i}\cdot\mathcal{L}_{\mathrm{reg}}), \quad(4)
$$

* **任务融合学习**：这个部分就是**为了探究这两个任务通过任务融合学习是否真的可以互惠**。通过前面已经介绍，根据两个任务的数据注释分析，理论上二者可以从三个方面互惠。为了进一步研究文章提出针对两个任务，可以使用预训练和共训练的方式来探究（文章进行了实验验证）。文章认为预训练的方法需要两个模型的参数集合用于连个任务，但是使用共训练可以同时训练两个任务，因此**本文方法采用了共训练的方式**。针对共训练文章提出了三种采样方式，分别是**协同任务采样、有选择任务采样和随机任务采样**。协同任务采样就是保证采样的数据包含两个任务，有选择任务采样就是将两个任务视为单独的任务，采样只需要执行一个任务即可，模型也在单任务的设置下更新。随机任务采样也就是随机采样，要么两个任务的数据都采样到了，要么仅采样了单任务数据。

# 5.模型性能

实验性能效果对比图，在弱监督的情况下，感觉这个性能其实也还行相比较全监督的话。![效果对比图1](https://pic1.imgdb.cn/item/67db7d9288c538a9b5c1657b.png)
![效果对比图2](https://pic1.imgdb.cn/item/67db7da788c538a9b5c16582.png)
![效果对比图3](https://pic1.imgdb.cn/item/67db7df088c538a9b5c1658f.png)

文章也验证了预训练和共训练的比较，下图展示了文章进行的消融实验，（b）展示了预训练和共训练的比较，可以看到共训练对两个任务的提升更显著。![消融实验](https://pic1.imgdb.cn/item/67db7e2188c538a9b5c1659b.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章就是关注重点是在探究时序动作检测和时刻检索这两个任务之间的互惠性，但是这篇文章在得到时间边界的时候，采用的也不是传统的找中心和位移这种方式，也是找的边界距离当前位置的距离，所以上一篇文章直接是重点研究这个问题，怎么描述我的想法呢，emm....
* **问题**：文章有提到认为使用卷积来处理就足够了，不需要使用注意力关注长距离信息，但是我觉得这个部分应该也有一个消融实验来证明吧，不然不就是自己的猜测吗，而且如果用了transformer性能提升比较大的话，那假设就不成立了吧。

