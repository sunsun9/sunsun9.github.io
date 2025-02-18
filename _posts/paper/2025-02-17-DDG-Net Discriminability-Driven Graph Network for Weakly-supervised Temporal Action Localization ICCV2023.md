---
layout: post
title: 'DDG-Net: Discriminability-Driven Graph Network for Weakly-supervised Temporal Action Localization ICCV2023'
subtitle: 'DDG-Net:  用于弱监督时间动作定位的可区分性驱动的图网络'
date: 2025-02-17
author: Sun
cover: 'https://pic1.imgdb.cn/item/67b03febd0e0a243d4ff9225.png'
tags: 论文阅读
---

> [DDG-Net: Discriminability-Driven Graph Network for Weakly-supervised Temporal Action Localization]([https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00010.pdf](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_DDG-Net_Discriminability-Driven_Graph_Network_for_Weakly-supervised_Temporal_Action_Localization_ICCV_2023_paper.html))
> 
> 1北京邮政与电信大学，中国；
> 2自动化研究所，中国科学院，中国；
> 3人工智能与机器人技术中心，HKISI_CAS，香港，中国；
> 4号中国科学院，UCAS，中国

# 1.文章针对痛点

这篇文章关注的弱监督时间动作定位*WTAL*，虽然这个名字与时序动作检测TAD不同，但是二者解决的问题是一样的，都是得到动作实例的开始、结束时间以及动作类别；至于弱监督时间动作定位，按照文章的解释就是在训练期间，仅提供动作类别用于训练学习。

而这篇文章关注的问题，一方面是认为**分类和定位任务之间存在一定的鸿沟**，想要在没有帧级别注释情况下，准确定位动作实例时困难的；另一方面文章认为**视频中具有大量的模棱两可的视频片段**（即不是动作片段与背景片段），这样的片段会**传递矛盾信息**，从而**降低连接片段的可区分性**。

# 2.主要贡献

为了解决上面提到的问题，文章提出了一种新的图网络结构，***Discriminablity-Driven  Graph Network (DDG-Net)***。该网络结构可以明显区分模棱两可片段与可区分性片段（指动作片段与背景片段）。此外，文章还提出了特征一致性损失，以维护用于定位的片段级表示特征。

原文将文章贡献总结为以下内容：

* 提出了一个新型的图形网络*DDG-NET*，该网络明确对具有不同类型的连接的模棱两可和歧视性片段进行了建模，旨在传播互补信息并增强片段级特征的可区分性，同时消除歧义信息的不利影响。
* 提出了*DDG-NET* 的特征一致性损失，从而防止了片段级特征的同化，并驱动图形卷积网络以生成更具歧视性表示。
* 广泛的实验表明文章方法是有效的，并在*Thumos14* 和*ActivityNet1.2* 数据集上建立了新的最新结果。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67b16bb4d0e0a243d4ffc100.png)

从上图中可以看到首先是利用两个不同的特征提取器，分别**提取RGB特征和光流特征**；之后计算两个特征的邻接矩阵，**得到注意力权重**；最后**建立**三种类型片段的**图结构**，并**进行推理学习**。

注：文章将视频片段分为了三种类型，即伪动作、伪背景和歧义片段。

# 4.实现细节

* **提取特征**：这个就不详细解释了，就是利用现有的模型架构提取两种特征。
* **计算邻接矩阵**：文章在计算邻接矩阵时，设计比较简单；仅仅是将特征相似度作为相邻节点的边权重，如下公式1。同时，为了利用两种特征类型的合作性，文章将得到的两种邻接矩阵求平均，如下公式2。<mark>注：伪动作、伪背景邻接矩阵构造，是在对应的伪动作、伪背景数据节点中构建的，也就是并涉及所有的视频片段，并且构建的是无向的；而歧义片段涉及所有的视频片段，其构造如公式3所示（计算的第一个结果也就是表明邻接矩阵只能从伪动作、伪背景开始，到歧义片段结束）。</mark>
  
  $$
  A_{ij}^{*}=s(f_{i}^{*},f_{j}^{*}), （1） \\A_{ij}=\frac{A_{ij}^r+A_{ij}^f}{2}，（2）
  $$
  
  $$
  A_m(i,j)=\left\{\begin{array}{ll}A(i,j), & j\in V_m\quad and\quad i\notin V_m; \\
  1, & i=j\quad and\quad i\in V_m; \\
  0, & otherwise.
  \end{array}\right.
  $$
* 图推理：这个过程是**涉及到增强伪动作、伪背景特征，同时不传递歧义片段特征**。增强特征也就是利用原来的特征与上一步构建的邻接矩阵进行一些运算，以伪动作特征为例，其增强特征公式如下，其中$$F_a^g$$是$$F_a^{avg}$$和$$F_a^{gcn}$$（也就是最后公式2结果）的均值。伪背景特征增强与之相同，至于歧义片段，只是$$F_m^{gcn}$$计算方式不同，公式4所示（其中，$$A_{m,a}$$是$$A_m$$中仅包含歧义片段特征的列和动作片段特征行。）。

$$
F_a^{avg}=F_aA_a, （1）\\
F_l^{gcn}=\sigma(W_a^lF_{l-1}^{gcn}A_a), （2）\\
F_a^{\prime}=\frac{F_a^g+F_a}{2}, （3） \\
F_{m}^{gcn}=F_{a}^{gcn}A_{m,a}+F_{b}^{gcn}A_{m,b}+F_{m}A_{m,m}， （4）
$$

* 一致性损失：文章还介绍了一致性损失，文**章认为*GCN* 更喜欢将所有片段特征转换为与分类任务的动作特征接近的特征，这对定位任务有害**。因此，文章提出功能一致性损失，以避免这种情况。

$$
\begin{gathered}
L_{fc}=\frac{1}{|V_{a}|}\sum_{t\in V_{a}}w(\mathbb{A}_{t})d(F_{a,t}^{gcn},F_{a,t}^{avg}) \\
+\frac{1}{|V_{b}|}\sum_{t\in V_{b}}w(1-\mathbb{A}_{t})d(F_{b,t}^{gcn},F_{b,t}^{avg}) \\
w(x)=exp(-(\frac{1}{x}-1)/\tau)
\end{gathered}
$$

# 5.模型性能

文章展示了所提出方法与其他方法的性能比较，这篇文章的性能目前来看是远远比不上全监督方法的性能，差别还是很大。![性能比较](https://pic1.imgdb.cn/item/67b29ce2d0e0a243d4001132.png)

# 6.改进/挑战/问题

这篇文章感觉只是将这个任务与弱监督结合起来，任务比较新颖，但是性能效果什么的并不是很好。

