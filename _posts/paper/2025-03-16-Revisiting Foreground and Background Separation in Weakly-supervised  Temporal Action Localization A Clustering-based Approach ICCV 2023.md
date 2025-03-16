---
layout: post
title: 'Revisiting Foreground and Background Separation in Weakly-supervised  Temporal Action Localization: A Clustering-based Approach ICCV 2023'
subtitle: '重新审视弱监督时间动作定位中前景和背景分离：一种基于聚类的方法'
date: 2025-03-16
author: Sun
cover: 'https://pic1.imgdb.cn/item/67d4e7da88c538a9b5bdf1e2.png'
tags: 论文阅读
---

> [Revisiting Foreground and Background Separation in Weakly-supervised  Temporal Action Localization: A Clustering-based Approach](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Revisiting_Foreground_and_Background_Separation_in_Weakly-supervised_Temporal_Action_Localization_ICCV_2023_paper.pdf)
> 💐💐提供代码
> 
> 📌作者单位
> 1.中国科技大学
> 2.合肥综合性国家科学中心人工智能研究院

# 1.文章针对痛点

这篇文章关注的是弱监督时间动作定位任务*WTAL*，也就是仅使用视频级动作标签。

但是现在的主流方法采用逐类定位管道，也就是将*WTAL* 任务视为视频动作分类问题，来学习时间类激活序列***T-CAS***。☹☹但是文章认为这种方法，**前景和背景分离仍然是一个开放问题，因为视频级标签无法提供任何背景类线索**。

而为了解决上述问题，也有方法提出解决措施。一种是基于多实例学习，使用*T-CAS* 来选择每个动作类别最自信的片段；另一种是注意力机制来学习类不可知前景权重（表明片段属于前景的可能性）。☹☹但是文章认为这些方法，仍十分**依赖视频分类损失来引导*T-CAS* 或注意力权重的学习**；但是这有固定的缺点，**损失很容易被显著片段最小化，导致不能探索整个片段的分布**，从而造成错误*T-CAS* 或注意力权重。

# 2.主要贡献

因此，为了解决上面的问题，文章提出了使用深度聚类来帮助前景和背景分离。具体来说，文章提出了一种新的网络架构*Clustering-Assisted F&B SEparation (CASE)*。首先通过**构建一个标准的*WTAL baseline*，提供*F&B* 片段初始估计**；之后**引入一个基于聚类的*F&B* 分类算法来细化*F&B* 分离**，这个算法包含两个主要组件，分别是片段聚类和簇分类，前者是将片段分为多个簇，后者是将簇进一步分为前景或背景；**而因为没有标签来训练组件，因此文章引入统一自标签机制来生成高质量伪标签**。

原文将文章工作总结如下：

1. 我们提出了一种基于集群的*WTAL F&B* 分离算法，该算法将*F&B* 分离问题视为片段集群和集群分类的结合。
2. 我们提出了一种基于最优传输的统一自标签机制来指导片段集群和集群分类。
3. 我们进行了广泛的实验，证明了我们的方法与现有方法相比的有效性和效率。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67d50ca888c538a9b5be3ba6.png)

这个图有点抽象，部分*a* 是文章采用的*baseline*，部分*b* 和*c* 是文章在*baseline* 的基础上提出的新的组件用于学习训练。后面详细介绍。

# 4.实现细节

* **baseline** ：*baseline* 分为两个分支，上面一个分支是视频分类分支，下面分支是注意力分支，这个分支应该是为了时间定位。在训练视频分类分支时，文章采用了多实例学习；训练注意力分支时，使用伪标签。
* **基于聚类算法**：这个模块分为了片段级聚类模块和簇分类模块。1️⃣片段级聚类模块：是将视频片段嵌入特征输入到聚类头里，得到片段级聚类分配概率，之后使用伪标签监督训练，公式如下1。（但是这个过程和损失在架构图中看不出来，到底在哪）2️⃣簇分类模块：这个部分是通过将簇原型投影到*F&B* 原型实现的。具体来说，首先计算k个簇原型，公式2所示；之后计算簇级别分类概率，如公式3；最后使用伪标签监督训练，损失公式如公式4。（这里面的伪标签没看懂是怎么得到的）

$$
\mathcal{L}_{S}=\frac{1}{N}\sum_{n=1}^{N}\mathcal{L}_{CE}(Q_{n}^{S},P_{n}^{S}). \quad(1)
$$

$$
\bar{E}_k^S=\frac{\sum_{n=1}^NQ_{n,k}^S\cdot E_n}{\sum_{n=1}^NQ_{n,K}^S}.\quad(2)
$$

$$
\boldsymbol{P}_{k,i}^{\boldsymbol{C}}=Softmax
\begin{pmatrix}
\rho\cdot\cos(\bar{\boldsymbol{E}}_{k}^{\boldsymbol{S}},\bar{\boldsymbol{E}}_{i}^{\boldsymbol{A}})
\end{pmatrix},\quad(3)
$$

$$
\mathcal{L}_C=\frac{1}{K}\sum_{k=1}^K\mathcal{L}_{CE}(Q_k^C,P_k^C). \quad(4)
$$

* 文章后面的内容解释了怎么生成的两个伪标签，但是看不懂，根本看不懂😭😭


# 5.模型性能

实验性能效果对比图，在弱监督的情况下，感觉这个性能其实也还行相比较全监督的话。![效果对比图1](https://pic1.imgdb.cn/item/67d6341a88c538a9b5bef41c.png)
![效果对比图2](https://pic1.imgdb.cn/item/67d634b488c538a9b5bef439.png)

# 6.改进/挑战/问题/想法

* **想法**：这种太数学原理的论文都看不懂 烦人☹☹

