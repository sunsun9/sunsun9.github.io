---
layout: post
title: 'Not All Inputs Are Valid: Towards Open-Set Video Moment  Retrieval Using Language ACM MM 2024'
subtitle: '不是所有的输入都是有效的：使用语言面向开放集视频时刻检索'
date: 2025-03-10
author: Sun
cover: 'https://pic1.imgdb.cn/item/67cd0269066befcec6e182db.png'
tags: 论文阅读
---

> [Not All Inputs Are Valid: Towards Open-Set Video Moment  Retrieval Using Language](https://dl.acm.org/doi/10.1145/3664647.3680947)
> 
> 华中科技大学&北京大学&浙江工商大学&大连科技大学&上海交通大学&新疆大学&香港中文大学

# 1.文章针对痛点

这篇文章关注的是开放集设置下的视频时刻检索任务。视频时刻检索任务是检索未剪裁视频中与给定句子查询相关的视频时刻，传统的闭集任务是默认给定的句子一定与视频内容相关，也就是说一定可以在视频中找到对应的时刻；而开放集设置下，认为给定的句子不一定与视频内容相关，也就是说本应该在视频中找不到对应的内容。

然而，**现有的方法在真实开放世界时，可能给定查询与视频内容根本不相关，但是模型仍然给出了一个视频片段**，而这种错误在高风险场景中，是一种无法弥补的损失，例如给定句子描述了与视频不相关的犯罪活动，现有模型会将正常视频内容认为是犯罪场景。

此外，在真实世界场景中，**不仅仅存在大量与视频内容不相关的查询，也存在注释并不是逐帧的，往往只是视频级别的注释**。

# 2.主要贡献

因此，为了解决上面的问题，文章提出了一个新的具有挑战性的任务：开放集视频时刻检索任务，*open-set  VMR(OS-VMR)*。这个模型的目的不仅仅是检测出与*ID query* 相关的视频片段，还需要能够拒绝*OOD query*。*ID query* 指的是与视频相关的句子查询，*OOD query* 指的是与视频不相关的句子查询。

针对这个任务，文章提出了一个新的*OpenVMR* 框架。该框架首先设计了一个多层耦合块，来构建正常化流，学习基于多变量高斯分布假设的*ID query* 分布；之后，通过一个精心设计的不确定分数和每个查询的对数可能分布来推理*ID-OOD* 边界；并且针对*OOD* 查询检测，文章聚合*ID* 查询特征，来细化基于三重损失*ID-OOD* 边界；之后，对于视频和*ID* 查询，文章针对视频-查询匹配和帧-查询匹配，构建了跨模态交互；最后，文章设计了一个简单但是高效的具有预定义提案的正向无标签学习模块，来检索目标时刻。

文章总结的工作如下：

1. 据文章所知，文章是首次尝试在开放式视频时刻检索（*OS-VMR* ）任务中进行，这在开放式设置中从根本上更具挑战性，但非常有价值。在这种情况下，不仅应该检索视频时刻的*ID* 查询，还应拒绝*OOD* 查询。
2. 为了解决该项具有挑战性的*OS-VMR* 任务，我们提出了一个通用的*OpenVMR* 框架，该框架首先通过归一化流技术区分*ID* 和*OOD* 查询，然后在瞬间检索使用*ID* 查询。
3. 开放式设置和封闭设置的实验结果表明，文章所提出的模型的表现优于其他最先进的方法。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67cd2bd4066befcec6e1ba82.png)

整体就是两大部分。上面的而是为了构建*ID-OOD* 边界，也就是为了区分是*ID* 查询还是*OOD* 查询；下面的部分是确定了*OOD* 查询之后，怎么定位的问题。

# 4.实现细节

* ***ID* 知识获取** ：这个部分是为了学习*ID* 查询的分布。包含两个步骤：1️⃣正常化流构建，这一步就是构建了多个耦合层，文章构建的输入查询分布如下公式1所示，公式2是在训练分布$$p_Q$$的过程中获得的参数$${\omega}$$的集合。2️⃣是学习*ID* 查询特征分布。在使用多变量高斯分布的基础上得到$$p_{\mathcal{X}}(x)$$分布，如公式3；将公式2中所需的所有参数带进去，得到公式4.最后这部分损失如公式5。

$$
\log p_{\omega}(q)=\sum_{c=1}^{C} \log \left|\operatorname{det} J_{\Phi_{\mathcal{c}}}\left(k_{c-1}\right)\right|+\log p_{\mathcal{X}}\left(\Phi_{\omega}(q)\right)	\quad（1）
$$

$$
\omega^{*}=\underset{\omega}{\operatorname{argmin}} \mathbb{E}_{q \sim p_{Q}}\left[-\log p_{\omega}(q)\right] \quad（2）
$$

$$
p_{X}(x)=(2\pi)^{-\frac{d}{2}}\det(\sigma^{-\frac{1}{2}})\exp[-\frac{1}{2}(x-\mu)^{\top}\sigma^{-1}(x-\mu)] \quad（3）
$$

$$
\omega^{*}=\underset{\omega}{\operatorname*{\operatorname*{argmin}}}\mathbb{E}_{q\sim p_{Q}}\left[\frac{1}{2}\Phi_{\omega}(q)^{\top}\Phi_{\omega}(q)-\sum_{c=1}^{C}\log|\underset{\Phi_{c}}{\operatorname*{\operatorname*{det}}}J_{\Phi_{c}}(k_{c-1})|\right] \quad（4）
$$

$$
\mathcal{L}_{1}=\mathbb{E}_{q\in Q^{id}}\left[\frac{1}{2}\Phi_{\omega}(q)^{T}\Phi_{\omega}(q)-\sum_{c=1}^{C}\mathrm{log}|\mathrm{det}J_{\Phi_{c}}(k_{c-1})|\right] \quad（5）
$$

* ***OOD* 边界推理**：这个模块是为了找寻*OOD* 和*ID* 查询之间的明确边界，同时为了减少计算，文章考虑基于不确定性分数的查询边界；而上一步计算的对数可能性分布式可以转化成不确定性分数。这一部分也包含两步：1️⃣通过正常化流，可以估计每个查询特征的对数可能性，如下公式1。最后得到的不确定性分数如下公式2。2️⃣得到不确定分数后，就可以推理得到*ID-OOD* 边界。首先基于公式2的*ID* 对数可能性，可以找寻*ID* 对数可能性分布；之后，通过这个*ID* 对数可能性近似所有查询的对数可能性分布；之后，引入一个位置超参数$$\alpha$$来确定边界，文章设置分类*ID* 对数可能性分布的第$$\alpha$$个百分位数作为*ID* 边界，也作为容错率的上限；最后为了增强模型的鲁棒性，文章还应用了*margin* 超参数$$\Delta$$，定义OOD边界$$b_{o o d}=b_{i d}-\Delta$$。

$$
\mathrm{log}p(q)=\sum_{c=1}^C\mathrm{log}|\mathrm{det}J_{\Phi_c}(k_{c-1})|-\frac{1}{2}\Phi_\omega(q)^T\Phi_\omega(q) \quad（1）
$$

$$
u(q)=\max_{q^{\prime}\in\boldsymbol{Q}}(\exp(\operatorname{logp}(q^{\prime})))-\exp(\operatorname{logp}(q)), \quad（2）
$$

* ***ID-OOD* 边界细化模块**：这个部分时使用了对比学习，也就是让*ID* 查询更靠近，而*OOD* 查询要相对远离*ID* 查询。这个部分使用了一个三重损失，如下所示。
  $$
  \mathcal{L}_{2}=\sum_{i=1}^{N_{id}}|\min((\mathrm{log}p_{i}-b_{id},0)|+\sum_{j=1}^{N_{ood}}|\mathrm{max}((\mathrm{log}p_{j}-b_{id}+\Delta),0)|.
  $$
* **跨模态交互和训练**：这一部分就是包含视频-查询匹配和帧-查询匹配。前者也就是计算视频全局特征与查询之间的相似性，后者就是计算逐帧的注意力分数。此外，还提出了一个正样本-无标签学习，也就是将*ID* 查询作为正样本，而将*OOD* 查询作为无标记样本，文章认为将任务转换成半监督问题。

PS：文章涉及的公式是在太多了，对原理以及公式的解释感觉没有很详细，所以这些部分我看起来都是半懂不懂的状态😭

# 5.模型性能

实验性能效果对比图：![效果对比图1](https://pic1.imgdb.cn/item/67ce4e25066befcec6e24b0d.png)
![效果对比图2](https://pic1.imgdb.cn/item/67ce4e47066befcec6e24b1f.png)
![效果对比图3](https://pic1.imgdb.cn/item/67ce4e72066befcec6e24b27.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章在介绍方法时，确实太多公式了，感觉对于原理以及公式解释没有那么详细，所以看起来也是一知半解。了解这个任务是要接受*ID* 查询并进行相关时刻检索，同时拒绝*OOD* 查询；也清楚文章提出了几个部分，每个部分的核心功能是什么，但是完全不太清楚具体是怎么实现的。（科研太难了😭😭）
* **问题**：另外，在架构图中，部分1、部分2和部分3的数据流是流通的，但是这三部分与部分4之间感觉没有数据流的交互，所以那怎么实现拒绝OOD查询，还是训练和推理不一样呢。

