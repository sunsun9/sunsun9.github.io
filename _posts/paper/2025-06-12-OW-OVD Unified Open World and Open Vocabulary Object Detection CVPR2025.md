---
layout: post
title: 'OW-OVD: Unified Open World and Open Vocabulary Object Detection CVPR2025😊'
subtitle: 'OW-OVD: 统一开放世界和开放词汇物体检测'
date: 2025-06-12
author: Sun
cover: 'https://pic1.imgdb.cn/item/68491fdf58cb8da5c84414b0.png'
tags: 论文阅读
---

> [OW-OVD: Unified Open World and Open Vocabulary Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Xi_OW-OVD_Unified_Open_World_and_Open_Vocabulary_Object_Detection_CVPR_2025_paper.html)

> 💐💐[提供代码](https://github.com/xxyzll/OW\_OVD)
> 
> 📌作者单位
> 
> 1. 华南理工大学

# 1.文章针对痛点

这篇文章关注的问题是物体检测。目前的研究方向发展趋势开始向开放集靠拢，这篇文章考虑到的就是目前有很多研究开放视觉物体检测和开放词汇物体检测的问题，但是往往是仅研究其中的一种；而文章考虑到两种任务各有优势，因此可以将两个任务结合。

**开放词汇物体检测*OVD*** 利用一个预训练的文本编码器，将类别名称转换为文本嵌入，从而后续能够与视觉嵌入匹配，这样可以消除可检测类别数量的限制。**开放世界物体检测*OWOD*** 需要检测器识别未注释的类别并能通过增量学习识别新类别。<mark>前者具有良好的泛化能力，能够在不注释物体的情况下将新类别名称与未见类别匹配，实现零样本检测；但是该任务通常缺乏主动发现新物体的能力。后者能够检测未注释物体，并通过增量学习逐步更新；但是却缺乏开放词汇物体检测的零样本泛化能力。总的来说，前者具有良好的泛化能力，但是如果不告诉新类别名字，其不会发现新物体；而后者虽然能发现新物体，但是并不具备将名字与物体对齐的功能。</mark>

# 2.主要贡献

所以针对上面的这个问题，构建这两个任务限制之间的桥梁，文章提出了*OW-OVD*，一种用于有效解决*OVD* 和*OWOD* 任务的新检测器，结合了两种任务各自的优势。

为了实现这个目标，文章在现有的*OVD* 检测架构的基础上，增强其检测未见类别的能力（不破坏*OVD* 原有的零样本检测能力），即文章的目的就是让检测器可以支持*OVD* 和*OWOD* 任务。

具体来说，文章提出了视觉相似度属性选择方法，*Visual Similarity At tribute Selection (VSAS)*。该方法利用物体检测领域的标准匹配方法，首先将视觉嵌入分为正样本和负样本，之后计算并汇总所有属性和视觉嵌入之间的相似度，得到属性相似度的条件概率分布；通过比较正样本与负样本相似度分布，得到注释区域和未注释区域的差异；基于这个差异，选择注释区域和未注释区域普遍的属性。此外，为了避免选择的属性过于相似，文章引入了相似度约束。

对于未知类别的预测，文章提出了混合 属性-不确定性融合方法，*Hybrid Attribute-Uncertainty Fusion  (HAUF)*。该方法整合已知类不确定性和加权属性相似度来估计一个给定视觉区域被分类为未知类别的可能性。

原文总结贡献如下：

1. 据我们所知，我们是第一个提出同时具备 *OWOD* 和 *OVD* 任务优势的模型。
2. 我们提出了视觉相似性属性选择（*Visual Similarity Attribute Selection，VSAS* ）方法，该方法通过比较注释区域和未注释区域中属性的相似性分布来识别这些区域的共同属性。
3. 我们提出了混合属性-不确定性融合（*HAUF* ）方法，将已知类不确定性和加权属性相似性相结合，在不改变 *OVD* 推理过程的情况下识别未知对象。
4. 我们在 *OWOD* 基准任务、*M-OWODB* 和 *S-OWODB* 上验证了 *OW-OVD*。结果表明，在已知和未知类别中，*OWOVD* 都优于现有的最先进（*SOTA* ）模型，在 *U-Recall* 和 *mAP* 中分别提高了 +15.3 和 +4.3。

# 3.实现流程

下图展示了文章提出方法的架构图。这个架构主要就是基于属性相似度分布来给出结果。

![模型架构图](https://pic1.imgdb.cn/item/68495a4b58cb8da5c844fea8.png)

# 4.实现细节

* **属性生成**：这个模块可以说是文章想法的基础。为了最小化单个物体描述的潜在偏差，文章采用了使用大语言模型来生成属性。具体而言，给定已知类别名称，将该名称输入到大模型，让大模型生成一组与名称描述相关的特征（比如颜色、形状等等）；这些特征之后被插入到预定义的模板中，从而生成一个描述该物体的句子；最后使用文本编码器编码该句子，公式如下。类似于这个公式生成的描述嵌入，类别名称嵌入$$E_C$$也是这种编码方式。

$$
E_{att}=f_{txt}(Sen)\in\mathbb{R}^{s\times d}
$$

* **视觉相似性属性选择**：这是文章提出方法的核心部分。前面利用大语言模型生成的特征属性**经常是存在冗余或者重叠的**，为了减少这些冗余特征对模型性能的影响，<mark>文章提出了视觉相似性属性选择；该方法通过计算正负样本属性之间的相似性，从而选择最泛化的属性。</mark>该模块包含两个主要步骤：**分布构建和迭代属性选择**。1️⃣分布构建。**为了评估正负样本每个属性之间的差异，文章通过计算这些属性和对应区域之间相似性**。具体而言，文章通过对比学习头和*box* 头作用于编码器得到的特征，得到边界框和类别相似性分数，如下公式1所示。之后，将该相似性分数与所以已知描述匹配，从而得到正负样本，如下公式2所示。最后计算属性与正负样本之间的相似性，得到当前属性关于标记区域和未标记区域之间相似性分布，如下公式3所示。2️⃣迭代属性选择。文章目标是从这些描述中选择可用于未知对象的句子，而这些句子是根据已知类别生成的，因此它们必须足够通用，能够有效地描述未知类别，而不是专门针对已知类别。换句话说，这些句子在描述有注释的区域（已知类别）和无注释的区域（未知类别）时都应表现出相似性。因此，需要选择与$$D_{i}^{+}$$和 $$D_{i}^{-}$$最相似的特征，并使用 *JSD* 来评估两个概率分布之间的相似性，如下公式4所示。而为了避免迭代选择的属性过于相似，文章使用了惩罚进行约束，如下公式5所示。

$$
P_{cls}=h_{cls}(E_{vis},E_{C}),P_{box}=h_{box}(E_{vis}),	\quad(1)
$$

$$
\begin{aligned}
S= & \pi(\lambda\left(L_{cls}(P_{cls},G_{cls})\right)+L_{box}(P_{box},G_{box})), \\
 & & \left(2\right) \\
E_{vis}^{+}= & \{e_{vis_{i}}|S[i]\geq\alpha\},E_{vis}^{-}=\{e_{vis_{i}}|S[i]<\alpha\},
\end{aligned}
$$

$$
d_i^+=h_{cls}(e_{vis}^+,e_{att_i}),d_i^-=h_{cls}(e_{vis}^-,e_{att_i}),  \quad(3)
$$

$$
\begin{aligned}
\hat{E}_{att}= & \hat{E}_{att}\cup ArgminJSD(D_{i}^{+},D_{i}^{-}), \\
 & & \mathrm{(4)} \\
JSD(D_{i}^{+},D_{i}^{-})= & 
\begin{aligned}
\frac{1}{2}(KL(D_{i}^{+},M)+KL(D_{i}^{-},M)),
\end{aligned}
\end{aligned}
$$

$$
\begin{aligned}
\hat{E}_{att}= & \hat{E}_{att}\cup Argmin(\beta\cdot JSD(D_{i}^{+},D_{i}^{-})+ \\
 & (1-\beta)\cdot\frac{1}{|\hat{E}_{att}|}\sum\sigma(sim(e_{att_{i}},\hat{E}_{att}))),
\end{aligned}	\quad(5)
$$

* **混合属性-不确定性融合**：这个模块目的是在保留*OVD* 良好的零样本能力情况下，接入*OWOD* 任务。该模块摒弃之前方法采用的额外线性扩展层，而是以一种并行的方式预测未知类别物体。具体来说，文章从两个方面考虑未知类别物体，一个是它们与前面生成的通用属性的相似性，另外一方面是它们与已知类别的不确定性。文章使用到的方法如下公式1所示，其中$$P_b$$ 用于将未知物体从背景中区分出来，而 $$P_{un}$$ 则代表模型对当前物体相对于已知类别的预测熵。这两个角度都能评估一个物体成为正样本的可能性，但仅仅依靠这两个角度可能会导致将已知物体错误分类为未知物体。因此，文章引入了分布外概率，如下公式2所示。针对这一部分，文章展示了该部分在推理阶段的示意图。

$$
\begin{aligned}
P_{b} & =\frac{1}{\gamma}\sum_\gamma softmax(s_\gamma)s_\gamma,s=h_{cls}(e_{vis},\hat{E}_{att}), \\
 & & \mathrm{(1)} \\
P_{un} & =\frac{1}{k}\sum_{k}-(p_{k}log(p_{k})+(1-p_{k})log(1-p_{k})),
\end{aligned}
$$

$$
P_u=\frac{1}{2}(P_b+P_{un})(1-max(P_C)),	\quad(2)
$$

![示意图](https://pic1.imgdb.cn/item/684a6cdb58cb8da5c8471762.png)

# 5.模型性能

这篇文章也是稍微进行数据集重构。但是这个实验基本都是在验证*OWOD* 任务上的表现性能怎么样，虽然是在*OVD* 任务架构基础上，但是个人感觉也还是需要验证一下模型架构是否真的保留了*OVD* 的零样本检测能力。

![性能图1](https://pic1.imgdb.cn/item/684a6df558cb8da5c8472420.png)
![性能图2](https://pic1.imgdb.cn/item/684a6e3d58cb8da5c847277e.png)
![性能图3](https://pic1.imgdb.cn/item/684a6eab58cb8da5c8472d16.png)

# 6.改进/挑战/问题/想法

* **想法**：这个文章的切入点是值得借鉴的，现在由于具身智能的发展以及开放世界任务的发展，目前的研究认为在全监督的模式下训练是耗时耗力的，而开放世界或者开放词汇任务的提出，可以有效改善这个问题。感觉这是每个任务都可以切入的点。现在计算机视觉的热点基本都是具身智能、开放世界、生成式以及3d、4d生成等等。

