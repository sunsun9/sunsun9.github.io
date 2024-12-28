---
layout: post
title: 'Test-Time Zero-Shot Temporal Action Localization CVPR2024'
subtitle: '测试时间零样本时序动作定位'
date: 2024-12-25
author: Sun
cover: 'https://pic.imgdb.cn/item/676b6901d0e0a243d4e99f82.png'
tags: 论文阅读
---

> [测试时间零样本时序动作定位](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html)
> 
> 特伦托大学1 布鲁诺·凯斯勒基金会2（翻译软件译，可能有误)

# 1.文章针对痛点

首先这篇文章是关于时序动作检测的，题目所说的测试时间指的是测试阶段。

文章针对的痛点可以从以下两个方面讲述：

* **数据不可用**：在将大模型迁移到自己任务方向时，现有方法往往需要基于新的任务方向对大模型进行微调；但是这种方式需要大量的可用数据（新任务上的），而在真实情况下，这样的数据集往往是不可用的。（也就是说还是一个数据不足的情况，这算是使用深度学习模型都存在的问题）
* **域外泛化能力下降**：如果使用微调，会造成这种情况；文章提到微调本身会产生域外泛化能力下降的模型，并且认为这个问题会对零样本时序动作检测（*ZS-TAL* ）产生严重影响。（我认为这个原因，可能是本来的大模型具有很好的泛化能力，但是在特定数据集上进行微调，也就是说让它更关注了这部分数据的特征，那之前学习到的部分强大的泛化特征可能会丢失或者损坏）
  
  所以在这些问题下，文章认为这种预训练的方式会导致人们对*ZS-TAL* 方法适应性和鲁棒性的担忧。

# 2.主要贡献

为了解决上述的问题，作者也受到了测试阶段自适应方法的启发。**考虑可以不进行预训练，而是认为测试阶段的输入数据也可以用于对模型更新，这也是文章的核心想法**。因此，文章提出了一种新的方法*T3AL, Test Time adaptation for Temporal Action Localization* 。这种方法直接在推理阶段，对提供的可用视频进行自适应模型参数更新。

原文将文章贡献总结为了以下几部分：

1. 文章在没有训练数据的新实际场景中解决了*ZS-TAL* 问题。并通过实验证明这是一个具有挑战性的问题，因为最先进的任务方法在没有训练的情况下泛化能力很差。
2. 文章提出了 *T3AL*，这是第一种利用预先训练的 *VLM* 解决没有训练数据的 *ZS-TAL* 问题的方法。*T3AL* 受益于有效的 *TTA* 策略和从生成的字幕中获得的外部知识。
3. 通过经验证明，适应未标记的数据流是解决当前基于训练的 *ZS-TAL* 方法的分布不均问题的可行解决方案。

与现有方法的设计对比![模型对比](https://pic.imgdb.cn/item/6767cbedd0e0a243d4e800db.png)

# 3.实现流程

先放一下这篇文章的整体架构图：![整体架构图](https://pic.imgdb.cn/item/676919a6d0e0a243d4e87419.png)

首先要明确的是，这篇文章面向的是时序动作定位的零样本任务，且并不涉及训练阶段，所有的参数更新都是在推理也就是测试阶段完成的。

整体的流程大概可以包含以下几个步骤：

1. 首先就是先提取特征，包括提取视觉特征和文本特征；需要注意的是这两个特征提取器是冻结的；
2. 第二步就是将获取到的视觉特征进行平均，并于文本特征对齐，通过计算获得视频的伪标签；
3. 第三步是进行自监督学习，实现预测结果细化（主要是针对定位子问题）；
4. 最后一步是实现文本引导的区域抑制，就是指去除某些区域。

# 4.实现细节

主要介绍三个组件的实现细节：视频伪标签、预测结果细化和文本引导的区域抑制。

* **视频伪标签：**这个部分实现相对简单，就是将提取到的帧级别的视频特征进行**平均**（文章提到认为这样可以有效减弱非视频信息的噪声）；之后**与提取的文本特征进行对齐**，**计算余弦相似度**，相似度高的认为是伪标签。这个过程可以用下列公式表示：

$$
\bar{\mathcal{V}}=\frac{1}{N}\sum_{i=1}^{N}\mathcal{E}_{V}\left(x_{i}\right)
$$

$$
y^{*}=\underset{y\in\mathcal{C}}{\operatorname*{\mathrm{argmax}}}\cos\left(\bar{\mathcal{V}},\mathcal{E}_{L}\left(y\right)\right)
$$

* **预测结果细化：** 这个部分主要是针对上一步给定的粗粒度预测进行细化，从而有效定位相应动作的时间区域。一般而言，视频中包含于动作相关的帧，但是也包含于动作无关的内容；正常来说，模型是可以区分的；但是对于哪些与伪标签语义相关，而与动作执行无关的视频线索，模型很难区分。
  
  因此，文章**设计计算视频每一帧与伪标签的相近程度分数**（公式如下1）。分数高，认为该帧与前景相关程度高（也就是与动作更相关）；分数低，则认为与背景相关程度高（也就是与动作不那么相关）。之后，文章**利用这些帧进行自监督学习从而细化初始的预测**。具体来说，具有高分数的帧形成正样本，低分数的帧形成负样本（表示如下2）。最后的自监督学习目标如下3，也就是训练模型更新这三个参数。
  
  这部分的损失函数如下4，$$\mathcal{L}_z$$仅用于正样本，$$ \mathcal{L}_s$$可用于正负样本。具体来说，$$\mathcal{L}_z$$公式如下5，$$ \mathcal{L}_s$$公式如下6，这个公式的目的是为了让正样本的分数更靠近1，让负样本的分数更靠近0。
  
  注：这篇文章在收集正样本时，并不是使用的一些数据增强方式，而是假设语义知识是一个连续的函数，因此直接使用该帧附近的一些帧作为正样本。
  
  **在测试时间自适应的每一步**，都会重新计算 $$\mathcal{Z}^+$$ 和 $$\mathcal{Z}^-$$。在 T 步骤之后，自适应模型会为 V 中的每一帧分配最终分数。计算这些分数的移动平均值以进一步增强时间一致性。然后，通过使用阈值 γ 进行过滤来获得时间动作提议，即开始/结束时间位移 $${\{\hat{t}_{i}\}}_{i=1}^{\hat{M}}$$。

$$
s_i=\cos(\mathcal{E}_V(x_i),\mathcal{E}_L(y^*))   \hspace{2em} (1)
$$

$$
\mathcal{Z}^+=\{\left(z_i^+,s_i^+\right)\}_{i=1}^K,\quad\mathcal{Z}^-=\{\left(z_i^-,s_i^-\right)\}_{i=1}^K	\hspace{2em}(2)
$$

$$
\begin{pmatrix}
\theta_{\mathcal{P}_V}^*,\theta_{\mathcal{P}_L}^*,\tau^*
\end{pmatrix}=\underset{\theta_{\mathcal{P}},\tau}{\operatorname*{\operatorname*{argmin}}}\mathcal{L}		\hspace{2em}(3)
$$

$$
\mathcal{L}=\mathcal{L}_z+\mathcal{L}_s	\hspace{2em}(4)
$$

$$
\mathcal{L}_z=2-2\cdot\frac{<z_i^+,z_j^+>}{\|z_i^+\|_2\cdot\|z_j^+\|_2}		\hspace{2em}(5)
$$

$$
\mathbf{s}=\mathrm{concat}
\begin{pmatrix}
\{s_i^+\}_{i=1}^K,\{s_i^-\}_{i=1}^K
\end{pmatrix}\in\mathbb{R}^{2K}, \mathbf{b}=
\begin{bmatrix}
1_K \\
0_K
\end{bmatrix}\in\mathbb{R}^{2K}, \mathcal{L}_s=2-2\cdot\frac{<\mathbf{s},\mathbf{b}>}{\|\mathbf{s}\|_2\cdot\|\mathbf{b}\|_2}	\hspace{2em}(6)
$$

* **文本引导的区域细化：** 这一步的目的是减少一些潜在的不正确的预测动作提案。利用了现有的字幕模型生成文本实现文本引导。具体来说，所有被选取作为提案的帧都会<mark>进行字幕注释</mark>，然后将这些注释输入前面的文本编码器中，平均每个提案的文本特征，得到每个提案的区域级特征。之后<mark>建立拒绝标准</mark>，通过计算所有这些特征成对余弦相似度，可以如下1表示。然后，在阈值 β 处将其二值化并得到 $$\hat{D}$$，逐列求和此二值掩码中的元素，我们得到一个得分向量$$\mathbf{d}=\hat{\mathbf{D}}\operatorname{diag}\left(\mathbf{I}_{\hat{M}}\right)\in\mathbb{R}^{\hat{M}}$$，该向量衡量<mark>每个动作提议与其他动作提议的相似性</mark>。最后，如果**提议在 d 中的关联条目低于阈值 α，即其关联文本表示与其他文本表示不够接近**，则抑制该提议。

$$
\mathbf{D}=\left[d_{ij}\right],\quad d_{ij}=\cos\left(d_i,d_j\right)	\hspace{2em}(1)
$$

# 5.模型性能

文章提出来的这种在测试时间的更新参数的方法并没有达到传统的基于训练的方法的性能，但是也是一种方向吧。

![性能对比](https://pic.imgdb.cn/item/676b66edd0e0a243d4e99f3d.png)
注：前面的三种方法文章提到*的是，在预训练的*VLM* 基础上提出了三种基线，可能是用于对比；好像就是选择不同的*VLM* 进行简单的时间动作定位。下面就是本文提出的方法，最小面紫色的是基于训练的方法。

![消融实验](https://pic.imgdb.cn/item/676b671bd0e0a243d4e99f3f.png)
注：这个图显示的是模型的消融实验部分，文章主要验证模型两部分的有效性。一个是提出的损失函数，这里提到的*z* 就是前面的对比学习，*s* 就是前面提到的应该让在正样本集合中的帧的分数更靠近1；另外一个是验证的文本引导的抑制的作用，通过视频注释计算与其他正样本的相似度那个。

# 6.改进/挑战/问题

其实这篇文章没有太看懂，主要就是用了自监督对比学习喜欢提案，文本提示进一步细化提案，我觉得这种利用字幕提示减少潜在的提案是一种很好的方法，后面可以试试。

其次，这篇文章的原文有点难读，用词比较偏僻，且句子较长😕

* **问题：** 这里有一个没懂的是，这个测试时间训练按照文章的说法应该不进行训练了才对，但是这篇文章在设置实验的时候还是选择了训练，分为了50%-50%以及75%-25%这两种设置。
* **挑战：** 这种在测试阶段的更新参数方式，用于解决分布外问题，既然都发了*CVPR*，虽然性能可能没有那么好，但是也可能是一种方向吧。应该要考虑怎么完全不利用训练阶段，仅仅只利用测试阶段的数据。
* **改进：** 这个性能没有达到那么好，感觉可以在两个模态交互上再做一下改进。这种方法是不死也会忽略全局性，感觉可以参考一些其他的方法更好的完成语义对齐。
