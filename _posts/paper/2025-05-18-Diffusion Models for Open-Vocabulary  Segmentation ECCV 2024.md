---
layout: post
title: 'Diffusion Models for Open-Vocabulary  Segmentation ECCV 2024'
subtitle: '用于开放词汇分割的扩散模型'
date: 2025-05-18
author: Sun
cover: 'https://pic1.imgdb.cn/item/6829412f58cb8da5c8f98e6c.png'
tags: 论文阅读
---

> [Diffusion Models for Open-Vocabulary  Segmentation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00794.pdf)

> ❌❌[不提供代码]
> 
> 📌作者单位
> 
> 1. Visual Geometry Group, Department of Engineering Science, University of Oxford

# 1.文章针对痛点

这篇文章关注的问题是开放词汇设置下的图像分割问题，该任务的目的是分割图像中与自然语言文本匹配的区域。

然而，文章认为开发适用于密集定位任务（如语义分割），不仅需要大量的训练，还需要昂贵的掩码注释。现有用于开放词汇的图像分割方法主要分为两种方式，**第一种是通过大视觉语言模型**，例如*CLIP* 等等，依赖标记数据来微调图像级表示，但是这种方式需要已知类别的密集注释，同时可以扩展分割到未见类别通过与文本合作。**另外一种方式**考虑到大视觉语言模型在理解物体定位上有一定的缺陷，所以使用额外的分组机制扩展这些模型，可以仅只用图像级描述，而不需要掩码监督，但是文章认为这需要大量额外的对比训练；另外，这些方法在设置阈值分类背景时，阈值设置也是一个挑战性任务。

# 2.主要贡献

所以针对上面的这个问题，文章提出使用一组冻结的模型来解决分割任务，不需要额外的数据甚至微调。

具体来说，文章引入了*OVDiff*，<mark>该模型将现有的一些基准模型变为一个图像分割器工厂，也就是使用这些基准模型按照需求合成一个分类器，从而适用任何用自然语言描述的新概念</mark>。

***OVDiff* 包含三步：生成，表示和匹配**。生成：给定一个文本提示，*OVDiff* 使用现成的图像到文本生成器（例如文章提到的StableDiffusion）生成一组支持图像。表示：在表示阶段使用特征提取器提取表示文本类别的特征原型。匹配：使用简单的最近邻域匹配机制，利用上一步得到的特征原型来分割目标图像。

文章认为，<mark>大规模文本到图像生成模型可以帮助弥合视觉与语言之间的差距，而无需注释或昂贵的训练</mark>。此外，**扩散模型还能生成具有语义意义和良好定位的潜在空间**。这就解决多模态嵌入很难学习的问题，而且往往存在模态之间的模糊性和细节差异。

原文总结贡献如下：

1. 我们介绍了一种使用预训练扩散模型完成开放词汇分割任务的方法，该方法无需额外数据、掩码监督或微调。
2. 我们提出了一种处理背景的原则性方法，即通过文本到图像生成模型中内置的上下文先验形成原型。
3. 一套进一步提高性能的附加技术，如多重原型、类别过滤和 “东西 ”过滤。

# 3.实现流程

下图展示了文章提出方法的架构图。
整体来说，就是按照前面提到的三个步骤，第一步是根据自然语言描述，利用生成模型生成一组图像；之后就是获取表征，这里文章提到学习的类别表征，同时包含正负样本对（但是这个地方，不太理解的是，负样本对是怎么生成的）；最后利用上一步学习到的特征提取器，完成图像分割。

关于最后一步，大概是前面通过正负样本对，模型学习到了什么是前景 什么是背景信息；因此，在后面提取特征后，通过过滤器，可以通过余弦相似度计算图像与原型之间的分数，从而得到分割图。

![模型架构图](https://pic1.imgdb.cn/item/6829704958cb8da5c8fa3a0c.png)

# 4.实现细节


# 5.模型性能


# 6.改进/挑战/问题/想法

* **想法**：主要关注文章的*idea*，看可不可以迁移，但是这篇文章也没有提供代码。*idea* 就是上面提到的三步，不过感觉在视频领域不太适合，感觉看文章描述效果还是不错的。

