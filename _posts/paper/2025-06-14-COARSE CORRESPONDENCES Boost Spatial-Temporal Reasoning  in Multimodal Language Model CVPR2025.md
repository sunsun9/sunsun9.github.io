---
layout: post
title: 'COARSE CORRESPONDENCES Boost Spatial-Temporal Reasoning  in Multimodal Language Model CVPR2025😊'
subtitle: 'COARSE CORRESPONDENCES促进多模态语言模型中的时空推理'
date: 2025-06-14
author: Sun
cover: 'https://pic1.imgdb.cn/item/684ce57358cb8da5c84b3c37.png'
tags: 论文阅读
---

> [COARSE CORRESPONDENCES Boost Spatial-Temporal Reasoning  in Multimodal Language Model](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Coarse_Correspondences_Boost_Spatial-Temporal_Reasoning_in_Multimodal_Language_Model_CVPR_2025_paper.pdf)

> 💐💐[提供代码](https://coarse-correspondence.github.io/)
> 
> 📌作者单位
> 
> 1. 华盛顿大学
> 2. 清华大学
> 3. 腾讯
> 4. Allen Institute for AI
> 5. Cornell University

# 1.文章针对痛点

这篇文章关注的是多模态语言模型，文章认为虽然当前多模态语言模型取得了不错的进展，但是多模态语言模型在时空推理上仍然存在着很大的问题。

当前的模型为了增强多模态语言模型的时空理解能力，大多数方法都是分开增强其中的某一项推理能力。例如，为了增强多模态语言模型的空间理解能力，现在的研究方法主要又三种，提供3D数据作为模型的输入或者为3D任务设计专用的架构或者对3D数据采用监督微调的方式。在增强时间理解能力上，也是采用了类似的方式。

# 2.主要贡献

所以针对上面的这个问题，文章提出了 *COARSE CORRESPONDENCES*，这是一种简单而有效的免训练视觉提示方法，可联合增强 *MLLM* 中的空间-时间推理能力。*COARSE CORRESPONDENCES* 使用跟踪模型提取多张图像中的对象级对应关系，然后通过视觉提示在图像上表示最突出的对应关系。

文章之所以称这种方法为 “粗对应”，是因为以下几点： 1. 只对实例级对应关系而非点级对应关系进行视觉提示。2. 使用现成的跟踪模型提取实例级对应关系。3. 我们只可视化少数突出的对应实例。


# 3.实现流程

下图展示了文章提出方法的大致流程图。文章所提出的*COARSE CORRESPONDENCES* 包含4个步骤，分别是：追踪对应关系，梳理帧，选择，可视化粗粒度对应关系。

![模型架构图](https://pic1.imgdb.cn/item/684d0c5058cb8da5c84b59e3.png)

# 4.实现细节

* ***Tracking correspondences*** ：这个模块就是使用离线的视频对象追踪模型，将给定图像中的所有对象使用掩码标记出来。
* ***Sparsify frames*** ：这个步骤是考虑到计算量的问题。为了避免直接使用所有图像导致的计算量巨大的问题，文章对时间使用了下采样，避免计算量大的问题。
* ***Selecting coarse correspondences*** ：如果将前面的信息全部作为提示提供给多模态语言模型，可能会产生信息超负荷问题，所以在这一步，文章通过选择一些最突出的实例对象，提供给多模态语言模型。
* ***Visualizing coarse correspondences*** ：这个步骤就是将第三步选择的对象进行一个标记，并覆盖在原图像的上面，就是前面流程图看到的标记一样。



# 5.模型性能

性能如下

![性能图1](https://pic1.imgdb.cn/item/684d0e5658cb8da5c84b5cba.png)
![性能图2](https://pic1.imgdb.cn/item/684d0e7558cb8da5c84b5cc6.png)
![性能图3](https://pic1.imgdb.cn/item/684d0e8c58cb8da5c84b5cd1.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章是为了了解怎么进行时空学习的，但其实发现在空间学习上，是通过外在的方式对图像进行了标记，从而辅助模型理解空间位置关系。但是这种很明确的标记方式，我比较存疑在对于模型是否真的理解空间位置关系上，其实对于模型可能只需要理解标记1和标记1是对应的，标记2和标记2是对应的，我认为更关键的可能只是学习到了标记，而不是真正的物体对应关系上。所以我认为多模态语言模型其实本质上并没有证明具有时空学习能力，而是具有标记学习能力。

