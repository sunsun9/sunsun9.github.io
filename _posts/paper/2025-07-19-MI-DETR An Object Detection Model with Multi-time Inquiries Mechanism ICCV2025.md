---
layout: post
title: 'MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism ICCV2025'
subtitle: 'MI-DETR: 具有多次查询机制的物体检测模型'
date: 2025-07-19
author: Sun
cover: 'https://pic1.imgdb.cn/item/687b4b6e58cb8da5c8c6accc.png'
tags: 论文阅读
---

> [MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](https://arxiv.org/abs/2503.01463)

> ❌❌[未提供仓库]
> 
> 📌作者单位
> 
> 1. 重庆大学
> 2. 清华大学

# 1.文章针对痛点

这篇文章关注的是物体检测，主要是认为目前常用的*DETR-like* 方法存在一定的缺陷，即采用了多层编码的结构，但是在解码器的时候往往仅使用最后一层的编码特征，认为这样可能会忽略一些底层的有用信息。所以文章针对这个问题进行了改进。

# 2.主要贡献

具体来说，文章提出了一种多次查询的结构，在解码器的每一层并不是简单使用编码器的最后一层特征或者使用编码器对应层的特征，而是将对应层和最后一层的特征*concat* 一下输入到对应的解码器层。

另外，查询也进行了一些特殊的处理，具体可以参考文章结构图。


# 3.实现流程

下图展示了文章提出方法的大致流程图。现有的*DETR-like* 结构，当前层的查询往往是使用上一层的结果初始化当前层的查询，但是文章使用了一种融合的方式得到当前层的初始化查询。

![模型架构图](https://pic1.imgdb.cn/item/687b4e5358cb8da5c8c6b45f.png)

# 4.实现细节

具体实现细节参考原文。

# 5.模型性能


# 6.改进/挑战/问题/想法

* **想法**：刚好与自己的方向有些相关，参考一下修改思路。

