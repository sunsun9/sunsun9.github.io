---
layout: post
title: 'FineSports: A Multi-person Hierarchical Sports Video Dataset for Fine-grained Action Understanding CVPR 2024'
subtitle: 'FineSports：用于细粒度动作理解的多人分层运动视频数据集'
date: 2024-10-30
author: Sun
cover: 'https://pic.imgdb.cn/item/67219a81d29ded1a8cd68249.png'
tags: 论文阅读
---

> [FineSports：用于细粒度动作理解的多人分层运动视频数据集](https://openaccess.thecvf.com/CVPR2024)（说明：虽然这篇文章在原文中放了数据集github仓库地址，但是目前仓库是空的）

> 这篇文章的主要工作是**（1）**构建了一个具有细粒度动作注释的体育多人运动视频数据集，**（2）**提出了一种提示驱动的时空动作定位方法*PoSTAL*，包含两部分（详见下文），**（3）**大量实验证明了数据集的可行性，以及（2）中提出方法在*STAL*（时空动作定位）任务上的有效性。
> 
> 这篇文章的主要工作还是在数据集上，因为和后面实验部分只选取了２个方法进行比较。其次模型使用了多个损失项，后面两个损失项其实是很好理解的，因为这个模型想要完成的任务，但是前两个损失项文章并没有介绍为什么要设置，有什么用处。

# 摘要

由于运动员快速移动和肢体对抗激烈，容易造成**严重遮挡**，导致多人运动中的细粒度动作分析是复杂的。此外，目前的多人运动数据集**缺乏时间和空间上的细粒度动作注释**，这也**造成细粒度动作分析的困难**性。

因此，**（1）**本文构建了一个新的多人篮球运动视频数据集*FineSports*，该数据集包含10000*NBA*比赛视频上的细粒度语义和时空注释，覆盖52中细粒度动作类型，16000动作实例，123000时空边界框。**（2）**本文也提出了一种新的提示驱动的时空动作定位方法*PoSTAL*，该方法由两部分组成，分别是提示驱动的目标动作编码器*PTA*和动作特定管道检测器*ATD*。

# 引言

目前基于深度学习的视频动作理解方法都是数据驱动的，而现在大部分可用的动作视频数据集通常缺乏高质量的细粒度注释，从而导致时空以及语义关系之间的细粒度分析很困难，严重阻碍了细粒度动作理解的时空模型的发展。

相较于其他动作的理解，团体运动视频的细粒度动作理解是非常具有挑战性的，因为团体运动往往很混乱。总的来说，多人运动往往具有以下几个特点：**（1）**主体具有主动关系。主要是指人具有主观能动性。**（2）**快速改变。在体育多人运动中，进攻和防守模式的转变往往是非常迅速的。**（3）**极端的身体姿势。这些特性往往也会造成人体互相重叠严重，此外体育运动中还有很多我们并不关注的人，比如裁判员和观众，这些往往会给细粒度分析带来很多噪声。

考虑到这些，本文构建了一个新的具有细粒度注释的数据集。此外，本文还提出了一个提示驱动的时空动作定位方法，包含*PTA*和*ATD*，前者通过描述词提取目标动作特征，后者同时获取一组动作目标管道和对应的动作类别。

# 相关工作

本文从细粒度运动视频数据集和时空动作定位两方面阐述了现有工作。这两方面也是本文主要进行的两个工作。

### 细粒度运动视频数据集

现存在提供细粒度注释的运动视频数据集大致可以分为四类：识别、定位、检测和评价。本文也根据这个分类，对现有的这类数据集进行了大致的整理。相较于之前的数据，本文认为其数据集具有以下特点：（1）样本规模大，（2）动作类别多样且分层，（3）细粒度实例的数量。数据集整理如下。![数据集](https://pic.imgdb.cn/item/671f6c05d29ded1a8c28a5b9.png)

### 时空动作定位

时空定位重点关注在帧或视频级别识别视频序列目标动作发生的*where(spatial) and when(temporal)*，之前的工作通常是处理视频的每一帧并越策每一帧的边界框，最后连接作为最后的结果。**帧级定位**主要关注帧内的空间信息，通过利用基于*2D CNN*的检测网络和区域提议网络来获得检测结果。随着*3D-CNN*的发展。**视频级定位**利用*backbone*，不仅可以捕获单个帧内的空间信息，还可以对帧之间的时间动态进行建模。

本文的*PoSTAL*方法遵循视频级范式，利用文本提示精确引导学习目标动作时空特征。

# *FineSports*数据集

本文从数据集构建、数据统计和特点三个方面介绍本文构建的数据集。

### 数据集构建

运动视频是从*NBA*官方回放库中收集，经过处理后，包含10000个视频，并且只保留了从比赛场地的俯视镜头。

本文根据数据情况将动作分为了12个大类和52个子类。并为数据集构建了词汇表，词汇表由细粒度语义和时空架构两部分组成，每一个记录都包含两层注释。**（1）语义架构。**动作级标签描述了运动员有效动作过程的粗粒度动作标签，包含12个类别；步骤级标签描述了运动过程中，程序步骤的细粒度子类类型。**（2）时空架构。**数据集提供了每一个持球运动员动作执行过程中的**子类空间位置和时间边界**。在空间维度，步骤级标签是每帧红所有运动员的边界框；在时间维度，步骤级标签是动作过程中，子动作的开始和结束帧。
语义架构图![语义架构图](https://pic.imgdb.cn/item/6720901bd29ded1a8c050ccd.png)
时间架构图![时间架构图](https://pic.imgdb.cn/item/67209076d29ded1a8c056d97.png)

数据集的注释处理过程分类空间注释处理和时间注释处理。**空间注释处理**是利用*MixSort-OC*模型，跟踪每个视频中所有运动员，来获取运动员在赛场上的边界框，在视频中，每个运动员都有一个唯一的*ID*。时间注释处理时基于词汇表，注释每个视频中持球者有效动作片段的时间边界。

### 数据集统计

数据集包含10000个视频样本，12个动作类别和52个子动作类别，平均每个视频长11.74s，细粒度子类别分布详见下图，数据集具体信息详见前文*Table1*。
![细粒度子类别分布](https://pic.imgdb.cn/item/672091c5d29ded1a8c06cfa9.png)

### 数据特点

*Table1*详细记录了*FineSports*数据集和其他数据集的对比以及相关的分类情况。

# PoSTAL

*PoSTAL*方法将时空动作定位任务公式化为一个多任务学习问题，输入一个视频片段，输出每个目标运动员的动作管道和对应的细粒度动作类。其中，动作管道由目标运动员的一系列边界框组成，包括正在执行的目标动作的空间位置和时间边界。整体公式如下，其中*Y*表示*N*个动作管道，且每个管道包含*T*帧；*y*表示*N*个管道对每个类别的预测；*P*表示提示驱动的目标动作编码器*PTA*，*D*表示动作特定管道检测器*ATD*。
![整体公式](https://pic.imgdb.cn/item/67209aefd29ded1a8c0fec66.png)

*POSTAL*包含两部分，分别是*PTA*和*ATD*。

* *PTA*：该模块通过利用一个时空视觉语言交叉注意，来准确学习目标动作的表征（由目标运动员的外貌特点和对应的细粒度子类型来引导）。（1）首先在提示嵌入空间编码*text*和目标运动员细粒度子动作类型（这个类型标签是由视频特征编码得到的）。（2）合并提示嵌入空间和视频特征，从而学习提示驱动的目标动作表征。在这个部分分使用了一个多头交叉注意力模块。
* *ATD*：在获取提示驱动的目标动作表征之后，设计了*ATD*模块来预测目标动作的空间位置、时间边界和细粒度子动作类型。*ATD*模块包含两个块：一个单层动作特定管道*transformer*和一个多层动作特定管道*transformer*，其中，前者是 在时间和空间上定位每个目标动作，后者识别细粒度类型，二者并行进行。
  整体架构图如下，其中，*Epos*是一个位置嵌入，*TQ*是一组可学习的动作管道查询，近似*3D*长方体；*g*是一个*MLP*模块。（至于单层、多层名称和架构图中网络层数不一致，这个不太清楚）![整体架构图](https://pic.imgdb.cn/item/6721964fd29ded1a8cd2f6f0.png)

最后损失函数![损失函数](https://pic.imgdb.cn/item/672197c3d29ded1a8cd43d50.png)

# 实验

实验这个部分选取了两个*SOTA*方法进行了比较。消融实验（1）验证不同*PTA*设置的有效性。主要是验证描述词和可学习嵌入的有效性。（２）验证不同管道查询数量的有效性。

