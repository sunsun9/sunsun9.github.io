---
layout: post
title: 'STPro: Spatial and Temporal Progressive Learning for Weakly Supervised
Spatio-Temporal Grounding CVPR2025'
subtitle: 'STPro: 面向弱监督时空视频定位的时空渐进学习'
date: 2025-08-04
author: Sun
cover: 'https://pic1.imgdb.cn/item/689048ad58cb8da5c80321ee.png'
tags: 论文阅读
---

> [STPro: Spatial and Temporal Progressive Learning for Weakly Supervised Spatio-Temporal Grounding](https://aaryangrg.github.io/research/stpro)

> 💐💐[提供仓库地址，但是还没开源](https://aaryangrg.github.io/research/stpro)
> 
> 📌作者单位
> 
> 1. BITS Pilani
>    2.中佛罗里达大学 计算机视觉研究中心

# 1.文章针对痛点

这篇文章关注的是弱监督的视频时空定位，文章认为全监督的视频时空定位任务的缺陷是需要详细的、复杂的边界框注释，而弱监督不需要复杂全面的边界框注释，只需要自然语言的文本即可；同时，文章认为现有的弱监督方法大都是要么使用新的模态，要么使用各种复杂的模块或者多个分层算法。

# 2.主要贡献

针对上面的问题，文章提出了一种简单的结构-时空渐进学习*STPro*，不需要设计各种复杂的模块。*STPro* 包含两个渐进学习策略，分别是子动作时间课程学习和密集引导的空间课程学习；前者通过随时间逐渐增加任务难度，逐步提升模型对复杂动作的理解能力；后者关注建模逐渐稀疏的场景。

原文总结贡献如下：

1. 我们引入了时间推荐定位（*TRG* ）模块，该模块适应了弱监督的时空视频定位任务的定位模型，为处理时间和空间环境提供了强大的基线。
2. 我们提出了子动作时间课程学习（*SA-TCL* ）范式，该范式通过逐步提高动作序列的复杂性来提高模型的时间基础能力，从而更好地处理视频中的时间依赖性。
3. 我们引入了拥塞引导的空间课程学习（*CG-SCL* ），一种新颖的方法，可以逐步使模型适应更复杂而密集的场景结构，从而提高了其在挑战性的视频场景中了解空间关系的能力。

# 3.实现流程

整体的流程就是下面展示的这样，首先经过视频编码器编码，编码器之后会使用一个跟踪器算法，获取视频中每个目标对象的运动轨迹；之后先判断时间边界，去除掉一些不相关的目标对象管道；最后把剩下的未筛选掉的轨道继续与文本进行相似度计算，最后得到最终一个选择的轨迹。

![模型架构图](https://pic1.imgdb.cn/item/6890567e58cb8da5c803a2bb.png)

# 4.实现细节

文章的主要改进就是最后两部分的设计，也就是第一步时间管道的处理，和最后一步空间相似度计算。

# 5.模型性能

![模型性能1](https://pic1.imgdb.cn/item/6890579e58cb8da5c803abb3.png)
![模型性能2](https://pic1.imgdb.cn/item/6890583c58cb8da5c803b0bf.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章的思路也是可参考的，因为一些目标追踪算法肯定精度是非常高的；只需要把这些目标追踪算法得到的目标管道进行细化就可以了。看弱监督这一块精度还是比较低的。

