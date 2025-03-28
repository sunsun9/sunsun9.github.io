---
layout: post
title: 'Grounded Language-Image Pre-training CVPR2022'
subtitle: 'GLIP'
date: 2024-12-31
author: Sun
cover: 'https://pic1.imgdb.cn/item/67737fb5d0e0a243d4ecfcc5.png'
tags: 论文阅读
---

> 1加州大学洛杉矶分校、2微软研究院、3华盛顿大学、4威斯康星大学麦迪逊分校、5微软云和人工智能、6国际数字经济学院

简单概述一下这篇文章，这篇文章是将对象检测与短语定位任务进行了统一。与CLIP解决分类任务类似，这篇文章提出的GLIP也是在大规模的图像文本对上进行训练，也是**在图像和文本之间使用对比学习**。其次，**加强了模态融合**，文章提到发现深度的模态融合会让效果更好，因此，该方法在早期就进行了模态融合。

此外，文章提到GLIP使用与零样本任务，并且在对象检测与短语定位任务上达到了先进的性能。文章也提到，在使用GLIP迁移到其他任务上时，只需要进行检测的prompt格式调整，也可以达到在新任务上对GLIP进行全微调的效果。
![架构图](https://pic1.imgdb.cn/item/67737f30d0e0a243d4ecfcb8.png)

> 目前将文本模态结合到计算机视觉的各个子任务上已经成为了一种新的趋势，文本模态确实具有更可学的泛化能力，并且在结合计算机视觉之后，将特征对齐后，这种泛化能力是更强到的。未来，还是多结合多模态。

