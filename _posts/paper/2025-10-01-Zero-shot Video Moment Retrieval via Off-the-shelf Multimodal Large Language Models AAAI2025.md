---
layout: post
title: 'Zero-shot Video Moment Retrieval via Off-the-shelf Multimodal Large Language Models AAAI2025🙁'
subtitle: '通过现成的多模态大语言模型实现零样本时刻检索'
date: 2025-10-01
author: Sun
cover: 'https://pic1.imgdb.cn/item/68dcd4b3c5157e1a884c2ee7.png'
tags: 论文阅读
---

> [Zero-shot Video Moment Retrieval via Off-the-shelf Multimodal Large Language Models](https://arxiv.org/abs/2501.07972)

> ❌❌[未开源代码]
> 
> 📌作者单位
> 
> 1. 南京大学
> 2. 大连理工大学
> 3. 南京信息工程大学

# 1.文章针对痛点

这篇文章关注的是赋予时刻检索零样本设置，零样本任务的设置就是避免大规模的训练和高要求的数据集。


# 2.主要贡献

针对提出的零样本任务，文章提出了对应的解决方法和模型架构。


原文总结贡献如下：

1. 我们建议使用现成的MLLM进行直接推断的*Moment-GPT*，这是一种零样本的*VMR* 方法。
2. 我们制定了一种新的策略，以查询利用*Llama-3* 来提高性能。此外，我们设计了时间定位框生成器和时间定位框得分手，以有效利用*Minigpt-V2* 和*VideoChatgpt* 的视觉理解能力。
3. 广泛的实验结果表明，我们的方法在三个*VMR* 数据集上优于基于*MLLM SOTA* 的方法和*Zeroshot* 方法。值得注意的是，它也超过了大多数监督模型。


# 3.实现流程

文章提出模型的架构图如下所示，模型整体的架构设计不难理解，主要是用了很多现有的大语言模型辅助，从架构图上也没有看到标记了需要训练的部分，这篇文章就是不需要训练的零样本任务方法。

![模型架构图](https://pic1.imgdb.cn/item/68dcd64cc5157e1a884c3661.png)

# 4.实现细节

* **降低文本偏差模块**：这个模块设计是为了减少文本偏差带来的影响，文章考虑了文本不总是准确和常见的，可能文本存在一些的小的语法错误，虽然不影响人类理解，但是可能会给模型带来影响；此外，文本描述可能会出现一些不经常使用的词汇，对于非大语言模型是很难理解的。因此，在这个模块，文章使用了*LLaMA-3* 模型来修正文本描述，得到文本描述集$$D$$。![示意图](https://pic1.imgdb.cn/item/68dcd7c7c5157e1a884c370d.png)
* **生成候选框**：这个模块是通过*MiniGPT-v2* 为每个采样帧生成一个文本描述$$f$$，之后按照下列公式1计算上一个模块得到的修正后文本描述与刚刚为每个采样帧生成的文本描述之间的相似度。最后通过一个*Span Generator* 生成时间候选框。

$$
S^f=\cos(X^d,X^f)=\frac{X^d\cdot X^f}{\|X^d\|\|X^f\|}
$$

* **选择相关片段**：首先是使用*Video-ChatGPT* 为上一步得到的时间候选框生成片段级别的文本描述；之后，也是计算这些片段级别的文本描述与第一步得到的修正后的文本描述之间的相似性；最后，根据相似性分数选择得到最后的时间片段。

# 5.模型性能

![模型性能1](https://pic1.imgdb.cn/item/68dcdb3bc5157e1a884c393b.png)
![模型性能2](https://pic1.imgdb.cn/item/68dcdbe2c5157e1a884c398c.png)

# 6.改进/挑战/问题/想法

* **想法**：这个还是有一些参考性的在时间定位上。

