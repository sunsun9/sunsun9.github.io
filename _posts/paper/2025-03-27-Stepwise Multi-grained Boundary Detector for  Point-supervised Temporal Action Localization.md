---
layout: post
title: 'Stepwise Multi-grained Boundary Detector for  Point-supervised Temporal Action Localization ECCV 2024'
subtitle: '用于点监督时间动作定位的逐步多粒度边界检测器'
date: 2025-03-27
author: Sun
cover: 'https://pic1.imgdb.cn/item/67e38dcd0ba3d5a1d7e37b96.png'
tags: 论文阅读
---

> [Stepwise Multi-grained Boundary Detector for  Point-supervised Temporal Action Localization](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01159.pdf)

> ❌❌不提供代码
> 
> 📌作者单位
> 1.人机混合增强智能国家重点实验室、视觉信息与应用国家工程研究中心、西安交通大学人工智能与机器人研究所
> 2.DeepNight.ai
> 3.杜比实验室多模态体验研究实验室

# 1.文章针对痛点

这篇文章关注的是点监督时间动作定位，先说明一下什么是**点监督**。点监督是指在训练过程中需要每个未剪裁视频的一些动作实例标签，但是仅仅需要每个动作实例的一个随机注释片段即可，也是就可以获取的动作实例标签是单帧的或者是少量帧的。

而文章之所以关注这个任务是**考虑到全监督和弱监督的不足**。全监督因为要使用到大量的标记数据，这个过程是费时费力的；而弱监督仅有视频级注释而缺乏实例级注释，导致在区分动作实例和边界时较为棘手。所以文章选择关注位于二者之间的点监督。

但是**点监督也存在一定的问题**，在高度稀疏的点监督模式下，由于缺乏连续性，个体帧的稀疏标签会导致在检测动作边界时出现语义模糊性。针对这个问题，目前提出了一些解决方法，主要是通过迭代细化和弱监督风格两种方法。前者是交替出现预测和应用伪标签，但是这种方式严重依赖经验阈值的设置；后者是通过将点标签作为强动作级类别标签，这种方式容易产生不完整结果。

# 2.主要贡献

所以针对上面的问题，文章提出了一种新的架构，**称为*Stepwise Multi-grained  Boundary Detector (SMBD)***。该架构包含两部分，分别是**边界锚框生成器*BAG* 和双边检测器*DBD***。该架构的目的是首先检测视频中可靠的边界帧，之后在边界帧和动作标签之间搜索最优动作边界。

具体来说，首先引入**边界锚框生成器**来定位每对动作标签之间高置信度的边界帧，通过不同的分类器头投票。之后，文章认为动作边界一定位于每个相邻边界锚框和动作标签之间，基于这个想法，文章提出了**双边检测器**来定位动作边界，通过检测来自相邻边界锚框和动作标签之间的动作变化和背景变化。

双边检测器包含一个细粒度检测器和一个粗粒度检测器，前者通过观察动作分类分数的变化来找寻细粒度边界，后者通过评估前景和背景分数的不同来检索粗粒度边界。

原文总结贡献如下:

1. 本文通过强调仅使用稀疏点监督来学习整个动作语义，引入了一种用于*PTAL* 的分步多粒度边界检测器。它可以通过搜索动作边界来确保更多的视频帧有助于模型训练。
2. 我们提出了细粒度边界检测器和粗边界检测器，分别从检测动作变化和场景变化的互补角度定位动作边界。
3. 在*THUMOS14*、*GTEA* 和*BEARCH* 数据集上进行的大量实验验证了所提出的方法相对于现有的点监督*TAL* 方法的优越性。

# 3.实现流程

一样，还是先看一下原文展示的模型架构图：![模型架构图](https://pic1.imgdb.cn/item/67e39b8b0ba3d5a1d7e38162.png)

从架构图中可以看到首先也就是提取视频特征，后面紧接着就是边界锚框生成器，得到边界锚框后，就会送入双边边界检测器。

# 4.实现细节

* **特征提取** ：特征提取部分就不用也别说明了，就是使用预训练的模型提取特征。
* **边界锚框生成器**：这个使用了多实例学习的激活序列概念，也就是架构图中的有关变量$$p_i$$的曲线图。具体来说，文章使用了多个分类器来评估每个视频片段作为背景锚框的置信度。也就是通过多个分类器，可以得多个具体类别的激活分数$$p$$；根据这些激活分数可以获得，背景可能性$$p^{bkg}$$和方差$$\sigma$$（前者数值越大表示背景片段的可靠性越大，后者数值越小表示背景分类随机性更小，因此我们想要的目的就是前者的数值越大，后者的数值越小）。根据得到的前面两个变量来计算稳定性如下公式1，从而得到背景锚框计算公式，如下公式2。
  
  $$
  s=\overline{p}^\mathrm{bkg}/\boldsymbol{\sigma} \\
  {p}^\mathrm{bkg}= 1/n\sum_{i=1}^n\boldsymbol{p_i^\mathrm{bkg}} \quad(1)
  $$

$$
t^\mathrm{bkg}=\arg\max_t\{s\}, \quad(2)
$$

**双边检测器**：该模块认为边界一定位于相邻动作片段和边界片段之间，并将其定位为候选间隔。前面已经提到文章将该检测器分为了双支结构，分别是细粒度边界检测器和粗粒度边界检测器。1️⃣细粒度边界检测器：文章在给定的候选间隔内，计算$$\hat{t}$$是边界的不确定性，如下公式1和公式2（应该是分别是左边界和右边界的可能性），因此最后的目的就是想让这两个不确定性数值和最小，如下公式3。2️⃣粗粒度边界检测器：该分支不是像细粒度边界检测器从动作变化的角度来定位动作边界，而是文章认为动作边界也描述了前景和背景的转变，所以文章引入了该分支通过评估前景和背景分数的不同来搜寻动作边界。具体而言是计算前景和背景分数差值的均值，如下公式4和5所示。最后期望得到的结果就是两个均值越大越好，如下公式6。3️⃣边界融合，就是将前面两种边界融合即可，如下公式7所示。

$$
s_{\hat{t}}^l=\frac{1}{\hat{t}-t_i^{\mathrm{bkg}}}\sum_{t=t_i^{\mathrm{bkg}}}^{\hat{t}}\left(\left|p_t-\frac{1}{\hat{t}-t_i^{\mathrm{bkg}}}\sum_{n=t_i^{\mathrm{bkg}}}^{\hat{t}}p_n\right|\right), \quad(1)
$$

$$
s_{\hat{t}}^r=\frac{1}{t_j^\mathrm{act}-\hat{t}}\sum_{t=\hat{t}}^{t_j^\mathrm{act}}\left(\left|p_t-\frac{1}{t_j^\mathrm{act}-\hat{t}}\sum_{n=\hat{t}}^{t_j^\mathrm{act}}p_n\right|\right), \quad(2)
$$

$$
t_{\mathrm{FB}}=\arg\min_{\hat{t}}\left(s_{\hat{t}}^l+s_{\hat{t}}^r\right). \quad(3)
$$

$$
s_{\hat{t}}^l=\frac{1}{\hat{t}-t_i^{\mathrm{act}}}\sum_{t=t_i^{\mathrm{act}}}^{\hat{t}}\left(p_t^{\mathrm{fg}}-p_t^{\mathrm{bkg}}\right), \quad(4)
$$

$$
s_{\hat{t}}^r=\frac{1}{t_j^{\mathrm{bkg}}-\hat{t}}\sum_{t=\hat{t}}^{t_j^{\mathrm{bkg}}}\left(p_t^{\mathrm{bkg}}-p_t^{\mathrm{fg}}\right), \quad(5)
$$

$$
t_{CB}=\arg\max_{\hat{t}}\left(s_{\hat{t}}^{l}+s_{\hat{t}}^{r}\right). \quad(6)
$$

$$
t_B=\lambda t_{FB}+(1-\lambda)t_{CB}, \quad(7)
$$

* **损失函数**：损失函数包含三部分，分别是分类损失、定位损失和KL散度。需要说明的是文章通过对分数进行时间池化和阈值化来预测视频级别标签。分类损失如下公式1，定位巡视如下公式2，KL散度如下公式3，最后总损失如下公式4。
  
  $$
  \mathcal{L}_{\mathrm{cls}}=-\frac{1}{N^\mathrm{f}}\sum_{i=1}^{N^\mathrm{f}}\left(\boldsymbol{y}_i^\mathrm{p}(1-\boldsymbol{p}_i)^\beta\mathrm{log}\boldsymbol{p}_i+(1-\boldsymbol{y}_i^\mathrm{p})\boldsymbol{p}_i^\beta\mathrm{log}(1-\boldsymbol{p}_i)\right), \quad(1)
  $$
  
  $$
  \mathcal{L}_{\mathrm{act}}=-\frac{1}{N^\mathrm{f}}\sum_{i=1}^{N^\mathrm{f}}\left(\boldsymbol{y}_i^\mathrm{b}\mathrm{log}\boldsymbol{p}_i^\mathrm{b}+(1-\boldsymbol{y}_i^\mathrm{b})\mathrm{log}(1-\boldsymbol{p}_i^\mathrm{b})\right), \quad(2)
  $$
  
  $$
  \mathcal{L}_{\mathrm{KL}}=\frac{1}{N^\mathrm{f}}\sum_{i=1}^{N^\mathrm{f}}\mathrm{KL}(\boldsymbol{y}_i^\mathrm{p}||\boldsymbol{p}_i),\mathrm{KL}(\boldsymbol{y}_i^\mathrm{p}||\boldsymbol{p}_i)=\sum_{j=1}^C\boldsymbol{p}_{ij}\mathrm{log}\left(\frac{\boldsymbol{p}_{ij}}{\boldsymbol{y}_{ij}^\mathrm{p}}\right), \quad(3)
  $$

# 5.模型性能

实验性能效果对比图，可以看到虽然它的效果不如全监督的SOTA，但是相比于弱监督效果是好很多，并且与全监督相比只是仅低于少部分模型方法。![效果对比图1](https://pic1.imgdb.cn/item/67e4b4470ba3d5a1d7e473d7.png)
![效果对比图2](https://pic1.imgdb.cn/item/67e4b4c50ba3d5a1d7e47403.png)

# 6.改进/挑战/问题/想法

* **想法**：感觉点监督这种模式还是有一定的可行性的，因为点监督是会给定动作实例的随机一帧注释，也就是在训练阶段虽然不知道完整的动作实例标签，但是可以知道其中的某一帧一定是动作帧。而这篇文章是直接通过找边界的方式来定位，那或许也可能结合找动作帧来定义，考虑动作帧的邻居帧来定位动作片段。
* **问题**：

