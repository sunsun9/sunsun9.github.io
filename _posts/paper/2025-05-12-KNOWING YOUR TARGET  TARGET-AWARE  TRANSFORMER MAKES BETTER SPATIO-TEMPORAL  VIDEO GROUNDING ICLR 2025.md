---
layout: post
title: 'KNOWING YOUR TARGET  : TARGET-AWARE  TRANSFORMER MAKES BETTER SPATIO-TEMPORAL  VIDEO GROUNDING ICLR 2025😊'
subtitle: '清楚你的目标：目标感知的Transformer实现更好的时空视频定位'
date: 2025-05-12
author: Sun
cover: 'https://pic1.imgdb.cn/item/6820004958cb8da5c8eb98a5.png'
tags: 论文阅读
---

> [KNOWING YOUR TARGET  : TARGET-AWARE  TRANSFORMER MAKES BETTER SPATIO-TEMPORAL  VIDEO GROUNDING](https://openreview.net/forum?id=WOzffPgVjF)

> 💐💐[提供代码](https://github.com/HengLan/TA-STVG)
> 
> 📌作者单位
> 
> 1. 中科大
> 2. 科软
> 3. La Trobe University（澳大利亚）
> 4. University of North Texas（美国）
> 5. Brookhaven National Laboratory（美国）

# 1.文章针对痛点

这篇文章关注的问题是时空视频定位任务，文章提到目前该任务的一个痛点就是很多基于*transformer* 的方法，都是使用的一组时间和空间查询用于时空定位，但是**很多方法在初始化这些查询时，选择使用全0来初始化**。文章认为这种方法虽然也取得了很好的效果，但是**这种初始化方法缺乏目标相关的线索，因此在复杂场景从交互中学习判别性目标信息是困难的**。

# 2.主要贡献

所以针对上面的这个问题，文章提出能够生成与目标相关的初始化查询，让查询从一开始就是目标是什么或者是说知道要学习什么。因此，文章提出了新的架构，*Target-Aware Transformer for STVG (TA-STVG)*；该模型架构需要自适应初始化物体查询，通过从视频中探索获取具体目标的线索。

这个架构主要依赖于两个模块：*text-guided temporal sampling (TTS)* 和 *attribute-aware spatial  activation (ASA)*。<mark>前者</mark>，从时间角度上，首先是利用文字表达的整体信息从视频中选择与目标相关的帧；为了增强时间选择能力，*TTS* 同时考虑了外观和运动线索。<mark>后者</mark>，从空间角度上，重点从先前*TTS* 学习得到的目标感知时间线索中，探索获取物体细粒度视觉语义信息。

原文总结贡献如下：

1. 我们提出了 *TA-STVG*，这是一种新型目标感知变换器，可通过探索对象查询的特定目标线索来改进 *STVG*；
2. 我们提出了文本引导的时间采样（*TTS* ），用于从视频中选择目标相关的时间线索；
3. 我们提出了属性感知空间激活（*ASA* ），用于利用细粒度视觉语义属性信息生成目标查询；
4. *TA-STVG* 在大量实验中取得了新的一流性能，并显示出良好的通用性，证明了其功效。

# 3.实现流程

下图展示了模型的架构图，也是经典的编码器-解码器结构，与其他方法不同的地方在于，首先视觉特征提取了外观特征和运动特征两个方面；另外，中间的部分就是文章的重点部分；最后就是使用了两个解码器分别解码时间位置和空间位置。
![模型架构图](https://pic1.imgdb.cn/item/682006e758cb8da5c8eb99f7.png)

# 4.实现细节

* **多模态编码器**：这个模块就是文章的编码器部分，<mark>该模块的目的是获得视觉语言特征以便于后续的目标感知查询生成和用于定位的解码部分</mark>。该模块包含视觉和文本的特征提取，以及多模态特征融合。1️⃣视觉和文本特征提取。这个部分就是常见的使用预训练的模型进行特征提取。**对于视觉特征提取**，首先采样$$N$$帧，之后使用*ResNet101* 和*VidSwin* 分别提取外观特征和运动特征。**对于文本特征提取**，首先使用分词器分词，之后使用*RoBETRa* 提取文本特征。2️⃣多模态特征融合。文章采取的做法是将前面获得的三部分特征，**先**进行连接操作，如下公式1所示，其中$$f_i$$表示第$$i$$帧的特征，所有的帧的特征记作$$F$$；**之后**，对连接后的特征执行自注意力操作，最后得到融合后的多模态特征，如下公式2所示。
  
  $$
  f_{i}=[\underbrace{f_{i_{1}}^{a}, f_{i_{2}}^{a}, \ldots, f_{i_{H \times W}}^{a}}_{\text {appearance features } f_{i}^{a}}, \underbrace{f_{i_{1}}^{m}, f_{i_{2}}^{m}, \ldots, f_{i_{H \times W}}^{m}}_{\text {motion features } f_{i}^{m}}, \underbrace{f_{1}^{t}, f_{2}^{t}, \ldots, f_{N_{t}}^{t}}_{\text {textual features } \mathcal{F}_{t}}]	\quad(1)
  $$
  
  $$
  \tilde{\mathcal{F}}=\operatorname{SelfAttEncoder}\left(\mathcal{F}+\mathcal{E}_{p o s}+\mathcal{E}_{t y p}\right) 	\quad(2)
  $$
* **文本引导的时间采样（*TTS* ）** ：该模块的目的是生成特定目标的物体查询，能够在整体文本特征的引导下，识别和采样与目标相关的帧；简单来说，该模块的做法就是，预测每一帧与文本的相关性分数，然后根据相关性分数基于一定的阈值采样视频帧。关注这部分，文章给出了一个示意图，如下。具体来说，文章**首先**对前面得到的多模态特征进行解连接操作，得到外观特征$$\tilde{\mathcal{F}}_{a}$$、运动特征$$\tilde{\mathcal{F}}_{m}$$和文本特征$$\tilde{\mathcal{F}}_{t}$$。**之后**分别对这些特征进行平均池化操作，如下公式1所示。对得到的这些特征进行两个交叉注意力操作，**最后**送入$$MLP$$中分别得到两种视频特征与文本的相关性分数，整个过程可以参考公式2；将两个相关性分数按照公式3的方式计算得到最终的相关性分数。**之后**，根据相关性分数采样最开始的视频采样帧特征，如下公式4所示。![TTS示意图](https://pic1.imgdb.cn/item/68215a2d58cb8da5c8ecb31a.png)


$$
\mathcal{G}_{a}=\operatorname{AvgPooling}\left(\tilde{\mathcal{F}}_{a}\right) \quad \mathcal{G}_{m}=\operatorname{AvgPooling}\left(\tilde{\mathcal{F}}_{m}\right) \quad \mathcal{G}_{t}=\operatorname{Avg}\left(\tilde{\mathcal{F}}_{t}\right)	\quad(1)
$$

$$
s_a=\mathrm{MLP}(\mathrm{CA}(\mathrm{CA}(\mathcal{G}_a,\mathcal{G}_t),\mathcal{G}_t))\quad s_m=\mathrm{MLP}(\mathrm{CA}(\mathrm{CA}(\mathcal{G}_m,\mathcal{G}_t),\mathcal{G}_t))	\quad(2)
$$

$$
s=\delta\times s_a+(1-\delta)\times s_m	\quad(3)
$$

$$
\mathcal{R}_{a}=\mathrm{sample}(\tilde{\mathcal{F}}_{a},s,\theta)=\{\tilde{\mathcal{F}}_{a}(i)|s(i)>\theta\}\\
\mathcal{R}_{m}=\mathrm{sample}(\tilde{\mathcal{F}}_{m},s,\theta)=\{\tilde{\mathcal{F}}_{m}(i)|s(i)>\theta\}	\quad(4)
$$

* **属性感知空间激活（*ASA* ）** ：该模块的目的是探索获取更细粒度的目标信息，能够从先前获得的粗粒度时间特征$$\mathcal{R}_{a}$$和$$\mathcal{R}_{m}$$中，挖掘细粒度视觉语义信息。文章设计这个模块基于的直觉是，<mark>认为外观属性对于空间定位是重要的，而运动属性对于时间定位是重要的</mark>。该部分，文章也给出了示意图，如下。**具体来说**，文章**首先**会从文本特征中提取句子的主语特征（认为主语往往是需要定位的目标）。**之后**，重复这个主语特征适配视频特征的帧数。**再之后**，与前面类似，视频特征分别于文本特征进行两个交叉注意力操作。**另外**，为了增强属性感知空间特征的学习能力，文章设计使用从文本表达中生成的显式弱属性标签对 *ASA* 进行监督，使用多标签分类损失对外观和运动进行分类，这个部分如下公式1所示；通过用外观和运动属性标签对 $$c_a$$ 和 $$c_m$$ 进行监督，在交叉注意区块中作为查询的主体特征$$\mathcal{T}_{t}$$ 能够自适应地从 $$\mathcal{R}_{a}$$和 $$\mathcal{R}_{m}$$中学习与属性相关的细粒度特征，以进行属性分类。最后，生成的属性特征如下公式2所示。![ASA示意图](https://pic1.imgdb.cn/item/68215cfa58cb8da5c8ecc293.png)
  
  $$
  c_{a}=\mathrm{Softmax}(\mathrm{Linear}(\mathrm{CA}(\mathrm{CA}(\mathcal{T}_t,\mathcal{R}_a),\mathcal{R}_a)))\\
  c_{m}=\mathrm{Softmax}(\mathrm{Linear}(\mathrm{CA}(\mathrm{CA}(\mathcal{T}_t,\mathcal{R}_m),\mathcal{R}_m)))	\quad(1)
  $$
  
  $$
  \mathcal{A}_a=\mathcal{M}_a\otimes\mathcal{R}_a\\
  \mathcal{A}_m=\mathcal{M}_m\otimes\mathcal{R}_m 	\quad(2)
  $$
* **生成物体查询** ：在经过前面两个模块操作之后得到的$$\mathcal{A}_{a}$$和$$\mathcal{A}_{m}$$，被用于生成时间和空间的物体查询。具体来说，首先对两个特征进行平均池化操作，然后运用重复操作；整个过程可以如下公式所示。

$$
\mathcal{Q}_0^s=\mathrm{repeat}(\text{AvgPooling}(\mathcal{A}_a),N_v)
$$

$$
\mathcal{Q}_0^t=\mathrm{repeat}(\text{AvgPooling}(\mathcal{A}_m),N_v)
$$

* **解码器部分**：这部分其实就是很正常的解码器部分，文章也没有特别介绍。不过需要注意的是，文章对时间定位和空间定位分别使用了两个解码器，之后，解码器输出送入$$MLP$$中用于得到最后的预测结果。

# 5.模型性能

对比看了一下这篇文章与*CVPR2024* 的那篇文章，在*VidSTG* 这个数据集上，整体效果上来看，*CVPR2024* 这篇文章的效果要好一些；而在*HCSTVG* 这个系列的数据集上的效果，整体来看这篇文章要好一些。那说明这篇文章在处理长视频或者说视频长度变化较大的情况下效果不是很好，因为*HCSTVG* 视频都是20s，而*VidSTG* 视频长度更多样性一点。

损失函数参考:

$$
\mathcal{L} = \underbrace{\lambda_{\text{TTS}}(\mathcal{L}_{\text{BCE}}(s_a, (\mathcal{H}_s^*, \mathcal{H}_e^*)) + \mathcal{L}_{\text{BCE}}(s_m, (\mathcal{H}_s^*, \mathcal{H}_e^*)))}_{\text{loss of TTS}} + \underbrace{\lambda_{\text{ASA}}(\mathcal{L}_{\text{BCE}}(c_m^*, c_m) + \mathcal{L}_{\text{BCE}}(c_a^*, c_a))}_{\text{loss of ASA}} + \\\underbrace{\lambda_k(\mathcal{L}_{\text{KL}}(\mathcal{H}_s^*, \mathcal{H}_s) + \mathcal{L}_{\text{KL}}(\mathcal{H}_e^*, \mathcal{H}_e))}_{\text{loss of temporal decoder}} + \underbrace{\lambda_l\mathcal{L}_1(\mathcal{B}^*, \mathcal{B}) + \lambda_u\mathcal{L}_{\text{IoU}}(\mathcal{B}^*, \mathcal{B})}_{\text{loss of spatial decoder}}
$$

实验性能效果对比图：![效果对比图1](https://pic1.imgdb.cn/item/682160af58cb8da5c8ecc60b.png)
![效果对比图1](https://pic1.imgdb.cn/item/6821616d58cb8da5c8ecc656.png)

# 6.改进/挑战/问题/想法

* **想法**：这篇文章其实我觉得没有说有很大的创新性，因为这种模式也不是说没有文章提出来，也就是怎么初始化解码器查询。在*CVPR2024* 这篇文章中，并不是使用的全0的初始化查询，也是使用了一个文本引导的查询初始化模块，所以，我觉得这篇文章其实并没有说很大的创新性，并且只是看架构，我感觉这篇文章要比*CVPR2024* 那篇文章的计算量要大。另外，实现的效果也并没有说差别很大。

