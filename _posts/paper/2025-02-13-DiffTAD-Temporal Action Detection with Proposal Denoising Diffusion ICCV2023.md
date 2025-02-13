---
layout: post
title: 'DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion ICCV2023'
subtitle: 'DiffTAD: 具有提案去噪扩散的时序动作检测'
date: 2025-02-13
author: Sun
cover: 'https://pic1.imgdb.cn/item/67ac50add0e0a243d4fe9138.png'
tags: 论文阅读
---

> [DiffTAD: 具有提案去噪扩散的时序动作检测](https://openaccess.thecvf.com/ICCV2023)
> 1 英国萨里大学 CVSSP 2 英国科大讯飞-萨里人工智能联合研究中心 3 英国萨里以人为本人工智能研究所 4 英国伦敦帝国理工学院

# 1.文章针对痛点

这篇文章关注的时序动作检测问题*TAD*。

这篇文章并没有说针对什么问题，而是从一种新的角度来解决*TAD* 任务。现有的解决*TAD* 任务的方法大多都是基于判别式的方法，但是这篇文章从一种生成式的角度，利用去噪扩散模型来解决*TAD* 任务。

> 去噪扩散模型：这种方法常用于重建任务中，例如目前比较热门的*3D* 重建任务、图像生成、视频生成等等，在*TAD* 任务中还没有应用。<mark>这个方法的核心思想是在训练阶段，将*ground-truth* 逐步添加高斯噪声，在后续采用逆过程（也就是去噪），从而让模型能够学习到关键特征；而在推理阶段，一般是给一个随机高斯噪声分布，经过模型推理得到最后预测结果。</mark>也就是说这个去噪扩散是适用于回归性的任务的，而在*TAD* 任务中，得到动作实例的开始和结束时间本身就是一个回归任务。

但是要整合现有的*TAD* 检测模型与去噪扩散模型，还是存在一定的难点的：

* *TAD* 任务本身具有独特的挑战性，即**边界模糊性大**。首先这是由于动作本身是连续的，并不具有清晰的开始和结束时间；并且动作之间的过渡往往是随机的（也就是说在两个动作过渡之间，往往可以采取多种过渡动作形式）。另一方面，人类对于动作边界的感知是不同的。
* **去噪扩散模型和动作检测都是低效的**，二者结合的话可能会更严重。

# 2.主要贡献

因此，为了解决上述问题，文章提出了一种新的提案去噪扩散方法，可以用于以扩散公式的方式高效解决*TAD* 任务。

文章的主要贡献就是**将去噪扩散模型结合到了*TAD* 任务中**；其次，在推理过程中，为了提高效率，文章**提出了一个跨时间步选择条件机制**。这个机制包含两个关键部分：（1）通过过滤那些远离训练中生成的损坏提案分布的提案，最小化每个采样步骤中的中间预测的冗余度；（2）通过选定的提案调节下一个采样步骤，以调节扩散方向，从而实现更准确的推理。

原文将文章贡献总结为以下内容：

* 首次在*Transformer* 解码器框架中通过去噪扩散来制定时间动作检测问题。此外，将去噪扩散与此解码器设计相结合解决了典型的慢收敛限制。
* 通过在推理过程中引入一种新颖的选择性调节机制，进一步提高了扩散采样效率和准确性。
* 在*ActivityNet* 和*THUMOS* 基准上进行的大量实验表明，我们的*DiffTAD* 与现有技术替代方案相比取得了良好的性能

# 3.实现流程

还是先看一下整体的架构图![整体的架构图](https://pic1.imgdb.cn/item/67ad8d25d0e0a243d4fec7a3.png)

这个与判别式的架构不同，文章提出的*DiffTAD* 架构，在训练阶段分为两个主要的部分，分别是图片的上半部分-**生成噪声**；以及图片的下半部分-**去除噪声**，得到时间间隔预测。

其中训练过程伪代码可以表示如下：

```
def train(video_feat, gt_proposals):

	# Encode image features
	feats = video_encoder(video_feat)

	# Signal scaling
	pb = (pb * 2 - 1) * scale

	# Corrupt gt_proposals
	t = randint(0, T) # time step
	eps = normal(mean=0, std=1) # noise: [B, N, 2]
	pb_crpt = sqrt( alpha_cumprod(t)) * pb + sqrt(1 - alpha_cumprod(t)) * eps

	# Project to continuous embedding
	pb_crpt = project(pb_crpt) # query : [B, N, D]

	# Calculate Self-condition estimate
	pb_pred = zeros_like(pb_crpt)
	if self_cond and uniform(0,1) > 0.7:
		pb_pred = decoder(pb_crpt, pb_pred, feats, t)
		pb_pred = stop_gradient(pb_pred)

	# Predict
	pb_pred = decoder(pb_crpt, pb_pred, feats, t)

	# Set prediction loss
	loss = set_prediction_loss(pb_pred, gt_proposals)

	return loss
```

推理过程伪代码可以表示如下：

```
def infer(video_feat, steps, T):
	# Encode video features
	feats = video_encoder(video_feat)

	# noisy proposals: [B, N, 2]
	pb_t = normal(mean=0, std=1)

	# noisy embeddings: [B, N, D]
	pb_t = project(pb_t)
	pb_pred = zeros_like(pb_t)

	# uniform sample step size
	times = reversed(linespace(-1, T, steps))

	# [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
	time_pairs = list(zip(times[:-1], times[1:])

	for t_now, t_next in zip(time_pairs):
		# Predict pb_0 from pb_t
		if not self_cond:
			pb_pred = zeros_like(pb_t)
		pb_pred = decoder(pb_t, pb_pred, feats, t_now)

	# Estimate pb_t at t_next
	pb_t = ddim_step(pb_t, pb_pred, t_now, t_next)

	return pb_pred
```

# 4.实现细节

* **生成噪声阶段**：在训练期间，模型通过逐步向$$ground\_truth$$中添加高斯噪声，最后得到一个纯噪声分布。添加噪声的过程可以定义为下面公式所示，$${z}_{t}$$表示添加噪声之后的状态，$${z}_{0}$$表示未添加噪声的状态，在训练期间也就是$$ground\_truth$$表示的真实的时间间隔，$${\alpha}_{t}$$表示添加噪声的尺度。（注：生成噪声阶段仅训练期间进行）

$$
q\left(\boldsymbol{z}_{t} \mid \boldsymbol{z}_{0}\right)=\mathcal{N}\left(\boldsymbol{z}_{t} \mid \sqrt{\bar{\alpha}_{t}} \boldsymbol{z}_{0},\left(1-\bar{\alpha}_{t}\right) \boldsymbol{I}\right)
$$

* **去除噪声阶段**：在得到了时间提案的纯噪声分布之后，在该阶段，需要逐步去除噪声，得到预测的时间提案。具体来说，在去除噪声阶段，模型将视频特征也作为附加条件加入到去噪过程中。
  
  对于加入的视频数据的处理与判别式方法的处理是一样的，先使用$$backbone$$提取视频特征，之后使用编码器进行视频特征加工处理。在文章中，选择了$$RGB$$和光流特征，二者的关注重点也不同（前者关注$$appearance$$，后者关注$$motion$$）；因此，模型选择**后融合两种视频特征**，即使用两个分支处理特征后，再进行模态融合。这个处理架构图如下：![处理架构图](https://pic1.imgdb.cn/item/67ad92d7d0e0a243d4feca70.png)
  
  从架构图中可以看到再去噪过程中，还有一个选择条件过程***Cross-timestep selective conditioning***。具体来说，首先计算当前时间步的$$N$$个查询与上一个时间步$$N$$个查询之间的相似度矩阵$$A$$，并计算得到下面公式1；之后，在两个连续的时间步之间构建一个基于$$IoU$$的矩阵$$B$$，并计算得到下面公式2；最后按照公式3得到最终的查询集合。

$$
\hat{P}_{sim}=\{(i,j)|A[i,j]-\gamma_1>0\},	(1)
$$

$$
\hat{P}_{iou}=\{(i,j)|B[i,j]-\gamma_2>0\}, 	(2)\\
Q_{c}=(\hat{P}_{iou}/\hat{P}_{sim})\bigcup Q,	(3)
$$

这个过程的架构图表示如下：![选择条件过程](https://pic1.imgdb.cn/item/67ad93c1d0e0a243d4fecad5.png)

# 5.模型性能

实验结果比较：![实验结果比较](https://pic1.imgdb.cn/item/67ada02fd0e0a243d4fecf52.png)

从这个结果来看，这个模型的效果比不上同期的*CVPR2023 TriDet*那篇文章的效果，但是我觉得这个想法还是挺好的。

# 6.改进/挑战/问题

* **问题**：这篇文章的这个条件选择策略其实没太明白是要选择什么样的查询，可能应该是要选择与上一个时间步相似度大且*IoU*重叠高的查询。
* **改进**：个人觉得这个架构还是挺清晰的，目前*CV*方向大多都结合各种大语言模型或者多模态结合，可以考虑在这篇文章的基础上是不是也可以考虑加入文本模态的方式呢。

