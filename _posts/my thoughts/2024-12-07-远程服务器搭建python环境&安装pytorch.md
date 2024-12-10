---
layout: post
title: '远程服务器搭建python环境&安装pytorch'
date: 2024-12-07
author: Sun
cover: ''
tags: 个人杂记
---

因为组里服务器重装了系统，之前的环境都删除了。所以自己重新搭建了一遍环境，之前因为是公用账号，不需要自己下载conda，只需要配置自己的虚拟环境即可。

自己配置conda环境，可以参考[【Conda】超详细的linux-conda环境安装教程_linux安装conda-CSDN博客](https://blog.csdn.net/Alex_81D/article/details/135692506)，这篇文章详细介绍了怎么下载Anaconda，如果没有管理员权限的话，也提供了配置环境变量的方式。（这篇我没有管理员权限，也是成功安装了，只是后面下载pytorch速度太慢了）

下载pytorch，如果不配置国内镜像的话，下载速度很慢，但是个人没有管理员权限，因此不能使用vim命令（不知道为啥，也没法安装）。[linux ubuntu安装pytorch（深度学习环境搭建记录，无sudo权限）踩坑全记录_gpu服务器 没有sudo权限-CSDN博客](https://blog.csdn.net/triayuzu/article/details/124046522)，这篇文章提供了没有管理员权限怎么下载，以及虽然不能使用vim命令，但是这篇文章作者说可以用gedit命令添加镜像，但是我试了，也不可以。

[这篇文章](https://blog.csdn.net/nanguxiaosheng/article/details/129287923)最后成功添加镜像，直接在命令行输入提供的三个conda命令即可。和这篇文章的作者一样，配置了镜像但是使用conda命令下载仍然很慢，最后使用的pip命令。

最后希望再也没有装不完的环境！

