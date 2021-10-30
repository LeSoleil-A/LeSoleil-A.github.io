---
layout: post
title:  "笔记《Modeling Musical Context Using Word2Vec》"
date:   2021-10-14 21:00:00 +0800
categories: 音乐论文笔记
typora-root-url: ..\assets\img\1014
---



这篇文章是作为上一篇笔记[《From-Note2Vec-to-Chord2Vec》](https://lesoleil-a.github.io/%E9%9F%B3%E4%B9%90%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/2021/10/05/From-Note2Vec-to-Chord2Vec.html)的拓展阅读，上一篇笔记的工作有相当一部分是以此篇文章为基础的。

文章的主要工作即为上篇笔记所说的**Note2Vec**，即音符的嵌入，使用的仍为自然语言处理的方法，实际上，这篇文章应该是首次将 *skip-gram* 引入了和弦嵌入工作。

这篇论文有以下几个值得注意的点：

1. 将 *skip-gram with negative sampling* 引入音乐嵌入；
2. 使用 *cosine similarity* 来表示和弦之间的相似性；
3. 对于**Music slices**的处理。

<br>

### 摘要

<br>

【目标】

本文提出了一种语义上的向量空间模型，以捕获多音音乐语料的复杂语境。

【方法】

为了表示音乐片段，本文引入了一个基于 *skip-gram representation with negative sampling* 的**Word2Vec**模型。本文使用的数据集为贝多芬的钢琴奏鸣曲。

【结果与总结】

使用 *t-distributed stochastic neighbor embedding* 方法，可以得到降维后向量空间的可视化结果。结果显示，得到的嵌入向量空间捕获到了音调之间的关系，尽管没有对于音乐内容的任何明确表示。其次，实验基于音乐语境的相似性，将一段贝多芬月光奏鸣曲的摘录做出了部分替换，得到的音乐结果与原曲在音调上也有一个相对较小的差异距离。

<br>

### 介绍

<br>

本文的目标是，仅通过音乐片段的语境信息，获取语义相似性。

接下来，文章介绍了**过去一些研究中使用的音乐模型**：（1）RNNs combined with Restricted Bolzmann Machines；（2）Long-Short Term RNN models；（3）Markov models；（4）包含音乐信息（如音高、时长、时间间隔等）的其他统计学模型。

而在本文中，相较于关注音乐内容，更多的关注点是对音乐片段的语境建模。

接下来是对**向量空间模型**的一些介绍。向量空间模型在NLP中被用于将words嵌入到连续向量空间中。在这个空间里，语义上相似的words会有距离更近的向量表示。而在本文写作时新提出的 *word2vec* 模型可以非常高效地完成向量创建任务。

接下来，文章引用 **Besson** 和 **Schon** 在2001年的论文，阐述了语言与音乐在结构与其生成期望（expectancy）的相似性（这里的expectancy不是很懂具体是什么意思）。由此引出了使用NLP模型处理音乐的合理性。

本篇文章之前，使用语义向量空间对于音乐语境建模只有比较少的尝试。例如，Huang et al.[2016]使用 *word2vec* 来对和弦序列建模，以向新手作曲者推荐较好的和弦。而本篇文章的目标是找到一种 *word2vec* 方法，可以对音乐语境信息进行更为一般的建模，而不是对和弦序列做删减表示。**本文将复杂的音乐表示为一系列的等长切片，不对节奏、和弦音调等音乐概念做任何附加处理。**

在之后的文章中，会介绍 *word2vec* 模型，然后讨论音乐是如何被表示的，最后评估实验结果。

<br>

### Word2Vec

<br>

关于 *word2vec* 的介绍部分都是一些老生常谈的内容，在此不多赘述。后面提到了 *word2vec* 的两种基本途径：*a continuous bag-of-words (CBOW)* 或者 *a continuous skip-gram model*。关于这一部分在之前的笔记中也有涉及，具体可以参考[笔记《Chord2Vec: Learning Musical Chord Embeddings》](https://lesoleil-a.github.io/%E9%9F%B3%E4%B9%90%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/2021/09/30/Chord2Vec-learning-musical-chord-embeddings.html)中对于 *skip-gram* 部分的介绍。二者区别如下：*CBOW* 是根据上下文预测当前字，而 *skip-gram* 是根据当前文字预测上下文。二者均有比较低的计算复杂度，可以在几个小时内处理十亿数量级的语料。*CBOW* 更快，然而 *skip-gram* 更适合处理小数据集。因此在这里选用后者。

**Skip-gram with negative sampling**

这一部分有三个比较值得注意的点：

【降低梯度计算复杂度的方法】

由于计算过程中需要使用softman function，计算梯度的复杂度很大。所以产生了一些降低复杂度的方案：比如 *hierarchical softmax* 以及 *noise contrastive estimation* 等（关于这些方法要参考其他的资料，本文只是简单提到）。**本项目中的 *word2vec* 模型使用了后者的一个变种，即 *negative sampling*。**

【negative sampling的中心思想】

**Negative sampling的中心思想是：一个训练良好的模型应该可以区别出数据与噪声。**于是原始的训练目标就简化成了一个更为高效的新目标：应用二分类 *logistic regression* 来区分数据和噪音。当模型可以给数据分配高概率，噪音分配低概率的时候，目标达到最优化。

【cosine similarity】

**对于这部分的介绍应该是本文比较值得注意的点，在读上一篇文章的时候就对这个概念不是很理解。**

在本文的向量空间中，**Cosine similarity** 被用作两个音乐切片向量之间相似度的衡量。对于 n 维向量空间中夹角为 *θ* 的非零向量 A 和 B ，其 *Cosine similarity* 被定义为

![cosine](/assets/img/1014/cosine.PNG)

<br>

### Musical slices as words

<br>

这一部分讲的是音乐片段的处理。

对于音乐片段的处理不依赖节奏等音乐概念，而是完全数据驱动的，最终的字母表应为一系列的音乐切片。每一个音乐数据都被简单地切割为等长、不重叠的片段（这边对于长度选择的介绍不是很明白）。这些片段包括了所有在此出现的音调，不论是在这一片段中开始的还是作为保持音出现在片段中的。

下文讲的是对于slices的进一步处理，以下总结为个人的理解：

首先，不会把音调归为和弦，而是将每一个音高都单独记录下来；并且，C~4~、C~5~等不同八度内的音高不会被看做是同一个；另外，所有的片段会被移调为 *C大调* 或者 *A小调*。

这样做的优点是：可以使得数据集中有更多重复的片段， 并且允许模型可以在更少的数据集上获得更好的训练。

<br>

### 结果

<br>

主要是为了检验模型是否能够很好地捕捉到音乐语境。数据集来自于贝多芬的音乐奏鸣曲，最终的数据集规模为 *70305* 个字，总共有 *14315* 个 *unique occurrences* 。

接下来是对训练过程的介绍。这里在准确度和训练时间之间会有一个 *trade off* 。模型维度越高越准确，但是时间复杂度会增大。最终选取了128维。然后是对 *skip-window* 大小的选择，经过实验发现，*skip-window* 的大小为1是最理想的。

结果中值得注意的有以下几点：

【可视化】

使用 *t-SNE* 进行音乐可视化。在之前 *Hamel and Eck（2010）* 的工作中，*t-SNE* 被用于根据音乐特征对音乐类型做出可视化的聚类。在文章所展示的结果中，音高上相近的（如半音）会形成聚类，而这种音乐组合不常出现；但是五度等常常一同出现的却会分散开。这个结果与作者提出的“音高上相近的距离较小”的目标一致。但是按照个人理解，本文的目的是为了预测最有可能一起出现的音乐组合，音高之间的距离应该是一起出现的可能性，然而现在的可视化结果却是基于音高之间的距离，除非是认为距离较远的一起出现的可能性更大，否则我认为这个可视化结果对于核心目标缺乏说服力。

![tsne](/assets/img/1014/tsne.PNG)

【评价是否捕捉到语义信息】

采用的方法是，把原曲中的某一些和弦，替换成最相似的和弦，也就是在向量空间模型中 *cosine similarity* 距离最小的和弦。然而这里的评价标准是人的听觉感受——是否与原曲听起来相像。这个标准比较主观，另外还有客观的评价标准，即 *tonal distance*。

【tonal distance】

原slice与调整后slice之间的每对音符的平均步长，使用 *tonnetz representation* 表示（*tonnetz：tone networks*）[Cohn, 1997]。

<br>

### 结论

<br>

未来：An embedded model that combines both word2vec with ,for instance, a long-short term memory recurrent neural network based on musical features.
