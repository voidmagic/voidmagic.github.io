---
title: Fairseq漫游指南（1）——命令行工具
date: 2020-10-12 14:45:05
tags: 
    - Fairseq
---


2017年9月，Facebook AI Research开源了序列建模工具Fairseq。作为对Lua/Torch版本的改进，新款Fairseq基于Python和Pytorch，更加简单易用人性化。经过三年的迭代，fairseq目前已经拥有近两百位contributor，总代码量7万余行，功能和规模都已不同往日。

作为一个通用的序列建模工具，fairseq可以在多个自然语言处理任务上使用，如机器翻译、自动摘要、语音识别等文本生成任务，或者BERT、GPT等语言模型的训练；同时fairseq还实现了目前常用的多数模型，如RNN、CNN、Transformer、RoBERTa、XLM等。除了大量内置的任务和模型，fairseq还提供了极为简洁的接口，以便于使用者扩展已有模型、验证新的想法。

本文将以训练Transformer-based机器翻译模型为例，介绍fairseq的基本使用方法。

<!--more-->

## 环境搭建
深度神经网络模型的训练需要GPU支持，因此硬件方面需要安装有NVIDIA GPU的服务器，这里以GTX1080（驱动版本430.64，CUDA版本10.1）的Ubuntu 16.04为例，其他GPU、驱动、操作系统可能有细微差异。为了保证环境的一致性，在环境搭建中将从pytorch的安装开始。

1. 安装pytorch

使用conda安装pytorch的时候，可以同时指定CUDA版本，意思是安装使用指定CUDA预编译的pytorch：
```
conda create -n fairseq python=3.7
conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch -y 
```

2. 安装fairseq

由于直接使用pip安装的fairseq版本（0.9.0）还停留在2019年12月，为了使用更新的特性，我们选择GitHub上的最新版本（commit 77983ee）：
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq && git checkout 522c76b && pip install --editable ./
```

3. 安装apex（可选）

Apex是NVIDIA为Pytorch开发的混合精度训练库，在多卡训练、半精度训练的过程中可以带来更快的训练效率。安装apex的时候需要注意，由于安装过程会编译CUDA代码，且需要与pytorch使用同一版本的CUDA编译，因此要先安装与pytorch一致的CUDA。下面的命令会下载10.1版本的CUDA，并将其安装在用户目录下（不需要管理员权限），之后用其编译安装apex：
```
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
mkdir $HOME/.cuda && sh cuda_10.1.105_418.39_linux.run --silent --toolkit --toolkitpath=$HOME/.cuda --defaultroot=$HOME/.cuda
export CUDA_HOME=$HOME/.cuda
git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


4. 验证安装

```
python -c "import torch;print(torch.__version__, torch.version.cuda)"
```
会显示pytorch版本（1.6.0）和对应cuda版本（10.1）；

```
python -c "import fairseq;print(fairseq.__version__)"
```
会显示fairseq版本（0.9.0）；

```
python -c "import fairseq;print(fairseq.utils.multi_tensor_l2norm_available)"
```
会显示apex是否成功安装（True）。


## 准备数据
在机器翻译中，需要双语平行数据来进行模型的训练，在这里使用fairseq中提供的数据：
`bash fairseq/examples/translation/prepare-iwslt14.sh`

这个脚本会下载IWSLT 14 英语和德语的平行数据，并进行分词、BPE等操作，处理的结果为：
```
iwslt14.tokenized.de-en
├── code
├── test.de
├── test.en
├── tmp
├── train.de
├── train.en
├── valid.de
└── valid.en
```


## 数据二进制化
之后，使用fairseq的预处理命令fairseq-preprocess将文本数据转换为二进制的文件：
```
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref iwslt14.tokenized.de-en/train \
    --validpref iwslt14.tokenized.de-en/valid \
    --testpref iwslt14.tokenized.de-en/test
```
其中，`trainpref`、`validpref`和`testpref`代表两个语言对应文件的前缀（路径和文件名的前缀），`source-lang`和`target-lang`两个参数代表两个语言对应文件的后缀名（不代表具体语言，只是通过后缀区分两种语言的数据），fairseq通过这几个参数的组合，来寻找对应的文本数据。

例如我们的数据中只有训练数据和测试数据，且文件后缀为`src`和`tgt`，即`train.src`、`train.tgt`、`test.src`和`test.tgt`，那么通过指定`--source-lang src --target-lang tgt --trainpref train --testpref test`，也可以读取的对应的文件。

预处理命令首先会从训练文本数据中构建词表。在默认情况下，会将所有出现过的单词根据词频排序，并将这个排序后的单词列表作为最终的词表。同时，fairseq还提供了相关的参数来自定义词表：

`--thresholdsrc/--thresholdtgt`，分别对应源端（source）和目标端（target）的词表的最低词频，词频低于这个阈值的单词将不会出现在词表中，而是统一使用一个unknown标签来代替。

`--srcdict/--tgtdict`，其参数为一个文件名，即使用已有的词表，而不去根据文本数据中单词的词频构建词表。已有的词表文件中，每一行包含一个单词及其词频（这个词频只作为排序和阈值过滤的依据，不代表实际的词频）。

`--nwordssrc/--nwordstgt`，源端和目标端词表的大小，在对单词根据词频排序后，取前n个词来构建词表，剩余的单词使用一个统一的unknown标签代替。

`--joined-dictionary`，源端和目标端使用同一个词表，对于相似语言（如英语和西班牙语）来说，有很多的单词是相同的，使用同一个词表可以降低词表和参数的总规模。

构建的词表是一个单词和序号之间的一对一映射，这个序号是单词在词表中的下标位置。预处理命令在构建词表之后，会将文本数据转换为数值形式，也就是把文本中的每一个词，转换为对应的序号。之后，数值化的文本数据会被进一步编码，默认情况下使用Memory-Mapped IndexedDataset，这种数据编码方式不仅可以压缩文件大小，还可以根据索引进行随机读取，因此在训练的时候不需要加载全部数据，从而节约内存使用。

二进制化的数据文件会默认保存在data-bin目录下，包括生成的词表、训练数据、验证数据和测试数据。也可以通过--destdir参数，将生成的数据保存在其他目录。


## 模型训练
在对数据进行预处理之后，就可以开始训练翻译模型了。模型训练使用的命令是fairseq-train，在参数中需要指定训练数据、模型、优化器等参数：
```
fairseq-train data-bin --arch transformer_iwslt_de_en \
        --max-tokens 4096 --max-update 30000 \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0007 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --no-progress-bar --save-interval-updates 1000 
```

fairseq-train提供了大量的训练参数，从而进行定制化的训练过程，其中主要的参数可以分为数据（data）、模型（model）、优化（optimizing）、训练（分布式和多GPU等）、日志（log）和模型保存（checkpointing）等。

1. 数据部分

数据部分的常用参数主要有训练数据的位置（路径），训练时的batch size等。其中，batch size可以通过两种方法指定，`--max-tokens`是按照词的数量来分的batch，比如`--max-tokens 4096`指每个batch中包含4096个词；另外还可以通过句子来指定，如`--max-sentences 128`指每个batch中包含128个句子。

2. 模型部分

模型部分的参数主要有`--arch`，用指定所使用的网络模型结构，有大量预设的可以选择。其命名一般为“model_setting”，如“transformer_iwslt_de_en”就是使用Transformer模型和预设的iwslt_de_en超参数。在大数据上进行训练的时候，可以选择“transformer_wmt_en_de”、“transformer_wmt_en_de_big”等设置。除了Transformer之外，还有LSTM、FConv等模型及对应的预设超参数可以选择。

在使用预设模型的同时，还可以通过额外的命令行参数来覆盖已有的参数，如“transformer_wmt_en_de”中预设的编码器层数是6，可以通过`--encoder-layers 12`将编码器改为12层。

3. 优化部分

通过`--criterion`可以指定使用的损失函数，如cross_entropy等。和`--arch`参数一样，也可以通过命令行参数来覆盖特定损失的默认参数，比如通过`--label-smoothing`0.1，可以将label_smoothed_cross_entropy损失中默认为0的label-smoothing值改为0.1。

通过`--optimizer`可以指定所使用的优化器，如adam、sgd等；通过`--lr-scheduler`可以指定学习率缩减的方式。

通常来说，参数优化紧跟梯度计算，即每计算一次梯度，就会进行一次参数更新。在某些时候，我们希望在多次梯度计算之后进行一次更新，来模拟多GPU训练，可以通过--update-freq来指定，比如在单个GPU上指定“--update-freq 4”来训练，结果和4个GPU训练是基本等价的。

4. 训练部分

Fairseq支持单GPU、多GPU、多机器等多种训练方式，在默认情况下，会根据当前机器的GPU数量来确定训练方式。在绝大多数情况下，这部分参数都不需要关心，而是通过系统环境变量的方式，`export CUDA_VISIBLE_DEVICES=0,1`,来指定单卡、多卡训练。

如果所使用的GPU支持半精度，那么可以通过参数`--fp16`进行混合精度训练，可以极大提高模型训练的速度。通过`torch.cuda.get_device_capability(0)[0]`可以确定GPU是否支持半精度，如果该值小于7则不支持，大于等于7则支持。

5. 日志和模型保存

在默认情况下，fairseq使用tqdm和进度条来展示训练过程，但是这种方法不适合长时间在后台进行模型训练。通过`--no-progress-bar`参数可以改为逐行打印日志，方便保存。默认情况下，每训练100步之后会打印一次，通过`--log-interval`数可以进行修改。
Fairseq在训练过程中会保存中间模型，保存的位置可以通过--save-dir指定，其默认为checkpoints。中间模型保存的频率有两种指定方式，`--save-interval`指定了每N个epoch（遍历训练数据N次）保存一次；`--save-interval-updates`指定了每N步保存一次，这种通过step来保存模型的方法目前更为常用。
Note：在使用多GPU训练时，指定的batch size（max tokens或max sentences）是单个GPU上的数量，以token计算为例，最终batch size的大小为max-tokens、GPU数量、update-freq的乘积。

## 解码
在经过了充分训练之后，就可以使用模型来进行翻译了。Fairseq提供了两种解码的方式：批生成解码（fairseq-generate）和交互式解码（fairseq-interactive）。

1. fairseq-generate

fairseq-generate用来解码之前经过预处理（fairseq-preprocess）的数据：
```
fairseq-generate data-bin --path checkpoints/checkpoint_best.pt --remove-bpe
```

默认情况下，这个命令会从预处理的数据中，解码测试数据（test set）。通过—gen-subset可以指定解码其他部分，如`--gen-subset train`就会翻译整个训练数据。
如果不想得到翻译结果，只想看到翻译结果的BLEU分值，可以通过—quiet参数，只显示翻译进度和最后打分。

通过`--beam`、`--lenpen`和`--unkpen`，可以分别设置beam search中的beam size，长度惩罚和unk惩罚。

参数`--remove-bpe`可以指定对翻译结果的后处理，由于在准备数据的时候，使用了BPE切分，该参数会把BPE切分的词合并为完整的单词。如果不加该参数，则输出的翻译结果和BLEU打分都是按照未合并BPE进行的。如果准备数据时BPE切分使用的是sentencepiece（https://github.com/google/sentencepiece），那么参数的值还可以设为`--remove-bpe sentencepiece`，以合并sentencepiece的切分。

2. fairseq-interactive

fairseq-interactive可以进行交互式逐句解码，其参数和fairseq-generate基本一致。以下命令用来逐行翻译test.de文件中的句子：
```
cat test.de | fairseq-interactive data-bin --path checkpoints/checkpoint_best.pt --remove-bpe
```

3. 保存翻译结果

在默认情况下，两种解码方式会将翻译结果直接显示出来，如果想保存翻译结果，可以通过`--results-path`参数来指定保存结果的位置，或者可以通过重定向的方法将输出保存到文件：
```
fairseq-generate someargs > result.txt
```

这两种方法得到的翻译结果文件，包含了解码日志（log）、原文、译文、打分等信息，并且顺序与原文不一致，通过以下命令可以得到排序后的译文：
```
grep ^H result.txt | sort -n -k 2 -t '-' | cut -f 3
```


## 总结
本文以训练一个简单翻译模型为例，介绍了fairseq几个命令行工具（fairseq-preprocess、train、generate、interactive）的基本使用方法。通过命令行调用的方法，可以在不接触内部代码的情况下，完成训练解码的整个流程。

在后续的文章中，将会陆续介绍fairseq的扩展方法，比如定义新的任务，实现新的模型等。