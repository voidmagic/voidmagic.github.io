---
title: Fairseq漫游指南（2）——扩展模型
date: 2020-10-12 14:45:05
tags:
---


本文为Fairseq漫游指南系列的第二篇文章。前面一篇文章以基于Transformer的翻译模型为例，对Fairseq的命令行使用方法进行了初步的介绍。Fairseq预设了大量的任务和模型，可以根据需要准备数据，并参考对应任务、模型的参数进行训练和解码。

在实际的使用中，现有的模型可能无法满足真实任务的需要，我们可能需要处理不同类型的输入输出，或者需要对模型进行修改以验证新的想法。在这种情况下，只通过命令行调用预设任务和模型的方法就存在很大的局限，我们需要对Fairseq本身进行扩展，以满足实际多样化的需求。

本文以实现一个可以双向翻译（EN-DE和DE-EN）的Transformer模型为例，来介绍Fairseq扩展的使用方法。

## Fairseq扩展概述
Fairseq允许用户在不修改源代码的情况下，以插件的形式进行扩展。目前，可以自定义五种插件：
	1. 任务（Tasks）：任务定义了我们要完成的整个流程，包括读取数据组成batch、模型初始化、训练、测试等。
	2. 模型（Models）：模型定义了网络的结构、包含的参数、前向计算过程。
	3. 评价准则（Criterions）：评价准则也就是损失函数，用来根据网络输出和真实标签计算损失。
	4. 优化器（Optimizers）：在反向传播之后，优化器决定了更新模型参数的方式。
	5. 学习率调度器（Learning Rate Schedulers）：学习率调度器可以用来根据训练过的步数，动态调整学习率。

对于这五种插件，Fairseq自身的代码中提供了大量的预设，可以在对应的目录下查看，如`fairseq/models`目录下提供了多种模型的实现。在指定了这五种插件（可以为预设值，也可以为用户编写的插件）之后，fairseq的训练流程可以抽象为：
```
for epoch in range(num_epochs):
    itr = task.get_batch_iterator(task.dataset('train'))
    for num_updates, batch in enumerate(itr):
        task.train_step(batch, model, criterion, optimizer)
        average_and_clip_gradients()
        optimizer.step()
        lr_scheduler.step_update(num_updates)
    lr_scheduler.step(epoch)
```

如前所述，模型的单步训练过程在任务中定义，即`task.train_step`。默认情况下，其实现如下：
```
def train_step(self, batch, model, criterion, optimizer, **unused):
    loss = criterion(model, batch)
    optimizer.backward(loss)
    return loss
```

只通过命令行的方式，可以选择使用不同的预设插件，如LSTM、Transformer等不同的模型。但如果我们想要扩展Fairseq没有提供的一些功能，那么就需要我们自己编写一些插件，并进行注册，以便Fairseq在运行的时候可以加载我们自定义的插件。接下来我们以一个最简单的例子，来实现自己的Transformer模型。

首先需要建立我们的代码仓库，假设代码存放在`$HOME/codebase/custom`：
```
├── custom
    └── __init__.py
```

其中，`__init__.py`的内容如下：
```
from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture

@register_model('my_transformer')
class MyTransformer(TransformerModel):
    pass

@register_model_architecture('my_transformer', 'iwslt_arch')
def my_transformer_iwslt(args):
    transformer_iwslt_de_en(args)
```

在Fairseq中，模型称为`model`，模型对应的超参数称为`model_architecture`。在这个例子中，我们定义了一个名为`my_transformer`的模型，以及其对应的`iwslt_arch`超参数。由于模型直接继承了预设的`TransformerModel`，超参数直接调用了`transformer_iwslt_de_en`，因此其功能没有任何的改变，只是名字发生了改变。在编写了这个简单的插件后，就可以通过命令行来进行调用了：
```
fairseq-train data-bin --arch iwslt_arch --user-dir $HOME/codebase/custom --max-tokens 4096 --optimizer adam
```

其中，`data-bin`是上一篇文章”命令行工具“中预处理的数据路径。该命令可以在任何目录下执行，只要通过`--user-dir $HOME/codebase/custom`参数指定我们的插件代码位置即可。

从上面的例子可以看出，自定义并使用一个模型插件需要以下几个步骤：
	1. 创建一个python module，即包含`__init__.py`文件的目录（这个例子中为`$HOME/codebase/custom`）；
	2. 定义新的模型类（类名可以任意，只要不和其他重复即可），并用`@register_model('model_name')`装饰器来进行注册（model_name即模型名，Fairseq通过这个名字来定位插件对应的类）；
	3. 定义模型对应的预设超参数model_architecture，这是一个函数，接收`args`参数。比如想将dropout预设为0.1，可以通过`args.dropout = 0.1`来完成。和模型类似，想要Fairseq能够将其识别为预设超参数，需要使用`@register_model_architecture('model_name', 'arch_name')`来进行注册，其中`model_name`是模型名，`arch_name`是预设值的名字；
	4. 如果插件的实现在`__init__.py`之外的文件中，那么还需要在`__init__.py`文件中导入注册的model和model_architecture，这是因为fairseq在运行时通过查找已经导入（加载）的插件名（如模型名）来定位具体的实现，如果不进行导入，那么即便指定了`--user-dir`，fairseq也只能加载在`__init__.py`中的代码，而找不到在其他文件中定义的插件。在这个例子中，由于model和model_architecture都定义在了`__init__.py`文件中，因此不需要额外的导入；
	5. 在命令行调用的时候，指定`--user-dir`参数为插件路径，并使用`--arch`来告诉Fairseq使用我们自定义的模型和超参数。

定义新的任务、优化器等，和定义新的模型基本一致，都是通过定义一个新的类，并通过`@register_*`来注册。下面，我们将实现一个双向翻译、参数共享的翻译系统，来看一下扩展在实际中如何使用。

## 准备工作
我们使用和系列第一篇《命令行工具》中一致的环境：
	1. python 3.7
	2. pytorch 1.6.0
	3. Fairseq，commit 522c76b
	4. cuda 10.1
	5. Apex 0.1

对于数据，我们同样使用iwslt 14英德平行数据来进行训练。由于我们的目的是进行两种语言的双向翻译，编码器和解码器都需要拥有处理两种语言的能力，因此我们需要对两种语言使用共享的词表，在fairseq的预处理命令中，可以通过`--joined-dictionary`参数来指定：
```
bash fairseq/examples/translation/prepare-iwslt14.sh

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref iwslt14.tokenized.de-en/train \
    --validpref iwslt14.tokenized.de-en/valid \
    --testpref iwslt14.tokenized.de-en/test --joined-dictionary

```

默认情况下，预处理后的二进制数据文件保存在data-bin目录下。

## 目标
对于双向翻译任务，我们希望给定一个源语言的句子，模型能解码出一个目标语言的句子；给定一个目标语言的句子，模型能够解码出一个源语言的句子。为了达到这个目的，我们需要模型能够区分出输入是哪种语言，或者说，希望翻译为哪种语言。在多语言机器翻译中，一个简单而有效的做法是，在输入的句子前面加上一个标签来指明希望模型输出的语言，比如在句子前面加一个`__2<en>__`，来告诉模型我们希望得到英文的翻译结果。

为了给输入句子加上标签，我们需要在读取数据和组成batch之间进行处理，即读取所有句对，给句对的源端部分加上指明目标语言的标签，再根据句长，将相似长度的句子打包为一个batch，并将这个batch数值化，来构成模型的输入。如前所述，读取数据组成batch的操作需要在Task中进行，因此我们需要自定义一个Task，来对数据进行处理。

在模型部分，我们希望编码器和解码器共享自注意力和前馈神经网络中的参数，即Transformer中self attention和feed forward模块的参数。这一部分的改变在模型中体现，因此我们还需要自定义一个基于Transformer的Model，以实现参数的共享。

在明确了目标之后，我们首先需要创建代码库，保存在`codebase/custom`目录下：
```
└── custom
    ├── bidirectional_transformer.py
    ├── bidirectional_translation_task.py
    └── __init__.py

1 directory, 3 files
```

其中，`bidirectional_transformer.py`保存我们自定义的模型，`bidirectional_translation_task.py`保存我们自定义的任务。为了使Fairseq能够加载自定义模型和任务，需要在`__init__.py`中将其导入：
```
from . import (
    bidirectional_transformer as _,
    bidirectional_translation_task as _,
)
```

接下来，我们的目标就是实现`bidirectional_transformer`和`bidirectional_translation_task`了。


## 参数共享的模型
模型部分相对比较简单，由于Fairseq中实现了大量的预设模型，因此我们在实现自定义模型的时候，应该尽量复用已有的代码，通过模型类的继承、方法的重载来实现功能上的修改和扩展。我们直接使用Transformer的实现，并在模型初始化之后，指定参数共享的部分：

``` 
from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture

@register_model('bidirectional_transformer')
class BidirectionalTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.make_shared_component()
    
    def make_shared_component(self):
        for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
            dec_layer.self_attn.k_proj = enc_layer.self_attn.k_proj
            dec_layer.self_attn.v_proj = enc_layer.self_attn.v_proj
            dec_layer.self_attn.q_proj = enc_layer.self_attn.q_proj
            dec_layer.self_attn.out_proj = enc_layer.self_attn.out_proj
            dec_layer.fc1 = enc_layer.fc1
            dec_layer.fc2 = enc_layer.fc2

@register_model_architecture('bidirectional_transformer', 'iwslt_arch')
def iwslt_preset_hyperparameters(args):
    transformer_iwslt_de_en(args)
```

通过继承Fairseq中的`Transformer`模型，我们的`BidirectionalTransformerModel`就可以实现与Transformer相同的功能。在模型的实例化方法`__init__`中，首先调用父类`TransformerModel`的初始化方法，来初始化模型及其参数，然后调用`make_shared_component`方法，来共享编码器和解码器每一层中的`self_attn`和`fc1`、`fc2`参数。同时，我们使用了`transformer_iwslt_de_en`来定义名为`iwslt_arch`的预设超参数。最后通过`register_model`和`register_model_architecture`来注册模型，就可以在Fairseq中使用了。


## 双向翻译任务
在自定义的双向翻译任务中，我们需要将标签加到每个源端句子前面。由于我们的目的和翻译任务基本一致，因此可以复用Fairseq中的`TranslationTask`，只需要实现数据加载部分即可。完整代码如下：
```
import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset

@register_task('bidirectional_translation_task')
class BidirectionalTranslationTask(TranslationTask):
    def load_dataset(self, split, **kwargs):
        shared_dict = self.src_dict
        src, tgt = data_utils.infer_language_pair(self.args.data)
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))

        src_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, shared_dict)
        tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, shared_dict)

        src_prepend_dataset = PrependTokenDataset(src_raw_dataset, shared_dict.index('__2<{}>__'.format(self.args.target_lang)))
        tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, shared_dict.index('__2<{}>__'.format(self.args.source_lang)))

        src_dataset = src_prepend_dataset if split == 'test' else ConcatDataset([src_prepend_dataset, tgt_prepend_dataset])
        tgt_dataset = tgt_raw_dataset     if split == 'test' else ConcatDataset([tgt_raw_dataset,     src_raw_dataset])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, shared_dict, tgt_dataset, tgt_dataset.sizes, shared_dict)

    @classmethod
    def setup_task(cls, args, **kwargs):
        task = super(BidirectionalTranslationTask, cls).setup_task(args)
        for lang_token in sorted(['__2<{}>__'.format(args.source_lang), '__2<{}>__'.format(args.target_lang)]):
            task.src_dict.add_symbol(lang_token)
            task.tgt_dict.add_symbol(lang_token)
        return task
```

参考`fairseq/tasks/translation.py`的代码可以看到，数据加载实在方法`load_dataset`中完成的，我们可以在其基础上（加载源语言到目标语言的数据），增加目标语言到源语言数据的加载，并给加载的数据添加标签。`load_dataset`方法的基本流程是，通过`spilt`参数，来加载对应的数据，并将加载的数据赋值给`self.datasets[split]`。其中`split`参数一般为`train`、`valid`或者`test`。默认情况下，训练、验证、解码分别使用对应的数据，但也可以通过命令行来指定，如`fairseq-generate --gen-subset train`就会解码训练数据（即split为train）。

在我们的实现中，读取数据和添加标签的流程如下：
	1. 仿照`fairseq/tasks/translation.py`中的代码，使用`data_utils.load_indexed_dataset`来分别读取两种语言预处理后的二进制数据；
	2. 使用`PrependTokenDataset`给两种语言的数据都创建一个加标签的版本；
	3. 如果是测试的情况下`split == 'test'`，只使用 `src_prepend_dataset`和`tgt_raw_dataset`来构建数据集；如果是训练或者验证，则将加标签的源语言和目标语言数据使用`ConcatDataset`进行拼接，得到`src_dataset`，将两种语言不加标签的数据拼接，得到`tgt_dataset`，来构建数据集；
	4. 根据`src_dataset`和`tgt_dataset`，创建一个`LanguagePairDataset`，并赋值给`self.datasets[split]`。

在这个例子中，我们使用到了`PrependTokenDataset`、`LanguagePairDataset`、`ConcatDataset`三个Fairseq中定义的类来完成加标签、拼接数据等操作。在`fairseq/data`目录下，还有大量预定义的数据类可供使用，同时，我们还可以继承预定义的类来扩展其功能，完成更复杂的数据处理。


最后，由于我们使用了额外的标签来指定目标语言，所以需要在词表中添加对应的语言标签。通过查看`TranslationTask`的代码可知，词表的创建和初始化是在`setup_task`中进行的，我们通过重写该方法，在任务创建完成后，为`src_dict`和`tgt_dict`分别添加源语言标签和目标语言标签。


## 训练和解码

在创建了自定义的任务和模型后，就可以使用该插件来进行训练了。进行训练和解码的命令和前文所介绍的基本一致，只需要指定插件代码的位置`--user-dir`、模型结构`--arch`和任务`--task`：
```
fairseq-train data-bin --max-tokens 4096 --max-update 50000 \
        --arch iwslt_arch --task bidirectional_translation_task --user-dir $HOME/codebase/custom \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0007 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1
```

解码的命令不需要指定模型结构：
```
fairseq-generate data-bin --path checkpoints/checkpoint_best.pt --remove-bpe --user-dir $HOME/codebase/custom --task bidirectional_translation_task --source-lang en --target-lang de

fairseq-generate data-bin --path checkpoints/checkpoint_best.pt --remove-bpe --user-dir $HOME/codebase/custom --task bidirectional_translation_task --source-lang de --target-lang en
```

其中，参数`--source-lang`和`--target-lang`可以进行特定方向的翻译，用来验证模型训练得到的双向翻译能力。如果不指定这两个参数，则默认是和数据预处理时相同的翻译方向（德语到英语）。


## 总结
本文通过一个双向翻译的例子，介绍了Fairseq扩展插件的基本使用方法。大多数的NLP任务都可以在不修改源码的情况下，通过编写插件来实现，这在很大程度上简化了实验的流程，我们只需要编写插件实现与原方法、模型不同的部分，而不需要关注重复的模式和训练流程。

在实际开发插件的过程中，关键的问题在于如何定位我们需要修改的部分，以及如何最大程度地复用Fairseq已经实现的部分。后续文章将介绍Fairseq中已经实现的一些任务、模型，以及数据集等常用的工具，以便了解我们要实现的功能在fairseq中是否已经有对应的实现及实现对应的位置。

















