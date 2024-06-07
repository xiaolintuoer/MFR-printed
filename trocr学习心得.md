## 对于trOCR模型及其配置文件的一些理解

### 1. config.json

config.json是transformers下模型的必需且最重要的配置文件之一，用于存储模型的元数据和架构信息，用json格式编写，具有易读和兼容的优点，能保证transformers框架下的模型可以不直接访问源代码而被正确加载和应用。一些关键参数意思有：
* `model_type`: 指定模型的类型，如Bert, GPT等
* `architecture`: 列表形式，指定模型的架构，例如trOCR的模型框架是`["VisionEncoderDecoderModel"]`
* `decoder`,`encoder`: 字典形式，模型架构中具体组件模型的参数配置
* ……

**值得注意的是：**

`VisionEncoderDecoderModel`是专门用于视觉到视觉或视觉到语言的序列到序列任务。这个模型结合了两个主要部分：一个作为编码器的视觉模型（通常基于Transformer架构，如Vision Transformer - ViT）和一个作为解码器的语言模型（同样基于Transformer），两者之间通过编码器-解码器架构相互作用。

其中，编码器负责处理输入的视觉信息（如图像像素或经过预处理的图像特征），将其转化为一个高维向量表示。随后，这个向量被用作解码器的输入，解码器生成输出序列，这可以是文本描述（例如图像 captioning）、另一种形式的图像数据（例如图像翻译或超分辨率）或其他结构化的输出形式。

`VisionEncoderDecoderModel`允许用户根据特定的任务需求选择或构建合适的视觉编码器和文本解码器，因此它是TrOCR模型训练的最主要部分。

### 2. generation_config.json

非必需配置文件，通常出现在文本生成相关任务中，尤其是使用自回归模型（GPT, BART等）的情况。它主要控制文本生成过程的参数和配置，使用户可微调文本生成的具体方式。

### 3. preprocessor_config.json

预处理器的配置文件，负责在模型训练和推理前对数据进行必要的转换等处理，确保数据格式、结构等符合要求。在trOCR中采用`DeiTImageProcessor`型的图像处理器，并由transformers的模块`TrOCRProcessor`直接读取配置然后调用预训练模型，不必额外配置模型实例。

### 4. tokenizer_config.json

分词器的配置文件，负责在模型

## **Transformers模型修改再训练**

在Hugging Face的Transformers库中修改模型结构并进行训练主要涉及以下几个步骤：

### 1. 导入必要的库

首先，确保你已经安装了`transformers`库。如果还没有安装，可以通过pip安装：

```bash
pip install transformers
```

然后导入需要的模块：

```python
from transformers import AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch import nn
```

### 2. 继承并修改模型

假设你想在BERT模型的基础上添加一些自定义层，你可以通过继承现有的模型类并重写其部分方法来实现。以下是一个简单的例子，我们在`BertForSequenceClassification`模型中添加一个线性层：

```python
class CustomModel(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义层，例如一个线性层
        self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # 在原有输出上应用自定义层
        custom_output = torch.relu(self.custom_layer(outputs[1]))  # 使用隐层状态作为输入
        
        # 如果是分类任务，你可能需要调整这里的逻辑以整合自定义层的输出
        # 这里仅作为示例，实际使用时需根据任务需求调整
        outputs['logits'] = custom_output
        
        return outputs
```

### 3. 配置和初始化模型及分词器

接下来，选择或指定预训练模型的配置，并初始化你的自定义模型和分词器：

```python
config = BertConfig.from_pretrained('bert-base-uncased')  # 或者使用其他模型的配置
model = CustomModel(config)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

### 4. 准备数据集

你需要准备一个适合你任务的数据集，并使用tokenizer对其进行预处理。

### 5. 定义训练参数与训练器

使用`TrainingArguments`来设置训练参数，并使用`Trainer`来组织训练过程：

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 假设你已经有了预处理好的训练和验证数据
train_dataset = ...
eval_dataset = ...

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},  # 示例计算准确率
)
```

### 6. 开始训练

最后，调用`trainer.train()`开始训练：

```python
trainer.train()
```

这样，你就成功地修改了Transformers中的模型结构并进行了训练。请注意，根据你的具体需求，上述代码可能需要适当调整，比如数据集的准备、损失函数的调整、训练参数的选择等。

## **TrOCR在新语言上的微调**

### 1. 准备工作

1. **数据收集**：准备大量包含目标语言（如数学）字符的图片和对应的文本标签。确保数据质量高——标注准确，且覆盖目标语言的各种字符和形式。
2. **Tokenizer适应**：
    * **使用或创建特定语言的Tokenizer**：不同的Tokenizer处理不同序列的效果可能不一样。原始TrOCR使用的是英文Tokenizer，对于数学语言，大概率需要应用或创建一个新的Tokenizer，确保它能准确处理数学语言中的所有字符。
    * **字符映射**：对新Tokenizer进行调整，添加数学所有字符的映射。`"char": no.`

### 2. 模型微调

1. **模型结构调整**：TrOCR模型基于Vision Transformer (ViT) 和 Transformer-XL 架构，可能需要调整视觉部分（即encoder）。*不一定要改模型架构，但可能需要调整超参数*。

2. **预处理**：根据实际图像特性调整图像预处理，如图像尺寸、对比度增强、抗噪声等。

3. **后处理***：对于数学公式识别任务可能还需要进行后处理，例如对输出的latex命令进行纠错，防止出现命令报错的情况。

4. **微调策略**：
    * **小批量训练**：如果数据集数量不足，开始时可以采用较小的批次大小，并逐步增加。
    * **学习率调整**：使用较小的学习率开始，可以采用<u>余弦退火</u>或<u>热重启</u>等策略。
    * **早停策略**：监控验证集的性能，当性能不再提升时停止训练，防止过拟合

### 3. 特别注意事项

* **数据多样性**：确保数据集中字符和上下文的多样性，避免过拟合特定的字符排列或组合。
* **字符对齐**：确保图像中的字符与文本标签精确对齐，对训练效果至关重要。
* **评估指标**：选择合适目标语言的评估指标，常用有 CER(Character Error Rate) 或 WER(Word Error Rate)。对于数学指标可能需要适当调整。
* **图像噪声**：图片数据集中包含许多噪声，如①下划线；②其它题目的部分；③涂改（区分内部的和分离的）等。噪声的有效处理对后面字符识别的准确率具有非常重要的影响，需要使用高效的预处理算法或模型。
* **跨语言知识迁移**：考虑是否从其他语言中迁移知识，例如共享字符集。因为手写数学答案中可能也会出现一些中英文字符。
* **框架融合**：TrOCR可以和SAN等框架融合，更好地处理手写数学答案的问题。

