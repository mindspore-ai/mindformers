# Llama 3

## 模型描述

Llama 3，是开源Llama系列的最新产品，目前有二个版本：Llama3-8B，Llama 3-70B。Llama 3在来自公开可用来源的超过15T的数据上进行了预训练。微调数据包括公开可用的指令数据集，以及超过1000万个人工标注的示例。模型支持上下文窗口长度8K，并使用了新的分词器，词汇表大小达到128256个，采用了分组查询注意力机制(GQA)。Llama 3模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。目前Mindformers支持Llama 3-8B和Llama 3-70B。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                           |      Task       | SeqLength | Datasets |   Performance   |  Phase   |
|:-------------------------------------------------|:---------------:|:---------:|:--------:|:---------------:|:--------:|
| [llama3_8b](llama3_8b/finetune_llama3_8b.yaml)   | text_generation |   8192    |  alpaca  | 2581 tokens/s/p | Finetune |
| [llama3_70b](llama3_70b/predict_llama3_70b.yaml) | text_generation |   8192    |    -     |  335 tokens/s   | Predict |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                            |      Task       | SeqLength | Datasets |  Performance   |  Phase   |
|:--------------------------------------------------|:---------------:|:---------:|:--------:|:--------------:|:--------:|
| [llama3_70b](llama3_70b/finetune_llama3_70b.yaml) | text_generation |   8192    |  alpaca  | 337 tokens/s/p | Finetune |

## 模型文件

`Llama 3` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：

   ```text
   research/llama3
       ├── llama3_8b
       │     ├── predict_llama3_8b.yaml      # 8B推理配置
       │     └── finetune_llama3_8b.yaml     # 8B全量微调启动配置
       └── llama3_70b
             ├── predict_llama3_70b.yaml     # 70B推理配置
             ├── pretrain_llama3_70b.yaml    # 70B预训练配置
             └── finetune_llama3_70b.yaml    # 70B全量微调A启动配置
   ```

3. 数据预处理脚本和任务启动脚本：

   ```text
   research/llama3
       ├── llama3_tokenizer.py      # llama3 tokenizer处理脚本
       ├── llama3_conversation.py   # 微调数据集处理，将原始alpaca转换为对话形式alpaca
       └── llama3_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据集及权重准备

#### 数据集下载

MindFormers提供**Wiki103**作为[预训练](#预训练)数据集，**alpaca**作为[微调](#微调)数据集。

| 数据集名称   |            适用模型            |   适用阶段   |                                      下载链接                                       |
|:--------|:--------------------------:|:--------:|:-------------------------------------------------------------------------------:|
| Wiki103 | llama3-8b <br/> llama3-70b | Pretrain |    [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)     |
| alpaca  | llama3-8b <br/> llama3-70b | Finetune | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **Wiki103 数据预处理**

  使用`research/llama3/llama3_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
  python llama3_preprocess.py \
   --dataset_type wiki \
   --input_glob /{path}/wiki.train.tokens \
   --model_file /{path}/tokenizer.model \
   --seq_length 8192 \
   --output_file /{path}/wiki8192.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.train.tokens的文件路径
  model_file:   模型tokenizer.model文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

- **alpaca 数据预处理**

  1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

     ```shell
     python alpaca_converter.py \
       --data_path /{path}/alpaca_data.json \
       --output_path /{path}/alpaca-data-conversation.json

     # 参数说明
     data_path:   输入下载的文件路径
     output_path: 输出文件的保存路径
     ```

  2. 执行`research/llama3/llama_preprocess.py`，生成Mindrecord数据，将带有prompt模板的数据转换为mindrecord格式。

     ```shell
     # 此工具依赖fschat工具包解析prompt模板, 请提前安装fschat >= 0.2.13 python = 3.9
     python llama3_preprocess.py \
       --dataset_type qa \
       --input_glob /{path}/alpaca-data-conversation.json \
       --model_file /{path}/tokenizer.model \
       --seq_length 8192 \
       --output_file /{path}/alpaca-fastchat8192.mindrecord

     # 参数说明
     dataset_type: 预处理数据类型
     input_glob:   转换后的alpaca的文件路径
     model_file:   模型tokenizer.model文件路径
     seq_length:   输出数据的序列长度
     output_file:  输出文件的保存路径
     ```

> 数据处理时候注意bos，eos，pad等特殊`ids`要和配置文件中`model_config`里保持一致。

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

| 模型名称            | MindSpore权重 |                                        HuggingFace权重                                         |
|:----------------|:-----------:|:--------------------------------------------------------------------------------------------:|
| Llama3-8B       |      -      |                  [Link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                   |
| Llama3-70B      |      -      |                  [Link](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                  |

> 注: 请自行申请huggingface上llama3使用权限，并安装transformers=4.40版本

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 预训练

MindFormers提供`llama3_70b`多机多卡的预训练示例，请参照[数据集下载](#数据集下载)获取mindrecord格式的`Wiki103`数据集。参照[模型权重下载](#模型权重下载)获取Llama3-70B权重和分词器文件。

### 多机训练

以llama3_70b为例，使用`pretrain_llama3_70b.yaml`配置文件，执行8机64卡预训练。需要先对权重进行切分，切分权重可以参见[分布式权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/transform_weight.html#%E5%88%86%E5%B8%83%E5%BC%8F%E6%9D%83%E9%87%8D%E5%88%87%E5%88%86%E4%B8%8E%E5%90%88%E5%B9%B6)（如果是共享盘也可以开启自动权重转换，使用完整权重）。

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，各个参数位置含义参见[使用指南](../../README_CN.md#三使用指南)。

在每台机器上运行以下命令，多机运行命令在每台机器上仅`node_num` 不同，从0开始计数，命令中主节点ip为第0个节点ip。

```shell
# 节点0，设0节点ip为192.168.1.1，作为主节点ip，总共64卡且每个节点8卡
# 节点0、节点1、...节点7 依此修改node_num，比如8机，node_num为0~7。
export MS_DEV_RUNTIME_CONF="inline:False"
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3 \
 --config research/llama3/llama3_70b/pretrain_llama3_70b.yaml \
 --load_checkpoint /path/model_dir/ \
 --train_dataset dataset_dir
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode train" \
 64 8 {主节点ip} 8118 {node_num} output/msrun_log False 300
```

## 微调

### 全参微调

MindFormers提供`Llama3-8b`单机多卡以及`Llama3-70b`多机多卡的微调示例，过程中使用`alpaca`数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

#### 单机训练

以Llama3-8b为例，支持**单机/多机训练**。

使用`finetune_llama3_8b.yaml`进行训练，或修改默认配置文件中的`model_config.seq_length`，使训练配置与数据集的`seq_length`保持一致。

执行命令启动微调任务，在单机上拉起任务。

```shell
# 单机8卡默认快速启动
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3 \
 --config research/llama3/llama3_8b/finetune_llama3_8b.yaml \
 --load_checkpoint model_dir/xxx.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune \
 --train_data dataset_dir"

# 参数说明
config:          配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode:        运行模式, 微调时设置为finetune
train_data:      训练数据集路径
```

#### 多机训练

多机多卡微调任务启动预训练类似，可参考[预训练章节](#预训练)并对启动命令进行如下修改：

1. 增加脚本入参`--load_checkpoint model_dir/xxx.ckpt`加载预训练权重
2. 设置启动脚本中的`--train_data dataset_dir`加载微调数据集
3. 设置启动脚本中的`--run_mode finetune`

## 推理

MindFormers提供`Llama3-8b`和`Llama3-70b`的推理功能，支持单卡、多卡以及多batch推理。

### 单卡推理

以`Llama3-8b`单卡推理为例。

1. 修改模型配置文件`research/llama3/llama3_8b/predict_llama3_8b.yaml`

   在对应位置修改为tokenizer.model文件的路径。

   ```yaml
   processor:
     tokenizer:
        vocab_file: "/path/tokenizer.model"
   ```

2. 执行推理命令

   ```shell
   python run_mindformer.py \
    --register_path research/llama3 \
    --config research/llama3/llama3_8b/predict_llama3_8b.yaml \
    --load_checkpoint /path/llama3_8b.ckpt \
    --auto_trans_ckpt False \
    --use_parallel False \
    --run_mode predict \
    --predict_data "I love Beijing, because"

   # 推理结果
   # I love Beijing, because it is a city of contrasts. It is a city of the past and the future, a city of the old and the new. ...
   ```

   多batch推理

   ```shell
   python run_mindformer.py \
    --register_path research/llama3 \
    --config research/llama3/llama3_8b/predict_llama3_8b.yaml \
    --load_checkpoint /path/llama3_8b.ckpt \
    --auto_trans_ckpt False \
    --use_parallel False \
    --run_mode predict \
    --predict_data "I love Beijing, because" "Hey how are you doing today?" \
    --predict_batch_size 2

   # 推理结果
   # I love Beijing, because it is a city of contrasts. It is a city of the past and the future, a city of the old and the new. ...
   # Hey how are you doing today? I am doing well. I am a little bit tired because I have been working a lot. ...
   ```

### 多卡推理

以`Llama3-70b`4卡推理为例。

1. 修改模型配置文件`research/llama3/llama3_70b/predict_llama3_70b.yaml`

   在对应位置修改为tokenizer.model文件的路径。

   ```yaml
   processor:
     tokenizer:
        vocab_file: "/path/tokenizer.model"
   ```

2. 执行推理命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/llama3 \
    --config research/llama3/llama3_70b/predict_llama3_70b.yaml \
    --load_checkpoint /path/llama3_70b.ckpt \
    --auto_trans_ckpt True \
    --use_parallel True \
    --run_mode predict \
    --predict_data \"I love Beijing, because\"" 4

   # 推理结果
   # Hey how are you doing today? I am doing great. I am a little tired but I am doing great. ...
   ```
