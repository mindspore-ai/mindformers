# Qwen3

## 模型描述

Qwen3 是 Qwen 系列最新一代的大型语言模型。基于广泛的训练，Qwen3 在推理、指令跟随、代理能力和多语言支持方面实现了突破性进展。

```text
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report},
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
```

## 支持规格

|    模型名称    |  规格  | 支持任务 | 模型架构  |                       支持设备                        |                  模型级别                  |
|:----------:|:----:|:----:|:-----:|:-------------------------------------------------:|:--------------------------------------:|
|   Qwen3    | 32B  | 预训练  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Validated](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 32B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Validated](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 14B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Preliminary](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 8B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Preliminary](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 4B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Preliminary](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 1.7B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Preliminary](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 0.6B  | 微调  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Preliminary](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 32B  |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Released](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)           |
|   Qwen3    | 0.6B |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Validated](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    |  8B  |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Validated](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)          |
|   Qwen3    | 1.7B |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Untested](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)           |
|   Qwen3    |  4B  |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Untested](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)           |
|   Qwen3    | 14B  |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |          [Untested](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E7%BA%A7%E5%88%AB%E4%BB%8B%E7%BB%8D)           |

说明：

- 模型架构：`Mcore` 表示 1.6.0 发布的新模型架构，`Legacy` 表示原有模型架构。详见[架构说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/overview.html)。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 版本配套

Qwen3 当前支持的版本配套如下。

|           | Mindspore Transformers | MindSpore | CANN | HDK |
|:---------:|:----------------------:|:---------:|:----:|:---:|
| 当前支持的版本 |           在研版本           |    在研版本     |  在研版本  | 在研版本  |

## 使用样例

MindSpore Transformers 支持使用 Qwen3 进行预训练和推理。各任务的整体使用流程如下：

| 任务  | 前期准备                    | 使用流程                       |
|:---:|:------------------------|:---------------------------|
| 预训练 | 环境安装 -> 预训练数据集下载        | 数据预处理 -> 修改任务配置 -> 启动预训练任务 |
| 微调 | 环境安装 -> 模型下载        |  修改任务配置 -> 启动微调任务 |
| 推理  |  环境安装 -> 模型下载                       |    修改任务配置 -> 启动推理任务                        |

### 前期准备

#### 环境安装

按照上述版本配套，参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从Hugging Face、ModelScope等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等（重头预训练不需加载权重）。 链接如下：

|      模型名称       | 下载链接                                                                                                                                                                                       | 说明 |
|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---|
| Qwen/Qwen3-0.6B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-0.6B)                                                              |    |
| Qwen/Qwen3-1.7B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-1.7B)                                                              |    |
|  Qwen/Qwen3-4B  | [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-4B)                                                                  |    |
|  Qwen/Qwen3-8B  | [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B)                                                                  |    |
| Qwen/Qwen3-14B  | [Hugging Face](https://huggingface.co/Qwen/Qwen3-14B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-14B)                                                                |    |
| Qwen/Qwen3-32B  | [Hugging Face](https://huggingface.co/Qwen/Qwen3-32B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-32B)         |    |

#### 数据集下载

MindSpore Transformers 以下面的数据集为例提供了 Qwen3 的预训练流程的使用案例，实际训练时可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节制作数据集。请在执行任务前提前下载所需数据集。链接如下：

| 任务  |    数据集名称     | 下载链接         | 说明 |
|:---:|:------------:|:-------------|:---|
| 预训练 | WikiText-103 | [Download](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) | 用于预训练的大规模文本数据集 |

### 预训练样例

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html)。

#### 1. 数据预处理

MindSpore Transformers 预训练阶段当前已支持[Megatron格式的数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)。用户可以参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节，使用 MindSpore 提供的工具将原始数据集转换为 Megatron 格式。

制作Megatron格式数据集，需要经过两个步骤。首先将原始文本数据集转换为jsonl格式数据，然后使用MindSpore Transformers提供的脚本将jsonl格式数据转换为Megatron格式的.bin和.idx文件。

- `wiki.train.tokens` 转为 `jsonl`格式数据

用户需要**自行将`wiki.train.tokens`数据集处理成jsonl格式的文件**。作为参考，文档末尾的[FAQ](#faq)部分提供了一个临时转换方案，用户需要根据实际需求自行开发和验证转换逻辑。

下面是jsonl格式文件的示例：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
...
```

- `jsonl`格式数据 转为 `bin`格式数据

MindSpore Transformers提供了数据预处理脚本`toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py`用于将jsonl格式的原始文本预料转换成.bin或.idx文件。

> 这里需要提前下载[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)模型的tokenizer文件。

例如：

```shell
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/wiki103-megatron \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/Qwen3-32B # 其他规格的模型可以调整为对应的tokenizer路径
```

> 运行完成后会生成`/path/to/wiki103-megatron_text_document.bin`和`/path/to/wiki103-megatron_text_document.idx`文件。
> 填写数据集路径时需要使用`/path/to/wiki103-megatron_text_document`，不需要带后缀名。

#### 2. 修改任务配置

MindSpore Transformers 提供了预训练任务的配置文件，用户可以根据实际情况修改配置文件。以下是一个示例配置文件片段，用户需要根据自己的数据集路径和其他参数进行相应修改。

- 数据集配置

```yaml
# Dataset configuration
train_dataset: &train_dataset
  data_loader:
    ...
    sizes:
      - 8000  # 数据集的大小，可以根据实际数据集大小进行调整
      ...
    config:
      ...
      data_path:  # 采样比例和Megatron格式数据集路径
        - '1'
        - "/path/to/wiki103-megatron_text_document" # 替换为实际的Megatron格式数据集路径，此处不带后缀名
```

数据集路径需要替换为实际的Megatron格式数据集路径。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 3. 启动预训练任务

通过指定模型路径和配置文件[configs/qwen3/pretrain_qwen3_32b_4k.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/pretrain_qwen3_32b_4k.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，进行16卡分布式训练。可以参考如下方式拉起两台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`port`设置为当前进程的端口号。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可实时查看训练状态（由于开启了流水并行，真实loss只显示在最后一个pipeline stage的日志中，其余pipeline stage会显示`loss`为`0`）

```shell
tail -f ./output/msrun_log/worker_15.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于Qwen3预训练的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

### 微调样例

SFT（Supervised Fine-Tuning，监督微调）采用有监督学习思想，是指在预训练模型的基础上，通过调整部分或全部参数，使模型更适应特定任务或数据集的过程。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/supervised_fine_tuning.html)。

MindSpore Transformers支持全参微调和LoRA高效微调两种SFT微调方式。全参微调是指在训练过程中对所有参数进行更新，适用于大规模数据精调，能获得最优的任务适应能力，但需要的计算资源较大。LoRA高效微调在训练过程中仅更新部分参数，相比全参微调显存占用更少、训练速度更快，但在某些任务中的效果不如全参微调。

#### 1. 配置文件修改

MindSpore Transformers 提供了微调任务的配置文件，用户可以根据实际情况修改配置文件。以下是一个示例配置文件片段，用户需要根据自己的数据集路径和其他参数进行相应修改。代码仓中提供了Qwen3-32B全参微调的配置文件[configs/qwen3/finetune_qwen3.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)，如果需要修改其他模型，例如Qwen3-14B、Qwen3-8B、Qwen3-4B、Qwen3-1.7B、Qwen3-0.6B等，可以参考该配置文件进行相应修改。并参考[附录](#附录)中的[并行配置建议](#并行配置建议)章节进行修改

**全参微调配置示例：**

```yaml
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: HFDataLoader
    path: "llm-wizard/alpaca-gpt4-data-zh" # alpaca风格数据集，确保网络环境能够访问huggingface，以实现自动下载数据集功能。
    # path: "json"  # 如果使用本地json文件离线加载数据集，可以取消注释下面两行，并注释掉上面一行
    # data_files: '/path/to/alpaca_gpt4_data_zh.json'
    handler:
      - type: take # 调用datasets库的take方法，取前n条数据用于示例
        n: 2000    # 取前2000条数据用于示例，实际使用时可以去掉这一行和上面一行
```

**LoRA微调配置示例：**

LoRA微调可以在单机8卡环境下运行，资源需求较低。以下是配置示例：

```yaml
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: HFDataLoader
    path: "llm-wizard/alpaca-gpt4-data-zh" # alpaca风格数据集，确保网络环境能够访问huggingface，以实现自动下载数据集功能。
    # path: "json"  # 如果使用本地json文件离线加载数据集，可以取消注释下面两行，并注释掉上面一行
    # data_files: '/path/to/alpaca_gpt4_data_zh.json'
    handler:
      - type: take # 调用datasets库的take方法，取前n条数据用于示例
        n: 2000    # 取前2000条数据用于示例，实际使用时可以去掉这一行和上面一行

# LoRA配置
model:
  model_config:
    ...
    # 在model_config层级下添加pet_config
    pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      lora_a_init: 'normal'
      lora_b_init: 'zeros'
      target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
      freeze_include: ['*']
      freeze_exclude: ['*lora*']
```

`pet_config`关键参数说明：

| 参数 | 说明 |
|:-----|:-----|
| `pet_type` | 参数高效微调技术类型 |
| `lora_rank` | LoRA的秩 |
| `lora_alpha` | LoRA缩放因子alpha |
| `lora_dropout` | LoRA中的dropout概率 |
| `lora_a_init` | LoRA的A矩阵初始化方式 |
| `lora_b_init` | LoRA的B矩阵初始化方式 |
| `target_modules` | 应用LoRA的模块，上述配置对word_embeddings、attention和mlp的权重矩阵应用LoRA |

#### 3. 启动微调任务

- **多机多卡训练（以Qwen3 32B全参微调为例）**

通过指定模型路径和配置文件[configs/qwen3/finetune_qwen3.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，启动卡分布式训练。

下列脚本可以参考如下方式拉起**两台Atlas 800T A2（64G）训练**。

在每台服务器上执行如下命令。设置：

- `total_rank_num=16`表示两台Atlas 800T A2（64G）共有`2x8=16`个NPU；
- `local_rank_num=8`表示每台Atlas 800T A2（64G）有8个NPU；
- `master_ip`为主节点IP地址；
- `node_rank`为每个节点的序号；
- `port`为当前进程的端口号（可在50000~65536中选择）。

```bash
total_rank_num=16
local_rank_num=8
master_ip=192.168.1.1
node_rank=0
port=50001
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/finetune_qwen3.yaml \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode train \
--pretrained_model_dir /path/to/Qwen3-32B \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 4 \
--parallel_config.pipeline_stage 4 \
--parallel_config.micro_batch_num 4 \
--recompute_config.recompute True" \
$total_rank_num $local_rank_num $master_ip $port $node_rank output/msrun_log False 7200
```

> `--pretrained_model_dir` 可以用于选择不同规格的Qwen3模型进行微调，例如`/path/to/Qwen3-14B`、`/path/to/Qwen3-8B`、`/path/to/Qwen3-4B`、`/path/to/Qwen3-1.7B`、`/path/to/Qwen3-0.6B`等。
> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`port`设置为当前进程的端口号。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可实时查看训练状态

```bash
tail -f ./output/msrun_log/worker_15.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于Qwen3全参微调的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

- **单机多卡训练（以Qwen3 32B LoRA微调为例）**

通过指定模型路径和配置文件[configs/qwen3/finetune_qwen3.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，启动卡分布式训练。

下列脚本可以参考如下方式拉起**一台Atlas 800T A2（64G）训练**。

```bash
total_rank_num=8
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/finetune_qwen3.yaml \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode train \
--pretrained_model_dir /path/to/Qwen3-32B \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 8 \
--parallel_config.pipeline_stage 1 \
--parallel_config.micro_batch_num 1 \
--recompute_config.recompute True" \
$total_rank_num
```

> `--pretrained_model_dir` 可以用于选择不同规格的Qwen3模型进行微调，例如`/path/to/Qwen3-14B`、`/path/to/Qwen3-8B`、`/path/to/Qwen3-4B`、`/path/to/Qwen3-1.7B`、`/path/to/Qwen3-0.6B`等。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可实时查看训练状态

```bash
tail -f ./output/msrun_log/worker_7.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于Qwen3 LoRA微调的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

#### 4. 权重合并

`output`目录下的`checkpoint`文件夹中会保存微调过程中生成的分布式safetensors权重文件，用户可以根据需要选择合适的权重进行权重合并，得到完整的safetensors权重，适用于后续推理流程。

使用MindSpore Transformers提供的safetensors权重合并脚本，按照如下方式进行safetensors权重合并。合并后的权重格式为完整权重。

```bash
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy False
```

参数说明

- **src_strategy_dirs**：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 `output/strategy/` 目录下。分布式权重需根据以下情况填写：

    - **源权重开启了流水线并行**：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 `ckpt_strategy_rank_x.ckpt` 文件合并，并在文件夹下生成 `merged_ckpt_strategy.ckpt`。如果已经存在 `merged_ckpt_strategy.ckpt`，可以直接填写该文件的路径。
    - **源权重未开启流水线并行**：权重转换可基于任一策略文件，填写任意一个 `ckpt_strategy_rank_x.ckpt` 文件的路径即可。

    **注意**：如果策略文件夹下已存在 `merged_ckpt_strategy.ckpt` 且仍传入文件夹路径，脚本会首先删除旧的 `merged_ckpt_strategy.ckpt`，再合并生成新的 `merged_ckpt_strategy.ckpt` 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。
- **mindspore_ckpt_dir**：分布式权重路径，请填写源权重所在文件夹的路径，源权重应按 `model_dir/rank_x/xxx.safetensors` 格式存放，并将文件夹路径填写为 `model_dir`。
- **output_dir**：目标权重的保存路径，默认值为 "/new_llm_data/******/ckpt/nbg3_31b/tmp"，即目标权重将放置在 `/new_llm_data/******/ckpt/nbg3_31b/tmp` 目录下。
- **file_suffix**：目标权重文件的命名后缀，默认值为 "1_1"，即目标权重将按照 `*1_1.safetensors` 格式查找。
- **has_redundancy**：合并的源权重是否是冗余的权重，默认为 `True`。
- **filter_out_param_prefix**：合并权重时可自定义过滤掉部分参数，过滤规则以前缀名匹配。如优化器参数"adam_"。
- **max_process_num**：合并最大进程数。默认值：64。

更多Safetensors权重相关的操作请参考[MindSpore Transformers - Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html#%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6)

### 推理样例

推理是指在预训练模型的基础上，利用已学习到的语言知识对新的输入数据进行预测或生成。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/inference.html)。

#### 1. 修改任务配置

MindSpore Transformers 提供了推理任务的[配置文件](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml)，用户可以根据实际情况修改此配置文件中的权重路径和其他参数。

当前推理可以直接复用Hugging Face的配置文件和tokenizer，并且在线加载Hugging Face的safetensors格式的权重，使用时配置修改如下：

```yaml
pretrained_model_dir: '/path/hf_dir'
parallel_config:
  data_parallel: 1
  model_parallel: 1
```

参数说明：

- pretrained_model_dir：Hugging Face模型目录路径，放置模型配置、Tokenizer等文件。`/path/hf_dir`中的内容如下：

```text
📂Qwen3-0.6B
├── 📄config.json
├── 📄generation_config.json
├── 📄merges.txt
├── 📄model-xxx.safetensors
├── 📄model-xxx.safetensors
├── 📄model.safetensors.index.json
├── 📄tokenizer.json
├── 📄tokenizer_config.json
└── 📄vocab.json
```

- data_parallel：数据并行，当前推理并不支持此并行策略，默认为1；
- model_parallel：模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_nu（即实际使用的卡数）。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 启动推理任务

使用 `run_mindformer` 统一脚本执行推理任务。

单卡推理可以直接执行[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，多卡推理需要借助[scripts/msrun_launcher.sh](https://atomgit.com/mindspore/mindformers/blob/master/scripts/msrun_launcher.sh)来启动。

run_mindformer.py的参数说明如下：

| 参数                             | 参数说明                                                       |
|:-------------------------------|:-----------------------------------------------------------|
| config                         | yaml配置文件的路径                                                |
| run_mode                       | 运行的模式，推理设置为predict                                         |
| use_parallel                   | 是否使用多卡推理                                                   |
| predict_data                   | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入                   |
| predict_batch_size             | 多batch推理的batch_size大小                                      |
| pretrained_model_dir           | Hugging Face模型目录路径，放置模型配置、Tokenizer等文件                     |
| parallel_config.data_parallel  | 数据并行，当前推理们模式下设置为1                                          |
| parallel_config.model_parallel | 模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数） |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

单卡推理：

当使用完整权重推理时，推荐使用默认[配置](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml)，执行以下命令即可启动推理任务：

```shell
python run_mindformer.py \
--config configs/qwen3/predict_qwen3.yaml \
--run_mode predict \
--use_parallel False \
--pretrained_model_dir '/path/hf_dir' \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 1 \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 `text_generation_result.txt` 文件中。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

多卡推理：

多卡推理的配置要求与单卡存在差异，需参考下面修改配置：

1. 模型并行model_parallel的配置和使用的卡数需保持一致，下文用例为2卡推理，需将model_parallel设置成2；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

当使用完整权重推理时，需要开启在线切分方式加载权重，参考以下命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 2
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 text_generation_result.txt 文件中。详细日志可通过`./output/msrun_log`目录查看。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

多卡多batch推理：

多卡多batch推理的启动方式可参考上述[多卡推理](#多卡推理)，但是需要增加`predict_batch_size`的入参，并修改`predict_data`的入参。

`input_predict_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
```

以完整权重推理为例，可以参考以下命令启动推理任务：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data path/to/input_predict_data.txt" 2
```

推理结果查看方式，与多卡推理相同。

多机多卡推理：

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3/predict_qwen3.yaml" \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data 帮助我制定一份去上海的旅游攻略" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`$local_worker`设置为当前节点上拉起的进程数(当前机器使用的卡数)；将`$worker_num`设置为参与任务的进程总数(使用的总卡数)；将`$port`设置为启动任务的端口号。

推理结果查看方式，与多卡推理相同。

## 附录

### 模型文件说明

Qwen3的模型文件包括以下内容：

```text
📦mindformers
├── 📂mindformers
│   └── 📂models
│       └── 📂qwen3
│           ├── 📄__init__.py                   # Qwen3模块初始化文件
│           ├── 📄configuration_qwen3.py        # Qwen3模型配置类定义
│           ├── 📄modeling_qwen3.py             # Qwen3模型主体实现
│           ├── 📄modeling_qwen3_infer.py       # Qwen3推理模型实现
│           ├── 📄modeling_qwen3_train.py       # Qwen3训练模型实现
│           └── 📄utils.py                      # Qwen3工具函数和基础类
├── 📂configs
│   └── 📂qwen3
│       ├── 📄pretrain_qwen3_32b_4k.yaml       # Qwen3-32B 4k 预训练配置
│       ├── 📄predict_qwen3.yaml               # Qwen3推理配置
│       └── 📄parallel_speed_up.json           # 数据集并行通信配置
└── 📄run_mindformer.py                        # 主要执行脚本
```

### 并行配置建议

以下配置为训练或推理场景下，不同模型规格的推荐配置。其中部分配置为经过验证的最佳配置，部分配置为可以运行的配置。用户可根据实际情况选择合适的配置。

> 注意：max_device_memory 在 Atlas 800T A2 和 Atlas 900 A3 SuperPoD 等机器上一般设置≤60GB，在 Atlas 800I A2 上一般设置≤30GB。

- 预训练/全参微调：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>序列长度</th>
    <th>并行配置</th>
    <th>重计算配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>32B</td>
    <td>2 × Atlas 800T A2 (8P)</td>
    <td>16</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 4
  pipeline_stage: 4
  micro_batch_num: 4
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>14B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>8B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>4B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>1.7B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>0.6B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
</table>

- LoRA微调：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>序列长度</th>
    <th>并行配置</th>
    <th>重计算配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>32B</td>
    <td>Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>14B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>8B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>4B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>1.7B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>0.6B</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
</table>

- 推理：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>并行配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>32B</td>
    <td>1 × Atlas 800T A2 (2P)</td>
    <td>2</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 2</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Released </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>0.6B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>8B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>1.7B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>4B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>14B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Untested </td>
  </tr>
</table>

### FAQ

Q1：我有两台Atlas 800T A2服务器，如何进行Qwen3的预训练？拉起任务的指令是什么？

A1：根据指导修改配置后，参考如下命令拉起任务：

- 机器1 IP: 192.168.1.1 （作为主节点）

```bash
# 机器1的启动指令
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

- 机器2 IP: 192.168.1.2

```bash
# 机器2的启动指令
master_ip=192.168.1.1
node_rank=1
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

Q2: 数据集准备部分中，应该如何将`wiki.train.tokens` 转为 `jsonl`格式数据？

A2: [社区issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY)中提供了一个临时转换脚本，仅作为参考使用。用户需要根据自己的数据特点和需求，自行开发和验证适合的转换逻辑。

Q3：如果修改了配置中的参数，使用`run_mindformer.py`拉起任务时，还需要重新传参吗？

A3：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py`会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

Q4：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？

A4：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从50000~65536中选取，避免端口号冲突的情况发生。

Q5：我想看看我训练下来的权重效果怎么样，可以直接使用训练权重做推理吗？

A5：当然可以！你可以通过以下两种方式进行推理：

1. **直接使用训练权重进行推理**，可以参考[《训练后模型进行评测》](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/evaluation.html#%E8%AE%AD%E7%BB%83%E5%90%8E%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%AF%84%E6%B5%8B)文档，使用去优化器合并的训练权重进行推理。
2. **反转训练权重为 Hugging Face 格式，复用 Hugging Face 生态进行推理**，可以参考 [Qwen3 反转脚本](../../toolkit/weight_convert/qwen3/README.md)进行权重反转后，再进行推理任务。