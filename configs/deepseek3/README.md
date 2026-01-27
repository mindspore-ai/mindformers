# DeepSeek-V3

## 模型描述

DeepSeek-V3 系列模型是深度求索（DeepSeek）公司推出的一款高性能开源大语言模型，具有强大的自然语言处理能力。该模型在多个领域展现出了卓越的表现，包括代码生成、数学推理、逻辑推理和自然语言理解等。模型总参数 6710 亿，激活参数 370 亿， 其中 DeepSeek-R1 模型和 DeepSeek-V3.1 模型是基于 DeepSeek-V3 Base 模型进一步优化的推理特化模型，通过多阶段强化学习训练，在复杂推理、数学和编程任务上达到国际顶尖水平，同时大幅降低幻觉率。

```text
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Haowei Zhang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Li and Hui Qu and J. L. Cai and Jian Liang and Jianzhong Guo and Jiaqi Ni and Jiashi Li and Jiawei Wang and Jin Chen and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Xu and Leyi Xia and Liang Zhao and Litong Wang and Liyue Zhang and Meng Li and Miaojun Wang and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingming Li and Ning Tian and Panpan Huang and Peiyi Wang and Peng Zhang and Qiancheng Wang and Qihao Zhu and Qinyu Chen and Qiushi Du and R. J. Chen and R. L. Jin and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runxin Xu and Ruoyu Zhang and Ruyi Chen and S. S. Li and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaoqing Wu and Shengfeng Ye and Shengfeng Ye and Shirong Ma and Shiyu Wang and Shuang Zhou and Shuiping Yu and Shunfeng Zhou and Shuting Pan and T. Wang and Tao Yun and Tian Pei and Tianyu Sun and W. L. Xiao and Wangding Zeng and Wanjia Zhao and Wei An and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and X. Q. Li and Xiangyue Jin and Xianzu Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaojin Shen and Xiaokang Chen and Xiaokang Zhang and Xiaosha Chen and Xiaotao Nie and Xiaowen Sun and Xiaoxiang Wang and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xinnan Song and Xinxia Shan and Xinyi Zhou and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and Y. K. Li and Y. Q. Wang and Y. X. Wei and Y. X. Zhu and Yang Zhang and Yanhong Xu and Yanhong Xu and Yanping Huang and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Li and Yaohui Wang and Yi Yu and Yi Zheng and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Ying Tang and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yu Wu and Yuan Ou and Yuchen Zhu and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yukun Zha and Yunfan Xiong and Yunxian Ma and Yuting Yan and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhen Huang and Zhen Zhang and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhipeng Xu and Zhiyu Wu and Zhongyu Zhang and Zhuoshu Li and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Ziyi Gao and Zizheng Pan},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## 支持规格

|     模型名称      |    规格    |   支持任务    | 模型架构  |                       支持设备                        |        模型级别         |
|:-------------:|:--------:|:---------:|:-----:|:-------------------------------------------------:|:-------------------:|
|  DeepSeek-V3  |   671B   | 预训练/微调/推理 | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Preliminary](#模型级别介绍)     |
|  DeepSeek-R1  |   671B   | 预训练/微调/推理 | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Preliminary](#模型级别介绍)     |
| DeepSeek-V3.1 |   671B   | 预训练/微调/推理 | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Preliminary](#模型级别介绍)     |

说明：

- 模型架构：`Mcore` 表示新模型架构。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 版本配套

DeepSeek-V3 当前支持的版本配套如下。

|           | Mindspore Transformers | MindSpore | CANN  |  HDK   |
|:---------:|:----------------------:|:---------:|:-----:|:------:|
| 当前支持的版本 |         1.8.0          |   2.7.2   | 8.5.0 | 25.5.0 |

## 使用样例

MindSpore Transformers 支持使用 DeepSeek-V3 进行预训练，微调和推理。各任务的整体使用流程如下：

| 任务  | 前期准备                    | 使用流程                       |
|:---:|:------------------------|:---------------------------|
| 预训练 | 环境安装 -> 预训练数据集下载        | 数据预处理 -> 修改任务配置 -> 启动预训练任务 |
| 微调  | 环境安装 -> 模型下载  | 修改任务配置 -> 启动微调任务  |
| 推理  | 环境安装 -> 模型下载  | 修改任务配置 -> 启动推理任务  |

### 前期准备

#### 环境安装

按照上述版本配套，参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从Hugging Face、ModelScope等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等（重头预训练不需加载权重）。 链接如下：

| 模型名称                         |                                                                     下载链接                                                                     | 说明 |
|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------:|:---|
| deepseek-ai/DeepSeek-V3      |      [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3)      |    |
| deepseek-ai/DeepSeek-V3-Base | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-Base) |    |
| deepseek-ai/DeepSeek-V3-0324 | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324) |    |
| deepseek-ai/DeepSeek-R1 |                                [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1)                                |    |
| deepseek-ai/DeepSeek-R1-Zero |                                      [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Zero)                                      |    |
| deepseek-ai/DeepSeek-R1-0528 |                                      [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528)                                      |    |
| deepseek-ai/DeepSeek-V3.1 |                                       [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1)                                        |    |
| deepseek-ai/DeepSeek-V3.1-Base |                                     [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Base)                                     |    |
| deepseek-ai/DeepSeek-V3.1-Terminus |                                   [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Terminus)                                   |    |

#### 数据集下载

MindSpore Transformers 以下面的数据集为例提供了 DeepSeek-V3 的预训练和微调流程的使用案例，实际训练时可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节制作数据集。请在执行任务前提前下载所需数据集。链接如下：

| 任务  |    数据集名称     | 下载链接                                                                                            | 说明             |
|:---:|:------------:|:------------------------------------------------------------------------------------------------|:---------------|
| 预训练 | WikiText-103 | [Download](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) | 用于预训练的大规模文本数据集 |
| 微调  |    Alpaca    | [Download](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)            | 用于微调的大规模文本数据集  |

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

> 这里需要提前下载[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)模型的tokenizer文件。

例如：

```shell
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/wiki103-megatron \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/DeepSeek-V3 # 其他规格的模型可以调整为对应的tokenizer路径
```

> 运行完成后会生成`/path/to/wiki103-megatron_text_document.bin`和`/path/to/wiki103-megatron_text_document.idx`文件。
> 填写数据集路径时需要使用`/path/to/wiki103-megatron_text_document`，不需要带后缀名。

#### 2. 修改任务配置

MindSpore Transformers 提供了两份预训练任务的配置文件分别为：满配的DeepSeek-V3配置文件[configs/deepseek3/pretrain_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/pretrain_deepseek3_671b.yaml)和缩层到12b的DeepSeek-V3配置文件[configs/deepseek3/pretrain_deepseek3_12b_16p_pp16.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/pretrain_deepseek3_12b_16p_pp16.yaml)，这两份配置分别基于32台和2台Atlas 800T A2（64G），使用WikiText-103数据集进行预训练，用户可以根据实际情况修改配置文件。以下是一个示例配置文件片段，用户需要根据自己的数据集路径和其他参数进行相应修改，对完整模型进行预训练。

- 数据集配置

    ```yaml
    # Dataset configuration
    train_dataset: &train_dataset
      data_loader:
        ...
        sizes:
          - 128000  # 数据集的大小，可以根据实际数据集大小进行调整
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

通过指定模型路径和配置文件[configs/deepseek3/pretrain_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/pretrain_deepseek3_671b.yaml)或者[configs/deepseek3/pretrain_deepseek3_12b_16p_pp16.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/pretrain_deepseek3_12b_16p_pp16.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，进行分布式训练。可以参考如下方式拉起32台Atlas 800T A2（64G）进行预训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`255`；`port`为当前进程的端口号。

```shell
master_ip=192.168.1.1
node_rank=0
port=8118

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/deepseek3/pretrain_deepseek3_671b.yaml" \
256 8 $master_ip $port $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可查看训练状态（由于开启了流水并行，真实loss只显示在最后一个stage的日志中，其余卡显示`loss`为`0`）

```shell
tail -f ./output/msrun_log/worker_255.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于DeepSeek-V3预训练的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

### 微调样例

SFT（Supervised Fine-Tuning，监督微调）采用有监督学习思想，是指在预训练模型的基础上，通过调整部分或全部参数，使模型更适应特定任务或数据集的过程。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/supervised_fine_tuning.html)。

MindSpore Transformers支持全参微调和LoRA高效微调两种SFT微调方式。全参微调是指在训练过程中对所有参数进行更新，适用于大规模数据精调，能获得最优的任务适应能力，但需要的计算资源较大。LoRA高效微调在训练过程中仅更新部分参数，相比全参微调显存占用更少、训练速度更快，但在某些任务中的效果不如全参微调。

#### 1. 修改任务配置

MindSpore Transformers 提供了微调任务的配置文件，用户可以根据实际情况修改配置文件。以下是示例配置文件片段，用户需要根据自己的数据集路径和其他参数进行相应修改。代码仓中提供了满配DeepSeek-V3全参微调的配置文件[configs/deepseek3/finetune_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/finetune_deepseek3_671b.yaml)和缩层到12b的DeepSeek-V3全参微调的配置文件[configs/deepseek3/finetune_deepseek3_12b_16p_pp16.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/finetune_deepseek3_12b_16p_pp16.yaml)，如果需要修改其他模型，例如DeepSeek-V3-0324、DeepSeek-R1、DeepSeek-R1-0528、DeepSeek-V3.1等，可以参考该配置文件进行相应修改。并参考[附录](#附录)中的[并行配置建议](#并行配置建议)章节进行修改。

**全参微调配置示例：**

仓库中提供了两份微调配置文件，使用alpaca数据集进行全参微调。用户需要根据自己的数据集路径和其他参数进行相应修改，并参考附录中的并行配置建议章节进行修改。

- 修改数据集配置

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

LoRA微调可以在多机多卡环境下运行，资源需求较低。用户可以参考全参微调配置示例，做出如下修改：

```yaml
pretrained_model_dir: "/path/to/DeepSeek-V3"

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
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.
      lora_a_init: 'normal'
      lora_b_init: 'zeros'
      target_modules: '.*linear_qkv|.*linear_fc1|.*linear_fc2'
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

##### 3. 启动微调任务

通过指定模型路径和配置文件[configs/deepseek3/finetune_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/finetune_deepseek3_671b.yaml)或者[configs/deepseek3/finetune_deepseek3_12b_16p_pp16.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/finetune_deepseek3_12b_16p_pp16.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，启动多卡分布式训练。

下列脚本可以参考如下方式拉起**2台Atlas 800T A2（64G）训练**：

在每台服务器上执行如下命令。设置：

- `total_rank_num=16`表示2台Atlas 800T A2（64G）共有`2x8=16`个NPU；
- `local_rank_num=8`表示每台Atlas 800T A2（64G）有8个NPU；
- `master_ip`为主节点IP地址；
- `node_rank`为每个节点的序号；
- `port`为当前进程的端口号（可在50000~65536中选择）。

```shell
total_rank_num=16
local_rank_num=8
master_ip=192.168.1.1
node_rank=0
port=8118
export MS_DEV_RUNTIME_CONF="multi_stream:true"

bash scripts/msrun_launcher.sh "run_mindformer.py \
--pretrained_model_dir /path/checkpoint_path \
--load_ckpt_format safetensors \
--output_dir ./output \
--auto_trans_ckpt True \
--config /path/to/finetune_deepseek3_671b.yaml \
--run_mode finetune" \
$total_rank_num $local_rank_num $master_ip $port $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。
> pretrained_model_dir修改为原始权重路径，output_dir修改为用户想要保存训练后权重的路径。
> 如开启自动权重切分auto_trans_ckpt，pretrained_model_dir路径与output_dir路径需要是多机共享路径。
> 该配置在通信并发下有带宽抢占引发的性能劣化，通过配置`MS_DEV_RUNTIME_CONF="multi_stream:true"`控制通信单流来规避该劣化。
> `--pretrained_model_dir` 可以用于选择不同规格的DeepSeek-V3模型进行微调，例如`/path/to/DeepSeek-V3-0324`、`/path/to/DeepSeek-R1`、`/path/to/DeepSeek-R1-0528`、`/path/to/DeepSeek-V3.1`等。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，在node_rank最后的机器使用以下命令可查看训练状态：

```shell
tail -f ./output/msrun_log/worker_15.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于DeepSeek-V3 全参微调的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

### 推理样例

DeepSeek-V3模型总参数量671B，Bfloat16权重参内存占用高达1.4T，最少需要4台Atlas 800T A2。MindSpore Transformers可以通过统一脚本实现单卡多卡以及多机的推理。

#### 1. 修改任务配置

MindSpore Transformers 提供了推理任务的配置文件[predict_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/predict_deepseek3_671b.yaml)，用户可以根据实际情况修改此配置文件中的权重路径和其他参数。

当前推理可以直接复用Hugging Face的配置文件和tokenizer，并且在线加载Hugging Face的safetensors格式的权重，使用时配置修改如下：

```yaml
pretrained_model_dir: '/path/hf_dir'
parallel_config:
  data_parallel: 1
  model_parallel: 32
```

参数说明：

- pretrained_model_dir：Hugging Face模型目录路径，放置模型配置、Tokenizer等文件。`/path/hf_dir`中的内容如下：

```text
📂DeepSeek-V3
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

- data_parallel：数据并行，默认值为 1，执行大小EP推理时需要修改此配置。
- model_parallel：模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数）。
- expert_parallel：专家并行，默认值为 1，当执行大小EP推理时需要修改此配置。

> 当执行大小EP推理的时候，data_parallel及model_parallel指定attn及ffn-dense部分的并行策略，expert_parallel指定moe部分路由专家并行策略，data_parallel * model_parallel可被expert_parallel整除。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 本地纯TP推理

使用 `run_mindformer` 统一脚本执行推理任务。

DeepSeek-V3因为参数量只能用多卡推理，多卡推理需要借助scripts/msrun_launcher.sh来启动。

run_mindformer.py的参数说明如下：

| 参数                             | 参数说明                                                      |
|:-------------------------------|:----------------------------------------------------------|
| config                         | yaml配置文件的路径                                               |
| run_mode                       | 运行的模式，推理设置为predict                                        |
| use_parallel                   | 是否使用多卡推理                                                  |
| predict_data                   | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入                  |
| predict_batch_size             | 多batch推理的batch_size大小                                     |
| pretrained_model_dir           | Hugging Face模型目录路径，放置模型配置、Tokenizer等文件                    |
| parallel_config.data_parallel  | 数据并行，当前推理模式下设置为1                                         |
| parallel_config.model_parallel | 模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数） |
| parallel_config.expert_parallel  | 数据并行，当前推理模式下设置为1                                         |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

多机多卡推理：

DeepSeek-V3总参数量671B，只能进行多机多卡推理，在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 32 \
 --predict_data 请介绍一下北京" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`$local_worker`设置为当前节点上拉起的进程数(当前机器使用的卡数)；将`$worker_num`设置为参与任务的进程总数(使用的总卡数)；将`$port`设置为启动任务的端口号；`$parallel_config.model_parallel`需要设置成实际卡数。

推理结果会保存到当前目录下的 text_generation_result.txt 文件中，推理过程中的日志可通过如下命令查看：

```shell
tail -f ./output/msrun_log/worker_0.log
```

#### 3. 本地大EP推理

大EP，指的是路由专家仅仅按EP分组，不做其他切分。DeepSeek-V3总参数量671B，非MoE参数量大致为20B，大EP浮点推理至少为64卡，即四台A3机器或者八台A2机器。相较于纯tp推理，启动命令的入参需要修改并行配置和`predict_data`的入参，并且增加`predict_batch_size`的入参为DP的倍数，具体执行命令如下:

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 64 \
 --parallel_config.model_parallel 1 \
 --parallel_config.expert_parallel 64 \
 --predict_data path/to/input_data.txt \
 --predict_batch_size 64" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

`input_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
请介绍一下北京
请介绍一下北京
请介绍一下北京
......
请介绍一下北京
```

推理结果和过程日志查看同本地纯TP推理。

#### 4. 本地小EP推理

小EP推理，指的是路由专家不仅仅按EP分组，同时专家本身被TP切分，浮点推理至少为32卡，即两台A3机器或者八台A2机器。相较于纯tp推理，启动命令的入参需要修改并行配置和`predict_data`的入参，并且增加`predict_batch_size`的入参为DP的倍数，具体执行命令如下:

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 4 \
 --parallel_config.model_parallel 8 \
 --parallel_config.expert_parallel 4 \
 --predict_data path/to/input_data.txt \
 --predict_batch_size 4" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

`input_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
请介绍一下北京
请介绍一下北京
请介绍一下北京
......
请介绍一下北京
```

推理结果和过程日志查看同本地纯TP推理。

#### 5. 启动服务化推理任务

服务化推理支持量化、大小ep等特性，可以查看以下文档：[服务化推理](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/deployment.html)

## 附录

### 模型文件说明

DeepSeek-V3的模型文件包括以下内容：

```text
📦mindformers
├── 📂mindformers
│   └── 📂models
│       └── 📂deepseek3
│           ├── 📄__init__.py                         # DeepSeek-V3模块初始化文件
│           ├── 📄configuration_deepseek_v3.py        # DeepSeek-V3模型配置类定义
│           ├── 📄modeling_deepseek_v3.py             # DeepSeek-V3模型主体实现
│           ├── 📄modeling_deepseek_v3_infer.py       # DeepSeek-V3推理模型实现
│           ├── 📄modeling_deepseek_v3_train.py       # DeepSeek-V3训练模型实现
│           └── 📄utils.py                            # DeepSeek-V3工具函数和基础类
├── 📂configs
│   └── 📂deepseek3
│       ├── 📄pretrain_deepseek3_671b.yaml            # DeepSeek-V3预训练配置
│       ├── 📄finetune_deepseek3_671b.yaml            # DeepSeek-V3全参微调配置
│       ├── 📄pretrain_deepseek3_12b_16p_pp16.yaml    # DeepSeek-V3 12B预训练配置
│       ├── 📄finetune_deepseek3_12b_16p_pp16.yaml    # DeepSeek-V3 12B全参微调配置
│       └── 📄predict_deepseek3_671b.yaml             # DeepSeek-V3推理配置
└── 📄run_mindformer.py                               # 主要执行脚本
```

### 并行配置建议

以下配置为训练或推理场景下，不同模型规格的推荐配置。其中部分配置为经过验证的最佳配置，部分配置为可以运行的配置。用户可根据实际情况选择合适的配置。

> 注意：max_device_memory 在 Atlas 800T A2 和 Atlas 900 A3 SuperPoD 等机器上一般设置≤60GB，在 Atlas 800I A2 上一般设置≤30GB。

- 预训练/微调：

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
    <td>DeepSeek-V3</td>
    <td>671B</td>
    <td>32 x Atlas 800T A2(64G)</td>
    <td>256</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: 4
  model_parallel: 8
  pipeline_stage: 8
  expert_parallel: 32
  micro_batch_num: 32
  use_seq_parallel: True
  gradient_aggregation_group: 4</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  mp_comm_recompute: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "56GB"</code></pre>
    </td>
    <td> Preliminary </td>
  </tr>
  <tr>
    <td>DeepSeek-V3</td>
    <td>12B</td>
    <td>2 x Atlas 800T A2(64G)</td>
    <td>16</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 1
  pipeline_stage: 16
  expert_parallel: 1
  micro_batch_num: 16
  vocab_emb_dp: True
  use_seq_parallel: False
  gradient_aggregation_group: 4</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: False
  recompute_slice_activation: False</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Released </td>
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
    <td>DeepSeek-V3/R1</td>
    <td>671B</td>
    <td>4 × Atlas 800T A2 (8P)</td>
    <td>32</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: 1
  model_parallel: 32</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Released </td>
  </tr>
</table>

### FAQ

Q1：我有1台Atlas 800T A2（64G）服务器，配置已修改完成，如何进行DeepSeek-V3的LoRA微调？拉起任务的指令是什么？

A1：根据指导修改配置后，可以参考如下方式拉起1台Atlas 800T A2（64G）训练。

通过指定模型路径和配置文件[configs/deepseek3/finetune_deepseek3_671b.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/deepseek3/finetune_deepseek3_671b.yaml)以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，启动卡分布式训练。

下列脚本可以参考如下方式拉起**一台Atlas 800T A2（64G）训练**。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/deepseek3/lora_finetune_deepseek3_671b.yaml \
--run_mode finetune"
```

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可实时查看训练状态

```bash
tail -f ./output/msrun_log/worker_7.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于DeepSeek-V3 LoRA微调的相关问题，可以在MindSpore Transformers的AtomGit仓库中[提交ISSUE](https://atomgit.com/mindspore/mindformers/issues/new)以获取支持。

Q2: 数据集准备部分中，应该如何将`wiki.train.tokens` 转为 `jsonl`格式数据？

A2: [社区issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY)中提供了一个临时转换脚本，仅作为参考使用。用户需要根据自己的数据特点和需求，自行开发和验证适合的转换逻辑。

Q3：如果修改了配置中的参数，使用`run_mindformer.py`拉起任务时，还需要重新传参吗？

A3：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py`会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

Q4：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？

A4：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从50000~65536中选取，避免端口号冲突的情况发生。

Q5：我想看看我训练下来的权重效果怎么样，可以直接使用训练权重做推理吗？

A5：当然可以！你可以通过以下两种方式进行推理：

1. **直接使用训练权重进行推理**，可以参考[《训练后模型进行评测》](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/evaluation.html#%E8%AE%AD%E7%BB%83%E5%90%8E%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%AF%84%E6%B5%8B)文档，使用去优化器合并的训练权重进行推理。
2. **反转训练权重为 Hugging Face 格式，复用 Hugging Face 生态进行推理**，可以参考 [DeepSeek-V3 反转脚本](../../toolkit/weight_convert/deepseekv3/README.md)进行权重反转后，再进行推理任务。

更多FAQ请查看[官网FAQ](https://www.mindspore.cn/mindformers/docs/zh-CN/master/faq/model_related.html)