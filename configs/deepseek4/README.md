# DeepSeek-V4

## 模型描述

DeepSeek-V4 系列模型是深度求索（DeepSeek）公司推出的第四代高性能开源大语言模型。该系列包含两个 MoE（混合专家）模型——**DeepSeek-V4-Pro** 和 **DeepSeek-V4-Flash**，两款模型均原生支持 **100 万 token 的超长上下文**。

DeepSeek-V4 系列在架构和优化方面实现了多项关键升级：

- **混合注意力架构**：结合了压缩稀疏注意力（CSA）和重度压缩注意力（HCA），大幅提升长上下文处理效率。
- **流形约束超连接（mHC）**：增强传统残差连接，提升信号跨层传播的稳定性。
- **Muon 优化器**：实现更快的收敛速度和更高的训练稳定性。
- **DSA2 注意力机制**：融合了 DSA 与 NSA 两种稀疏注意力方案。
- **MoE 架构**：每层配置 384 个专家，每次推理激活 6 个专家。

两款模型均在超过 32T 的高质量 token 上完成预训练。DeepSeek 官方表示：“从现在开始，1M（一百万）上下文将是 DeepSeek 所有官方服务的标配。”

```text
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI and ...},
      year={2026},
      eprint={2606.19348},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.19348},
}
```

## 支持规格

| 模型名称 | 规格 | 支持任务 | 模型架构 | 支持设备 | 模型级别 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-V4-Pro | 1.6T（激活49B） | 预训练 | Mcore | Atlas 800T A2 / Atlas 800I A2 / Atlas 900 A3 SuperPoD | [Preliminary](#模型级别介绍) |
| DeepSeek-V4-Flash | 284B（激活13B） | 预训练 | Mcore | Atlas 800T A2 / Atlas 800I A2 / Atlas 900 A3 SuperPoD | [Preliminary](#模型级别介绍) |

说明：

- 模型架构：`Mcore` 表示新模型架构。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 使用样例

MindSpore Transformers 支持使用 DeepSeek-V4 进行预训练，微调和推理。各任务的整体使用流程如下：

| 任务 | 前期准备 | 使用流程 |
| :---: | :--- | :--- |
| 预训练 | 环境安装 -> 预训练数据集下载 | 数据预处理 -> 修改任务配置 -> 启动预训练任务 |

### 前期准备

#### 环境安装

参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从 Hugging Face、ModelScope 等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等（重头预训练不需加载权重）。链接如下：

| 模型名称 | 下载链接 |
| :--- | :--- |
| deepseek-ai/DeepSeek-V4-Pro | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |
| deepseek-ai/DeepSeek-V4-Pro-Base | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro-Base) |
| deepseek-ai/DeepSeek-V4-Flash | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) |
| deepseek-ai/DeepSeek-V4-Flash-Base | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-Base) |

#### 数据集下载

MindSpore Transformers 以下面的数据集为例提供了 DeepSeek-V4 的预训练和微调流程的使用案例，实际训练时可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节制作数据集。请在执行任务前提前下载所需数据集。链接如下：

| 任务 | 数据集名称 | 下载链接 | 说明 |
| :---: | :---: | :--- | :--- |
| 预训练 | WikiText-103 | [Download](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) | 用于预训练的大规模文本数据集 |
| 微调 | Alpaca | [Download](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | 用于微调的大规模文本数据集 |

### 预训练样例

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。在 MindSpore 官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html)。

#### 1. 数据预处理

MindSpore Transformers 预训练阶段当前已支持[Megatron格式的数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)。用户可以参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节，使用 MindSpore 提供的工具将原始数据集转换为 Megatron 格式。

制作 Megatron 格式数据集，需要经过两个步骤。首先将原始文本数据集转换为 jsonl 格式数据，然后使用 MindSpore Transformers 提供的脚本将 jsonl 格式数据转换为 Megatron 格式的 `.bin` 和 `.idx` 文件。

- `wiki.train.tokens` 转为 `jsonl` 格式数据

用户需要**自行将 `wiki.train.tokens` 数据集处理成 jsonl 格式的文件**。作为参考，文档末尾的 [FAQ](#faq) 部分提供了一个临时转换方案，用户需要根据实际需求自行开发和验证转换逻辑。

下面是 jsonl 格式文件的示例：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
...
```

- `jsonl` 格式数据 转为 `bin` 格式数据

MindSpore Transformers 提供了数据预处理脚本 `toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py` 用于将 jsonl 格式的原始文本预料转换成 `.bin` 和 `.idx` 文件。

> 这里需要提前下载 [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 或 [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 模型的 tokenizer 文件（两者 tokenizer 相同）。

例如：

```shell
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/wiki103-megatron \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/DeepSeek-V4-Pro  # 也可使用 DeepSeek-V4-Flash 的 tokenizer 路径
```

> 运行完成后会生成 `/path/to/wiki103-megatron_text_document.bin` 和 `/path/to/wiki103-megatron_text_document.idx` 文件。
> 填写数据集路径时需要使用 `/path/to/wiki103-megatron_text_document`，不需要带后缀名。

#### 2. 修改任务配置

MindSpore Transformers 提供了 DeepSeek-V4 Pro 和 Flash 两个版本预训练任务的全层配置文件和减层配置文件。用户可以根据实际硬件资源和需求选择合适的配置。（如果需要使用全层配置，并行需要自行进行调试）

- **DeepSeek-V4-Pro（1.6T）**：全层配置文件 `configs/deepseek4/pretrain_deepseek4_pro.yaml`。
- **DeepSeek-V4-Flash（284B）**：全层配置文件 `configs/deepseek4/pretrain_deepseek4_flash.yaml`。
- **DeepSeek-V4-Pro（1.6T）**：减层配置文件 `configs/deepseek4/pretrain_deepseek4_pro_reduction.yaml`。
- **DeepSeek-V4-Flash（284B）**：减层配置文件 `configs/deepseek4/pretrain_deepseek4_flash_reduction.yaml`。

以下是一个示例配置文件片段（以 Flash 为例），用户需要根据自己的数据集路径和其他参数进行相应修改。

- 数据集配置

```yaml
# Dataset configuration
train_dataset: &train_dataset
  dataloader:
    ...
    sizes:
      - 16000  # 数据集的大小，可以根据实际数据集大小进行调整
      ...
    config:
      ...
      data_path:  # 采样比例和 Megatron 格式数据集路径
        - '1'
        - "/path/to/wiki103-megatron_text_document" # 替换为实际的 Megatron 格式数据集路径，此处不带后缀名
```

数据集路径需要替换为实际的 Megatron 格式数据集路径。

#### 3. 启动预训练任务

通过指定模型路径和配置文件以 msrun 的方式启动 `run_mindformer.py` 脚本，进行分布式训练。

**对于 DeepSeek-V4-Pro-Reduction**，参考以下方式拉起 双机 Atlas 800T A2（64G）进行预训练：

在每台服务器上执行如下命令。设置 `master_ip` 为主节点 IP 地址，即 `Rank 0` 服务器的 IP；`node_rank` 为每个节点的 Rank 序号，从 `0` 到 `15`；`port` 为当前进程的端口号。

```shell
master_ip=192.168.1.1
node_rank=0
port=8118

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--mode 1 \
--config configs/deepseek4/pretrain_deepseek4_pro_reduction.yaml" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

**对于 DeepSeek-V4-Flash-Reduction**，参考以下方式拉起 单机 Atlas 800T A2（64G）进行预训练：

```shell
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--mode 1 \
--config configs/deepseek4/pretrain_deepseek4_flash_reduction.yaml" 8
```

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在 `./output/msrun_log` 下，使用以下命令可查看训练状态（由于开启了流水并行，真实 loss 只显示在最后一个 stage 的日志中，其余卡显示 `loss` 为 `0`）：

```shell
tail -f ./output/msrun_log/worker_7.log   # 单机训练时使用 8 卡时查看 worker_7.log
```

训练过程中的权重 checkpoint 将会保存在 `./output/checkpoint` 下。

如有关于 DeepSeek-V4 预训练的相关问题，可以在 MindSpore Transformers 的 AtomGit 仓库中[提交 ISSUE](https://atomgit.com/mindspore/mindformers/issues/new) 以获取支持。

## 附录

### 模型文件说明

DeepSeek-V4 的模型文件结构如下（以 `mindformers` 为例）：

```text
📦mindformers
├── 📂mindformers
│   └── 📂models
│       └── 📂deepseek4
│           ├── 📄__init__.py
│           ├── 📄configuration_deepseek_v4.py
│           ├── 📄modeling_deepseek_v4.py
│           ├── 📄modeling_deepseek_v4_pynative.py
│           └── 📄utils.py
├── 📂configs
│   └── 📂deepseek4
│       ├── 📄pretrain_deepseek4_flash.yaml
│       ├── 📄pretrain_deepseek4_flash_reduction.yaml
│       ├── 📄pretrain_deepseek4_pro.yaml
│       └── 📄pretrain_deepseek4_pro_reduction.yaml
└── 📄run_mindformer.py
```

训练过程中的权重 checkpoint 将会保存在 `./output/checkpoint` 下。

如有关于 DeepSeek-V4 LoRA 微调的相关问题，可以在 MindSpore Transformers 的 AtomGit 仓库中[提交 ISSUE](https://atomgit.com/mindspore/mindformers/issues/new) 以获取支持。

**Q2：数据集准备部分中，应该如何将 `wiki.train.tokens` 转为 `jsonl` 格式数据？**

A2：[社区 issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY) 中提供了一个临时转换脚本，仅作为参考使用。用户需要根据自己的数据特点和需求，自行开发和验证适合的转换逻辑。

**Q3：如果修改了配置中的参数，使用 `run_mindformer.py` 拉起任务时，还需要重新传参吗？**

A3：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py` 会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

**Q4：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？**

A4：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从 50000~65536 中选取，避免端口号冲突的情况发生。

**Q5：我想看看我训练下来的权重效果怎么样，可以直接使用训练权重做推理吗？**

A5：当然可以！你可以通过以下两种方式进行推理：

1. **直接使用训练权重进行推理**，可以参考[《训练后模型进行评测》](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/evaluation.html#%E8%AE%AD%E7%BB%83%E5%90%8E%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%AF%84%E6%B5%8B)文档，使用去优化器合并的训练权重进行推理。
2. **反转训练权重为 Hugging Face 格式，复用 Hugging Face 生态进行推理**，可以参考类似的权重反转脚本（需针对 V4 适配）进行权重反转后，再进行推理任务。

更多 FAQ 请查看[官网 FAQ](https://www.mindspore.cn/mindformers/docs/zh-CN/master/faq/model_related.html)