# TeleChat4 MoE

## 模型描述

**Xing4.0** 由中电信人工智能科技有限公司研发，是星辰语义大模型系列（原 [TeleChat](https://github.com/Tele-AI/TeleChat3)）的新一代模型，**完全基于国产算力**训练。

**Xing4.0-29B-A4B** 模型总参数量 29B，激活参数仅 4B，原生支持 256K 上下文，可扩展至 512K，是国内首个基于国产算力与国产框架完成训练、面向复杂工程任务深度优化的百亿参数大模型。主要特性如下：

- **面向 Agent 的架构设计**：采用 mHC + MLA + MTP 架构，支持多步骤规划、工具调用与复杂推理链路执行，保障长上下文下任务连贯与执行稳定。
- **国产算力深度协同**：面向昇腾 910C 集群，基于 MindSpore/MindFormers 完成 mHC 等新特性适配、跨框架精度对齐及融合算子开发，实现国产算力平台上的稳定高效训练。
- **训练效率大幅提升**：通过细粒度 MoE 通信优化、选择性重计算、DVM 图算自动融合、Ascend C mHC 融合算子等多层次协同优化，整体训练吞吐较开箱性能**提升约 96%**。
- **开源生态全面兼容**：训练微调支持 ms-swift、LLaMA-Factory、MindFormers（ms-swift [#10275](https://github.com/modelscope/ms-swift/pull/10275)、LLaMA-Factory [#10754](https://github.com/hiyouga/LlamaFactory/pull/10754) 已合入上游主线）；推理部署支持 SGLang（原生支持已合入主线 [#39793](https://github.com/sgl-project/sglang/pull/39793)）、vLLM（官方插件）；本地 GGUF 推理支持 llama.cpp（原生支持已合入主线 [#31589](https://github.com/ggml-org/llama.cpp/pull/31589)，b12174+）；针对 OpenCode、Claude Code、OpenClaw、Hermes 等 Agent 框架进行定向适配与格式对齐，可直接接入现有工作流。算力方面，基于智源 FlagOS 完成多款 AI 芯片平台适配，支持跨架构一键部署。
- **低资源场景易于适配**：模型具备良好的下游任务微调能力，可面向私域数据进行轻量化定制，适用于意图识别、表格理解、合同审计、知识问答等垂直场景，以较低成本实现领域能力的快速构建与落地。

## 支持规格

| 模型名称 | 规格 | 任务 | 模型架构 | 运行模式 | 模型级别 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| TeleChat4 / Xing4.0-29B-A4B | 29B（激活约 4B） | 预训练、继续训练 | Mcore | PyNative | Preliminary |

说明：

- `Mcore` 表示新模型架构，运行环境使用 Ascend 设备。
- 模型级别说明见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 使用样例

| 任务 | 前期准备 | 使用流程 |
| :---: | :--- | :--- |
| 预训练 | 环境安装、Tokenizer 和数据集准备 | 数据预处理 -> 修改任务配置 -> 启动训练 |
| 继续训练 | 环境安装、BF16 模型文件和数据集准备 | 数据预处理 -> 配置权重加载路径 -> 启动训练 |

### 前期准备

#### 环境安装

参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装 MindSpore Transformers 运行环境，并配置匹配的 HyperParallel、CANN 和 Ascend 驱动。

#### 模型下载

从开源模型仓库下载配置、Tokenizer 和 BF16 权重文件。从随机初始化开始预训练时，只需准备数据预处理使用的 Tokenizer。

| 模型名称 | 下载链接 |
| :--- | :--- |
| XingChen-AGI/Xing4.0-29B-A4B | [Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) |

本文使用的配置及权重结构参考 revision `acbdebfe496d117e6fde3ce152d2450fb0d5fa53`。加载 HF 权重时，目录中应包含 `config.json`、`model.safetensors.index.json` 和索引引用的全部 safetensors 分片。

#### 数据集准备

样例使用 `OrderedIndexDataLoader` 读取 `.bin` / `.idx` 索引数据。数据制作流程可参考[数据集指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)，Tokenizer 应使用上述 Xing4.0 模型的文件。

### 训练样例

#### 1. 数据预处理

将原始文本整理为包含 `text` 字段的 JSONL 文件，例如：

```jsonl
{"text": "第一条训练文本。"}
{"text": "第二条训练文本。"}
```

使用仓库提供的脚本将文本转成索引数据：

```shell
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/telechat4_text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/Xing4.0-29B-A4B \
  --append-eod
```

默认 `text` 字段生成的文件为 `telechat4_text_text_document.bin` 和 `telechat4_text_text_document.idx`。配置中填写不带后缀的文件前缀；也可以填写包含这些文件的目录，加载器会按文件名顺序读取有效的文件对。

`OrderedIndexDataLoader` 按索引读取样本，并通过移位生成 `input_ids` 和 `labels`。制作数据时应根据训练策略组织文档和样本长度；对于长度为 `sequence_length` 的训练输入，完整样本需要 `sequence_length + 1` 个 token。短样本会补齐，长样本会截断。样例使用 EOD token `2`、padding token `3`，应与实际数据处理约定一致。

#### 2. 修改任务配置

配置文件为 [pretrain_telechat4_29b.yaml](./pretrain_telechat4_29b.yaml)，使用 4K 序列进行预训练。256K 是中训练、后训练阶段扩展的上下文长度。启动前修改数据路径：

```yaml
train_dataset:
  dataloader:
    type: OrderedIndexDataLoader
    data_path: "/path/to/telechat4_text_text_document"
    sequence_length: 4096
    eod_token: 2
    pad_token: 3
```

主要配置如下：

| 配置项 | 样例值 | 说明 |
| :--- | :--- | :--- |
| `model.seq_length` / `train_dataset.dataloader.sequence_length` | `4096` | 4K 预训练，模型和数据集长度保持一致 |
| `model.position_embedding_type` / `model.rope_scaling.factor` | `yarn` / `64.0` | 位置编码参数与公开模型配置一致，实际训练序列长度为 `4096` |
| `model.experimental_attention_variant` | `null` | 使用普通 MLA |
| `model.enable_hyper_connections` | `true` | 主干启用 mHC |
| `model.mtp_enable_hyper_connections` | `false` | 与公开 MTP 权重结构一致 |
| `model.moe_router_score_function` | `sigmoid` | 生效的路由评分参数 |
| `optimizer.type` | `Muon` | 训练优化器 |
| `training.local_batch_size` / `training.global_batch_size` | `1` / `64` | 按实际并行度设置批量大小 |
| `parallelism.tensor_parallel` / `context_parallel` | `1` / `4` | 上下文并行使用 Ulysses |
| `parallelism.pipeline_parallel` / `expert_parallel` | `1` / `1` | 按设备资源调整 |
| `parallelism.data_parallel_shard` | `-1` | 根据并行配置推导分片大小 |

根据设备资源、序列长度和重计算策略调整并行配置。设备总数应与 TP、CP、PP 和 DP 的乘积一致，全局 batch size 应与 DP 和梯度累积步数匹配。

继续训练时，将 `checkpoint.load_path` 设置为本地 BF16 HF 权重目录：

```yaml
checkpoint:
  load_path: "/path/to/Xing4.0-29B-A4B"
  no_load_optim: true
  save_path: "./output/checkpoint"
```

从随机初始化开始训练时，保持 `checkpoint.load_path: ""`。训练步数、学习率和全局 batch size 根据实际任务调整；样例默认训练 `4000` 步，学习率为 `3.0e-5`。

#### 3. 启动训练任务

在仓库根目录通过 `scripts/msrun_launcher.sh` 启动 `run_mindformer.py`，并使用 `--mode 1` 选择 PyNative。将下列 `YOUR_*` 占位符替换为与并行配置和可用资源对应的值；多机任务需在每台服务器执行，并分别设置 `node_rank`。

```shell
worker_num=YOUR_TOTAL_DEVICE_COUNT
local_worker=YOUR_LOCAL_DEVICE_COUNT
master_ip=YOUR_MASTER_IP
node_rank=YOUR_NODE_RANK
port=8118

bash scripts/msrun_launcher.sh "run_mindformer.py \
--mode 1 \
--config configs/telechat4_moe/pretrain_telechat4_29b.yaml" \
${worker_num} ${local_worker} ${master_ip} ${port} ${node_rank} \
output/msrun_log False 7200
```

训练日志保存在 `./output/msrun_log` 下。样例默认 PP 为 `1`，可查看 `worker_0.log`；调整流水线并行后，应查看最后一个流水线 stage 对应的日志。

```shell
tail -f output/msrun_log/worker_0.log
```

训练 checkpoint 默认保存在 `./output/checkpoint` 下。

## 附录

### 模型文件说明

```text
mindformers
├── mindformers/models/telechat4_moe
│   ├── __init__.py
│   ├── configuration_telechat_v4.py
│   ├── config_converter_telechat_v4.py
│   ├── modeling_telechat_v4.py
│   └── utils.py
├── configs/telechat4_moe
│   ├── README.md
│   └── pretrain_telechat4_29b.yaml
└── run_mindformer.py
```

### FAQ

**Q：调整训练序列长度时需要修改哪些配置？**

同步修改 `model.seq_length` 和 `train_dataset.dataloader.sequence_length`，并按实际数据调整压缩 EOD 的容量。样例采用 4K 训练序列，并保留公开模型的 YaRN 配置。实际训练序列长度与位置编码缩放倍率分别设置。

**Q：如何设置 mHC 相关开关？**

样例按照公开 29B 模型结构配置，主干使用 `enable_hyper_connections: true`，MTP 使用 `mtp_enable_hyper_connections: false`。

**Q：在哪里反馈模型使用问题？**

可以在 MindSpore Transformers 仓库[提交 Issue](https://atomgit.com/mindspore/mindformers/issues/new)，附上运行环境、并行配置及相关错误日志。
