# Xing4

## 模型描述

Xing4 是由中国电信人工智能科技（北京）有限公司研发的星辰语义大模型系列。

Xing4.1-Preview 是 Xing 系列的最新模型，面向长上下文交互、复杂推理与智能体工作流。模型总参数量为 285B，每个 token 的激活参数量为 17B，原生支持 1M 上下文，面向智能体编程和长周期复杂智能体任务。

模型训练采用 8,192 颗昇腾 910C 芯片与 MindSpore 训练框架。在数据分析、前端生成和长文档理解等智能体任务中，模型表现出有竞争力的能力，并已使用 Codex、OpenCode、Pi、Claude Code 和 WorkBuddy 等多种主流智能体框架进行评测。

主要应用场景包括：

- **智能体协同办公**：多步骤办公流程、跨工具编排，以及结合多源材料交叉验证的结构化报告生成。
- **深度研究**：持续开展多源调研，输出结构化且可验证的成果。
- **智能体编程**：完整前端生成（单文件、零依赖、离线可用的 Web 应用）、3D 创意编程、数据分析与报告。
- **文档密集型后台业务**：按指定模式提取结构化信息，支持智能助手、请求路由和文档提取。

上述介绍译自 [Xing4.1-Preview 官方 HuggingFace 模型卡](https://huggingface.co/XingChen-AGI/Xing4.1-Preview/blob/dc8f3047517c3b79be40e1e367f2b50a05b68ec3/README.md)。

## 支持规格

| 模型名称 | 规格 | 任务 | 模型架构 | 运行模式 | 模型级别 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Xing4 285B | 285B | 预训练、微调 | Mcore | PyNative | Preliminary |

说明：

- `Mcore` 表示新模型架构，运行环境使用 Ascend 设备。
- 模型级别说明见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 使用样例

| 任务 | 前期准备 | 使用流程 |
| :---: | :--- | :--- |
| 预训练 | 环境安装、Tokenizer 和数据集准备 | 数据预处理 -> 修改任务配置 -> 启动训练 |
| 微调 | 环境安装、BF16 模型文件和指令数据集准备 | 制作 HF 数据集 -> 修改微调配置 -> 启动训练 |

### 前期准备

#### 环境安装

参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装 MindSpore Transformers 运行环境，并配置匹配的 HyperParallel、CANN 和 Ascend 驱动。

#### 模型下载

从 [XingChen-AGI 模型主页](https://huggingface.co/XingChen-AGI)下载对应型号的配置、Tokenizer 和 BF16 权重文件。模型配置、Tokenizer 和权重应使用同一个型号及版本。

285B 的公开模型为 [Xing4.1-Preview-MindSpore](https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore)，样例模型参数与其配置对齐。

加载 HF 权重时，目录中应包含 `config.json`、`model.safetensors.index.json` 和索引引用的全部 safetensors 分片。

#### 数据集准备

预训练样例使用 `OrderedIndexDataLoader` 读取 `.bin` / `.idx` 索引数据。数据制作流程可参考[数据集指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)，Tokenizer 应使用对应 Xing4 型号的文件。

### 预训练样例

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
  --output-prefix /path/to/xing4_text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/Xing4.1-Preview-MindSpore \
  --append-eod
```

默认 `text` 字段生成的文件为 `xing4_text_text_document.bin` 和 `xing4_text_text_document.idx`。配置中填写不带后缀的文件前缀；也可以填写包含这些文件的目录，加载器会按文件名顺序读取有效的文件对。

`OrderedIndexDataLoader` 按索引读取样本，并通过移位生成 `input_ids` 和 `labels`。制作数据时应根据训练策略组织文档和样本长度；对于长度为 `sequence_length` 的训练输入，完整样本需要 `sequence_length + 1` 个 token。短样本会补齐，长样本会截断。285B 样例使用 EOD token `1`、padding token `2`，应与实际数据处理约定一致。

#### 2. 修改任务配置

配置文件为 [pretrain_xing4_285b.yaml](./pretrain_xing4_285b.yaml)，使用 4K 序列进行预训练。长上下文在中训练、后训练阶段扩展，285B 公开配置的最大上下文声明为 1M。启动前修改数据路径：

```yaml
train_dataset:
  dataloader:
    type: OrderedIndexDataLoader
    data_path: "/path/to/xing4_text_text_document"
    sequence_length: 4096
    eod_token: 1
    pad_token: 2
```

主要配置如下：

| 配置项 | 样例值 | 说明 |
| :--- | :--- | :--- |
| `model.seq_length` / `train_dataset.dataloader.sequence_length` | `4096` | 4K 预训练，模型和数据集长度保持一致 |
| `model.position_embedding_type` / `model.rope_scaling.factor` | `yarn` / `32.0` | 原始上下文 `32768`，实际训练序列长度 `4096` |
| `optimizer.type` | `Muon` | 训练优化器 |
| `training.local_batch_size` / `training.global_batch_size` | `1` / `64` | 按实际并行度设置批量大小 |
| `parallelism.tensor_parallel` / `context_parallel` | `1` / `4` | 上下文并行使用 Colossal |
| `parallelism.pipeline_parallel` / `expert_parallel` | `1` / `1` | 按设备资源调整 |
| `parallelism.data_parallel_shard` | `-1` | 根据并行配置推导分片大小 |

根据设备资源、序列长度和重计算策略调整并行配置。设备总数应与 TP、CP、PP 和 DP 的乘积一致，全局 batch size 应与 DP 和梯度累积步数匹配。

继续训练时，将 `checkpoint.load_path` 设置为本地 BF16 HF 权重目录：

```yaml
checkpoint:
  load_path: "/path/to/Xing4.1-Preview-MindSpore"
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
--config configs/xing4/pretrain_xing4_285b.yaml" \
${worker_num} ${local_worker} ${master_ip} ${port} ${node_rank} \
output/msrun_log False 7200
```

训练日志保存在 `./output/msrun_log` 下。样例默认 PP 为 `1`，可查看 `worker_0.log`；调整流水线并行后，应查看最后一个流水线 stage 对应的日志。

```shell
tail -f output/msrun_log/worker_0.log
```

训练 checkpoint 默认保存在 `./output/checkpoint` 下。

### 微调样例

#### 1. 制作 HF 格式数据集

使用模型提供的[数据处理脚本](../../toolkit/data_preprocess/xing4/pack_sft_data.py)，将 `system/dialog` 格式的对话 JSONL 处理为定长 packed Arrow 数据集。脚本按输入文件并行 tokenize、按配比采样并分片，再进行贪心装桶和标签生成；输出可由 `HFDataLoader` 的 `load_from_disk` 直接读取。

在独立环境中安装数据处理依赖。285B 公开 Tokenizer 使用 `TokenizersBackend`，需要 `transformers>=5.0`：

```shell
python -m venv output/xing4_sft_env
source output/xing4_sft_env/bin/activate
python -m pip install "transformers>=5.0" datasets pyarrow pandas openpyxl
```

每行一条对话，例如：

```jsonl
{"system": "你是一个助手。", "dialog": [{"role": "user", "content": "什么是监督微调？"}, {"role": "bot", "content": "</think>监督微调使用指令与回答数据训练模型。", "select": true}]}
```

`role` 支持 `user`、`bot`（也可写为 `assistant`）和 `tool`，最后一轮必须为 `bot` 或 `assistant`。脚本使用对应 Tokenizer 中的 `<_system>`、`<_user>`、`<_bot>`、`<_observation>`、`<_end>` 和 `<_pad>` 组织对话。`select` 省略时默认为 `true`，设为 `false` 时该回答不计入 loss。

回答需以 `<think>` 或 `</think>` 开头。默认混合思考模式不监督该首标记；`--no-hybrid` 让首标记也参与 loss，输入仍使用相同的思考数据格式。两种模式都会检查回答首标记，不符合格式的回答不计 loss。用户轮、工具轮和系统提示词不计 loss。

准备 Excel 配比表 `weights.xlsx`，在 `Sheet1` 中填写以下两列：

| 文件名 | 权重 |
| :--- | :--- |
| `/path/to/sft/general.jsonl` | `1` |
| `/path/to/sft/code.jsonl` | `0.5` |

权重小于 `1` 时按概率采样；大于等于 `1` 时先按整数部分复制，小数部分按概率追加一份。文件名为相对路径时，通过 `--data-root` 指定其根目录。

以下以 285B 的 4K 微调为例，在仓库根目录执行：

```shell
python toolkit/data_preprocess/xing4/pack_sft_data.py \
  --weights /path/to/weights.xlsx \
  --tokenizer /path/to/Xing4.1-Preview-MindSpore \
  --output /path/to/xing4_285b_sft \
  --max-len 4096 \
  --shards 64 \
  --tokenize-workers 32 \
  --pack-workers 8

python toolkit/data_preprocess/xing4/pack_sft_data.py \
  --validate /path/to/xing4_285b_sft/summary \
  --tokenizer /path/to/Xing4.1-Preview-MindSpore \
  --max-len 4096
```

数据处理和校验完成后执行 `deactivate`，返回已配置好的 MindFormers 训练环境。

`--max-len` 必须与训练配置的 `model.seq_length` 一致。增大 `--shards` 可减少单个装桶进程处理的数据量，`--pack-workers` 根据可用内存设置。样例保留默认的标签左移，不添加 `--no-label-shift`；`--validate` 检查 Arrow 分片、序列长度、样本边界和标签对齐。

输出目录为：

```text
xing4_285b_sft/summary/
├── dataset_dict.json
└── train/
    ├── dataset_info.json
    ├── state.json
    └── shard-*.arrow
```

将 `summary` 目录作为训练的 `dataset_path`。数据包含 `input_ids`、`labels`、`loss_mask`、`position_ids` 和 `actual_seq_len`。`labels` 已左移一位，不监督的位置为 `-100`，加载时不再移位；`actual_seq_len` 记录桶内各样本的结束位置，使不同样本分别计算注意力。训练侧的 `HFDataLoader` 根据标签和边界生成有效监督掩码、各段的位置编号及定长压缩边界。`compressed_eod_mask_length` 按桶内样本数量设置，样例为 `128`。

特殊 token ID 由 Tokenizer 读取，不在脚本中硬编码。

#### 2. 修改微调配置

配置文件为 [finetune_xing4_285b.yaml](./finetune_xing4_285b.yaml)。微调样例保留公开版本的模型参数，使用 4K 序列进行全参数微调。修改权重路径和 HF 数据集路径：

```yaml
checkpoint:
  load_path: "/path/to/Xing4.1-Preview-MindSpore"
  no_load_optim: true
  save_path: "./output/finetune_xing4_285b/checkpoint"
train_dataset:
  dataloader:
    type: HFDataLoader
    load_func: load_from_disk
    dataset_path: "/path/to/xing4_285b_sft/summary"
    create_attention_mask: true
    create_compressed_eod_mask: true
    compressed_eod_mask_length: 128
    use_broadcast_data: true
    shuffle: true
```

`checkpoint.load_path` 支持 HF 权重目录和原生 MF checkpoint 目录。HF 权重应包含索引及其引用分片；原生 MF checkpoint 应保留 `metadata.json` 及其引用分片。权重转换方法见[转换工具说明](../../toolkit/weight_convert/xing4/README.md)。

样例训练 `1000` 步、学习率 `1.0e-5`、预热 `100` 步，并设置 `model.calculate_per_token_loss: true`，按有效监督 token 计算 loss。根据实际任务调整步数、学习率、全局 batch size 和并行配置。调整序列长度时，重新制作相同长度的数据并同步修改 `model.seq_length`。

#### 3. 启动微调任务

沿用预训练样例中的 `worker_num`、`local_worker`、`master_ip`、`port` 和 `node_rank` 设置，在每台服务器执行：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--mode 1 \
--config configs/xing4/finetune_xing4_285b.yaml" \
${worker_num} ${local_worker} ${master_ip} ${port} ${node_rank} \
output/finetune_msrun_log False 7200
```

训练日志保存在 `./output/finetune_msrun_log`，checkpoint 保存在对应配置的 `checkpoint.save_path` 下。

## 附录

### 模型文件说明

```text
mindformers
├── mindformers/models/xing4
│   ├── __init__.py
│   ├── configuration_xing4.py
│   ├── config_converter_xing4.py
│   ├── modeling_xing4.py
│   └── utils.py
├── configs/xing4
│   ├── README.md
│   ├── pretrain_xing4_285b.yaml
│   └── finetune_xing4_285b.yaml
├── toolkit/data_preprocess/xing4
│   └── pack_sft_data.py
└── run_mindformer.py
```

### FAQ

**Q：调整训练序列长度时需要修改哪些配置？**

同步修改 `model.seq_length` 和 `train_dataset.dataloader.sequence_length`，并按实际数据调整压缩 EOD 的容量。样例采用 4K 训练序列，YaRN 参数为 `rope_theta=100000`、原始上下文 `32768`，285B 倍率为 `32`。实际训练序列长度与位置编码缩放倍率分别设置。

**Q：在哪里反馈模型使用问题？**

可以在 MindSpore Transformers 仓库[提交 Issue](https://atomgit.com/mindspore/mindformers/issues/new)，附上运行环境、并行配置及相关错误日志。

### 命名迁移

| 原入口 | 当前入口 |
| :--- | :--- |
| `TeleChatV4ForCausalLM` | `Xing4ForCausalLM` |
| `mindformers.models.telechat4_moe` | `mindformers.models.xing4` |

285B 配置可通过 `Xing4Config.from_pretrained()` 读取，`model_type` 使用 `xing4_v41` 或公开配置身份 `Xing4.1-Preview`。缺失权重默认警告并保留初始化值，程序化调用 `load_hf_checkpoint(..., strict_load=True)` 可启用严格校验。
