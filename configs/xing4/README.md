# Xing4

## 模型描述

Xing4 是星辰语义大模型系列，提供 29B 和 285B 两款 MoE 模型。两款模型均采用 MLA 与逐层独立的 DSA indexer，29B 使用一层 MTP 和 mHC 主干；285B 的 MTP 层数为 0，主干使用标准残差连接。

模型共用 `Xing4ForCausalLM`、`Xing4ConfigConverter` 和权重转换模板。`Xing4V4Config` 对应 29B，`Xing4V41Config` 对应 285B；`v4`、`v41` 用于区分两种架构。架构参数使用 HF 字段名，执行选项通过框架参数配置。

| 型号 | 配置类 | `model_type` |
| :--- | :--- | :--- |
| 29B | `Xing4V4Config` | `xing4_v4` |
| 285B | `Xing4V41Config` | `xing4_v41` |

`Xing4Config.from_pretrained()` 按 `model_type` 恢复对应配置子类；直接构造时选择型号对应的配置类。

## 支持规格

| 模型名称 | 规格 | 任务 | 模型架构 | 运行模式 | 模型级别 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Xing4 29B | 29B | 预训练、继续训练 | Mcore | PyNative | Preliminary |
| Xing4 285B | 285B | 预训练、继续训练 | Mcore | PyNative | Preliminary |

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

从 [XingChen-AGI 模型主页](https://huggingface.co/XingChen-AGI)下载对应型号的配置、Tokenizer 和 BF16 权重文件。模型配置、Tokenizer 和权重应使用同一个型号及版本。

285B 的公开模型为 [Xing4.1-Preview](https://huggingface.co/XingChen-AGI/Xing4.1-Preview)，样例模型参数与其配置对齐。

加载 HF 权重时，目录中应包含 `config.json`、`model.safetensors.index.json` 和索引引用的全部 safetensors 分片。权重映射覆盖主干和 MTP，mHC 规则由模型配置控制。

#### 数据集准备

样例使用 `OrderedIndexDataLoader` 读取 `.bin` / `.idx` 索引数据。数据制作流程可参考[数据集指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)，Tokenizer 应使用对应 Xing4 型号的文件。

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
  --output-prefix /path/to/xing4_text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/Xing4-29B \
  --append-eod
```

默认 `text` 字段生成的文件为 `xing4_text_text_document.bin` 和 `xing4_text_text_document.idx`。配置中填写不带后缀的文件前缀；也可以填写包含这些文件的目录，加载器会按文件名顺序读取有效的文件对。

`OrderedIndexDataLoader` 按索引读取样本，并通过移位生成 `input_ids` 和 `labels`。制作数据时应根据训练策略组织文档和样本长度；对于长度为 `sequence_length` 的训练输入，完整样本需要 `sequence_length + 1` 个 token。短样本会补齐，长样本会截断。29B 样例使用 EOD token `2`、padding token `3`；285B 样例使用 EOD token `1`、padding token `2`，应与对应型号的实际数据处理约定一致。

#### 2. 修改任务配置

配置文件为 [pretrain_xing4_29b.yaml](./pretrain_xing4_29b.yaml) 和 [pretrain_xing4_285b.yaml](./pretrain_xing4_285b.yaml)，均使用 4K 序列进行预训练。长上下文在中训练、后训练阶段按型号配置扩展，285B 公开配置的最大上下文声明为 1M。以下数据处理示例使用 29B；285B 使用自身的 Tokenizer 和数据路径。启动前修改数据路径：

```yaml
train_dataset:
  dataloader:
    type: OrderedIndexDataLoader
    data_path: "/path/to/xing4_text_text_document"
    sequence_length: 4096
    eod_token: 2
    pad_token: 3
```

主要配置如下：

| 配置项 | 样例值 | 说明 |
| :--- | :--- | :--- |
| `model.seq_length` / `train_dataset.dataloader.sequence_length` | `4096` | 4K 预训练，模型和数据集长度保持一致 |
| `model.position_embedding_type` / `model.rope_scaling.factor` | `yarn`；29B：`8.0`，285B：`32.0` | 原始上下文 `32768`，实际训练序列长度 `4096` |
| `model.num_nextn_predict_layers` | 29B：`1`；285B：`0` | 与对应 checkpoint 的预测头结构对齐 |
| `model.experimental_attention_variant` | `dsa` | 逐层独立 DSA indexer |
| `model.index_n_heads` | 29B：`16`；285B：`64` | Indexer 头数 |
| `model.dsa_indexer_topk_freq` | `1` | 每层独立计算 indexer |
| `model.enable_hyper_connections` | 29B：`true`；285B：`false` | 主干残差结构 |
| `model.mtp_enable_hyper_connections` / `model.mtp_enable_dsa` | `false` / `false` | MTP 使用普通 MLA 与标准残差 |
| `model.moe_router_score_function` | `sigmoid` | 生效的路由评分参数 |
| `optimizer.type` | `Muon` | 训练优化器 |
| `training.local_batch_size` / `training.global_batch_size` | `1` / `64` | 按实际并行度设置批量大小 |
| `parallelism.tensor_parallel` / `context_parallel` | `1` / `4` | Sparse DSA 上下文并行使用 Colossal |
| `parallelism.pipeline_parallel` / `expert_parallel` | `1` / `1` | 按设备资源调整 |
| `parallelism.data_parallel_shard` | `-1` | 根据并行配置推导分片大小 |

根据设备资源、序列长度和重计算策略调整并行配置。设备总数应与 TP、CP、PP 和 DP 的乘积一致，全局 batch size 应与 DP 和梯度累积步数匹配。

继续训练时，将 `checkpoint.load_path` 设置为本地 BF16 HF 权重目录：

```yaml
checkpoint:
  load_path: "/path/to/Xing4-29B"
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
--config configs/xing4/pretrain_xing4_29b.yaml" \
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
├── mindformers/models/xing4
│   ├── __init__.py
│   ├── configuration_xing4.py
│   ├── config_converter_xing4.py
│   ├── modeling_xing4.py
│   └── utils.py
├── configs/xing4
│   ├── README.md
│   ├── pretrain_xing4_29b.yaml
│   └── pretrain_xing4_285b.yaml
└── run_mindformer.py
```

### FAQ

**Q：调整训练序列长度时需要修改哪些配置？**

同步修改 `model.seq_length` 和 `train_dataset.dataloader.sequence_length`，并按实际数据调整压缩 EOD 的容量。样例采用 4K 训练序列，YaRN 参数为 `rope_theta=100000`、原始上下文 `32768`，29B 倍率为 `8`，285B 倍率为 `32`。实际训练序列长度与位置编码缩放倍率分别设置。

**Q：如何设置 mHC 相关开关？**

29B 主干使用 `enable_hyper_connections: true`，285B 主干使用 `false`。29B 的 MTP 使用 `mtp_enable_hyper_connections: false`；285B 默认 MTP 层数为 0。

**Q：在哪里反馈模型使用问题？**

可以在 MindSpore Transformers 仓库[提交 Issue](https://atomgit.com/mindspore/mindformers/issues/new)，附上运行环境、并行配置及相关错误日志。

### 命名迁移

| 原入口 | 当前入口 |
| :--- | :--- |
| `TeleChatV4Config` | `Xing4V4Config`（29B） |
| `TeleChatV4ForCausalLM` | `Xing4ForCausalLM` |
| `mindformers.models.telechat4_moe` | `mindformers.models.xing4` |
| `configs/telechat4_moe/pretrain_telechat4_29b.yaml` | `configs/xing4/pretrain_xing4_29b.yaml` |

`Xing4Config.from_pretrained()` 接受 MF 型号 `xing4_v4`、`xing4_v41`，以及对应 HF 身份 `xing4_0`、`deepseek_v32`。原生 `xing4_0` 配置中的 indexer 字段用于区分普通 MLA 和 DSA；显式注意力设置优先生效。缺失的模板权重默认警告并保留初始化值，程序化调用 `load_hf_checkpoint(..., strict_load=True)` 可启用严格校验。
