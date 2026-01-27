<!--模型README模板-->

# [模型名称]

## 模型描述

[简要描述模型的发布者、优势、功能和用途。添加引用信息。]

## 支持规格

| 模型名称 | 规格 |   支持任务    |     模型架构     |                       支持设备                        |                       模型级别                        |
|:----:|:--:|:---------:|:------------:|:-------------------------------------------------:|:-------------------------------------------------:|
|  XX  | xB | 预训练/微调/推理 | Mcore/Legacy | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD | Released/Validated/Preliminary/Untested/Community |

说明：

- 模型架构：`Mcore` 表示 1.6.0 发布的新模型架构，`Legacy` 表示原有模型架构。详见[架构说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/overview.html)。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://atomgit.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

[标题里Mcore首字母大写，正文里出现使用小写]

## 版本配套

[模型名称] 当前推荐使用和首次支持的版本配套如下。当前推荐使用的版本一般是最新的经过验证的版本。如无特殊说明，首次支持的版本到推荐使用的版本之间的版本均可以使用。

|           | MindSpore Transformers | MindSpore | CANN | HDK |
|:---------:|:----------------------:|:---------:|:----:|:---:|
| 当前推荐使用的版本 |           xx           |    xx     |  xx  | xx  |
|  首次支持的版本  |           xx           |    xx     |  xx  | xx  |

## 使用样例

MindSpore Transformers 支持使用 [模型名称] 进行预训练、微调和推理。各任务的整体使用流程如下：

| 任务  | 前期准备                    | 使用流程                       |
|:---:|:------------------------|:---------------------------|
| 预训练 | 环境安装 -> 预训练数据集下载        | 数据预处理 -> 修改任务配置 -> 启动预训练任务 |
| 微调  | 环境安装 -> 模型下载 -> 微调数据集下载 | 数据预处理 -> 修改任务配置 -> 启动微调任务  |
| 推理  | 环境安装 -> 模型下载            | 修改任务配置 -> 启动推理任务           |

### 前期准备

#### 环境安装

按照上述版本配套，参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从Modelers、Hugging Face、ModelScope等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等（重头预训练不需加载权重）。链接如下：

|    模型名称    | 下载链接                                             | 说明 |
|:----------:|:-------------------------------------------------|:---|
|   XXX-xB   | [Modelers]() / [Hugging Face]() / [ModelScope]() |    |
|   XXX-xB   | [Modelers]() / [Hugging Face]() / [ModelScope]() |    |
| XXX-xB-AxB | [Modelers]() / [Hugging Face]() / [ModelScope]() |    |

#### 数据集下载

MindSpore Transformers 以下面的数据集为例提供了 [模型名称] 的预训练和微调流程的使用案例，实际训练时可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节制作数据集。请在执行任务前提前下载所需数据集。链接如下：

| 任务  |    数据集名称     | 下载链接         | 说明 |
|:---:|:------------:|:-------------|:---|
| 预训练 | WikiText-103 | [Download]() |    |
| 微调  |    Alpaca    | [Download]() |    |

### 预训练样例

[对预训练任务增加必要说明，包括使用场景、使用约束和限制]

#### 1. 数据预处理

[根据实际情况，清楚写出数据预处理的步骤和命令。]

#### 2. 修改任务配置

[给出修改的yaml文件名和链接，按修改含义分点归类写出修改的配置项和含义，对配置项进行必要说明]

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 3. 启动预训练任务

[给出启动预训练任务的命令，注意命令中需要包含必要的参数和配置文件路径。然后说明如何查看日志和结果。最后说明关于预训练的通用使用方法和相关特性介绍请参考官网预训练文档。]

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/{model_name}/pretrain_{model_name}.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 8 \
--run_mode train"
```

### 微调样例

[对微调任务增加必要说明，包括使用场景、使用约束和限制]

#### 1. 数据预处理

[根据实际情况，清楚写出数据预处理的步骤和命令。]

#### 2. 修改任务配置

[给出修改的yaml文件名和链接，按修改含义分点归类写出修改的配置项和含义，对配置项进行必要说明]

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 3. 启动微调任务

[给出启动微调任务的命令，注意命令中需要包含必要的参数和配置文件路径。然后说明如何查看日志和结果。最后说明关于微调的通用使用方法和相关特性介绍请参考官网微调文档。]

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/{model_name}/pretrain_{model_name}.yaml \
--pretrained_model_dir /path/to/pretrained_model
--auto_trans_ckpt True \
--use_parallel True \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 8 \
--run_mode finetune"
```

### 推理样例

[对推理任务增加必要说明，包括使用场景、使用约束和限制]

#### 1. 修改任务配置

[给出修改的yaml文件名和链接，按修改含义分点归类写出修改的配置项和含义，对配置项进行必要说明]

不同规格的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 启动推理任务

[给出启动推理任务的命令，注意命令中需要包含必要的参数和配置文件路径。然后说明如何查看日志和结果。最后说明关于推理的通用使用方法和相关特性介绍请参考官网推理文档。]

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/{model_name}/pretrain_{model_name}.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir /path/to/pretrained_model \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 8 \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 2
```

## 附录

### 模型文件说明

[模型名称]的模型文件包括以下内容：

```text
📦mindformers
├── 📂mindformers
│   ├── 📄xx.py
│   └── 📂xx         # [目录说明]
│       ├── 📄xx.py  # [文件说明]
│       └── 📄xx.py  # [文件说明]
├── 📂xx
│   └── 📄xx.yaml
└── 📄xx.xx
```

[符号建议：📦 根目录 / 📂 文件夹 / 📄 文件，用 ├── 和 └── 表示层级分支]

### 并行配置建议

以下配置为训练或推理场景下，不同模型规格的推荐配置。其中部分配置为经过验证的最佳配置，部分配置为可以运行的配置。用户可根据实际情况选择合适的配置。

> 注意：max_device_memory 在 Atlas 800T A2 上一般设置≤60GB，在 Atlas 800I A2 上一般设置≤30GB。

- 训练：

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
    <td>XX</td>
    <td>xxB</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 2
  model_parallel: 2
  pipeline_stage: 2
  micro_batch_num: 1
  vocab_emb_dp: True
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
</table>

- 推理：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>序列长度</th>
    <th>并行配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>XX</td>
    <td>xxB</td>
    <td>1 × Atlas 800T A2 (8P)</td>
    <td>8</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 8</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
</table>

### FAQ

Q1：[以用户的角度提出问题，注意问题要便于用户理解，有适当背景说明]

A1：[给出解答或解决方法，解决方法需要准确、清晰，便于用户理解和操作。无法直接解决的问题可以给出相关链接或issue联系方式。]

更多FAQ请查看[官网FAQ](https://www.mindspore.cn/mindformers/docs/zh-CN/master/faq/model_related.html)