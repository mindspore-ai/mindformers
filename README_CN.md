# 欢迎来到MindSpore Transformers（MindFormers）

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 一、介绍

MindSpore Transformers套件的目标是构建一个大模型预训练、微调、评测、推理、部署的全流程开发套件，提供业内主流的Transformer类大语言模型、多模态理解和全模态模型。期望帮助用户轻松地实现大模型全流程开发。

MindSpore Transformers套件基于MindSpore内置的多维混合并行技术和组件化设计，具备如下特点：

- 配置化一键启动大模型预训练、微调、评测、推理、部署流程。
- 对接Hugging Face、Megatron-LM、vLLM、OpenCompass等主流生态。
- 提供丰富的多维混合并行和调试调优能力，支持万亿规格模型训练。
- 大模型训推系统级深度优化，提升千亿稠密、万亿稀疏大模型训推性能。
- 训练高可用，保障大模型在万卡集群稳定运行。
- 提供细粒度多层级的训练监控能力，帮助训练异常定位分析。
- 通过Mcore架构升级和模块化设计，简化模型集成方式，带来更广泛的标准化和更强的生态支持。

欲获取MindSpore Transformers相关使用教程以及API文档，请参阅[**MindSpore Transformers文档**](https://www.mindspore.cn/mindformers/docs/zh-CN/master/index.html)，以下提供部分内容的快速跳转链接：

- 📝 [大模型预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html)
- 📝 [大模型微调](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/supervised_fine_tuning.html)
- 📝 [大模型评测](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/evaluation.html)
- 📝 [服务化部署](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/deployment.html)

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

如果对MindSpore Transformers的技术感兴趣，或者想参与贡献代码，欢迎加入[MindSpore Transformers SIG](https://www.mindspore.cn/sig/MindSpore%20Transformers)。

### 模型列表

当前MindSpore Transformers全量的模型列表如下：

| 模型名                                                                                                                                             | 支持规格                          |   模型类型   |     模型架构     |   最新支持版本   |
|:------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------|:--------:|:------------:|:----------:|
| [Qwen3](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3) ![Recent Popular](./docs/assets/hot.svg)                           | 0.6B/1.7B/4B/8B/14B/32B       |  稠密LLM   |    Mcore     | 1.8.0、在研版本 |
| [Qwen3-MoE](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3_moe) ![Recent Popular](./docs/assets/hot.svg)                   | 30B-A3B/235B-A22B             |  稀疏LLM   |    Mcore     | 1.8.0、在研版本 |
| [DeepSeek-V3](https://atomgit.com/mindspore/mindformers/blob/master/research/deepseek3) ![Recent Popular](./docs/assets/hot.svg)                | 671B                          |  稀疏LLM   | Mcore/Legacy | 1.8.0、在研版本 |
| [GLM4.5](https://atomgit.com/mindspore/mindformers/blob/master/configs/glm4_moe) ![Recent Popular](./docs/assets/hot.svg)                       | 106B-A12B/355B-A32B           |  稀疏LLM   |    Mcore     | 1.8.0、在研版本 |
| [GLM4](https://atomgit.com/mindspore/mindformers/blob/master/configs/glm4) ![Recent Popular](./docs/assets/hot.svg)                             | 9B                            |  稠密LLM   | Mcore/Legacy | 1.8.0、在研版本 |
| [Qwen2.5](https://atomgit.com/mindspore/mindformers/blob/master/research/qwen2_5) ![Recent Popular](./docs/assets/hot.svg)                      | 0.5B/1.5B/7B/14B/32B/72B      |  稠密LLM   |    Legacy    | 1.8.0、在研版本 |
| [TeleChat2](https://atomgit.com/mindspore/mindformers/blob/master/research/telechat2) ![Recent Popular](./docs/assets/hot.svg)                  | 7B/35B/115B                   |  稠密LLM   | Mcore/Legacy | 1.8.0、在研版本 |
| [Llama3.1](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/llama3_1) ![End of Life](./docs/assets/eol.svg)                       | 8B/70B                        |  稠密LLM   |    Legacy    |   1.7.0    |
| [Mixtral](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/mixtral) ![End of Life](./docs/assets/eol.svg)                         | 8x7B                          |  稀疏LLM   |    Legacy    |   1.7.0    |
| [CodeLlama](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md) ![End of Life](./docs/assets/eol.svg)          | 34B                           |  稠密LLM   |    Legacy    |   1.5.0    |
| [CogVLM2-Image](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md) ![End of Life](./docs/assets/eol.svg)  | 19B                           |    MM    |    Legacy    |   1.5.0    |
| [CogVLM2-Video](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md) ![End of Life](./docs/assets/eol.svg)  | 13B                           |    MM    |    Legacy    |   1.5.0    |
| [DeepSeek-V2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2) ![End of Life](./docs/assets/eol.svg)                   | 236B                          |  稀疏LLM   |    Legacy    |   1.5.0    |
| [DeepSeek-Coder-V1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5) ![End of Life](./docs/assets/eol.svg)         | 7B                            |  稠密LLM   |    Legacy    |   1.5.0    |
| [DeepSeek-Coder](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek) ![End of Life](./docs/assets/eol.svg)                 | 33B                           |  稠密LLM   |    Legacy    |   1.5.0    |
| [GLM3-32K](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/glm32k) ![End of Life](./docs/assets/eol.svg)                         | 6B                            |  稠密LLM   |    Legacy    |   1.5.0    |
| [GLM3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md) ![End of Life](./docs/assets/eol.svg)                    | 6B                            |  稠密LLM   |    Legacy    |   1.5.0    |
| [InternLM2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/internlm2) ![End of Life](./docs/assets/eol.svg)                     | 7B/20B                        |  稠密LLM   |    Legacy    |   1.5.0    |
| [Llama3.2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) ![End of Life](./docs/assets/eol.svg)            | 3B                            |  稠密LLM   |    Legacy    |   1.5.0    |
| [Llama3.2-Vision](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md) ![End of Life](./docs/assets/eol.svg)       | 11B                           |    MM    |    Legacy    |   1.5.0    |
| [Llama3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/llama3) ![End of Life](./docs/assets/eol.svg)                           | 8B/70B                        |  稠密LLM   |    Legacy    |   1.5.0    |
| [Llama2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md) ![End of Life](./docs/assets/eol.svg)                | 7B/13B/70B                    |  稠密LLM   |    Legacy    |   1.5.0    |
| [Qwen2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen2) ![End of Life](./docs/assets/eol.svg)                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | 稠密/稀疏LLM |    Legacy    |   1.5.0    |
| [Qwen1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5) ![End of Life](./docs/assets/eol.svg)                         | 7B/14B/72B                    |  稠密LLM   |    Legacy    |   1.5.0    |
| [Qwen-VL](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl) ![End of Life](./docs/assets/eol.svg)                          | 9.6B                          |    MM    |    Legacy    |   1.5.0    |
| [TeleChat](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/telechat) ![End of Life](./docs/assets/eol.svg)                       | 7B/12B/52B                    |  稠密LLM   |    Legacy    |   1.5.0    |
| [Whisper](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md) ![End of Life](./docs/assets/eol.svg)              | 1.5B                          |    MM    |    Legacy    |   1.5.0    |
| [Yi](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yi) ![End of Life](./docs/assets/eol.svg)                                   | 6B/34B                        |  稠密LLM   |    Legacy    |   1.5.0    |
| [YiZhao](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yizhao) ![End of Life](./docs/assets/eol.svg)                           | 12B                           |  稠密LLM   |    Legacy    |   1.5.0    |
| [Baichuan2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md) ![End of Life](./docs/assets/eol.svg)        | 7B/13B                        |  稠密LLM   |    Legacy    |   1.3.2    |
| [GLM2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md) ![End of Life](./docs/assets/eol.svg)                    | 6B                            |  稠密LLM   |    Legacy    |   1.3.2    |
| [GPT2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md) ![End of Life](./docs/assets/eol.svg)                    | 124M/13B                      |  稠密LLM   |    Legacy    |   1.3.2    |
| [InternLM](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md) ![End of Life](./docs/assets/eol.svg)           | 7B/20B                        |  稠密LLM   |    Legacy    |   1.3.2    |
| [Qwen](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md) ![End of Life](./docs/assets/eol.svg)                       | 7B/14B                        |  稠密LLM   |    Legacy    |   1.3.2    |
| [CodeGeex2](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md) ![End of Life](./docs/assets/eol.svg)          | 6B                            |  稠密LLM   |    Legacy    |   1.1.0    |
| [WizardCoder](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md) ![End of Life](./docs/assets/eol.svg)  | 15B                           |  稠密LLM   |    Legacy    |   1.1.0    |
| [Baichuan](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md) ![End of Life](./docs/assets/eol.svg)             | 7B/13B                        |  稠密LLM   |    Legacy    |    1.0     |
| [Blip2](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md) ![End of Life](./docs/assets/eol.svg)                    | 8.1B                          |    MM    |    Legacy    |    1.0     |
| [Bloom](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md) ![End of Life](./docs/assets/eol.svg)                    | 560M/7.1B/65B/176B            |  稠密LLM   |    Legacy    |    1.0     |
| [Clip](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md) ![End of Life](./docs/assets/eol.svg)                      | 149M/428M                     |    MM    |    Legacy    |    1.0     |
| [CodeGeex](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md) ![End of Life](./docs/assets/eol.svg)             | 13B                           |  稠密LLM   |    Legacy    |    1.0     |
| [GLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md) ![End of Life](./docs/assets/eol.svg)                        | 6B                            |  稠密LLM   |    Legacy    |    1.0     |
| [iFlytekSpark](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) ![End of Life](./docs/assets/eol.svg) | 13B                           |  稠密LLM   |    Legacy    |    1.0     |
| [Llama](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md) ![End of Life](./docs/assets/eol.svg)                    | 7B/13B                        |  稠密LLM   |    Legacy    |    1.0     |
| [MAE](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md) ![End of Life](./docs/assets/eol.svg)                        | 86M                           |    MM    |    Legacy    |    1.0     |
| [Mengzi3](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) ![End of Life](./docs/assets/eol.svg)                | 13B                           |  稠密LLM   |    Legacy    |    1.0     |
| [PanguAlpha](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md) ![End of Life](./docs/assets/eol.svg)          | 2.6B/13B                      |  稠密LLM   |    Legacy    |    1.0     |
| [SAM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md) ![End of Life](./docs/assets/eol.svg)                        | 91M/308M/636M                 |    MM    |    Legacy    |    1.0     |
| [Skywork](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md) ![End of Life](./docs/assets/eol.svg)                | 13B                           |  稠密LLM   |    Legacy    |    1.0     |
| [Swin](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md) ![End of Life](./docs/assets/eol.svg)                      | 88M                           |    MM    |    Legacy    |    1.0     |
| [T5](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md) ![End of Life](./docs/assets/eol.svg)                          | 14M/60M                       |  稠密LLM   |    Legacy    |    1.0     |
| [VisualGLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md) ![End of Life](./docs/assets/eol.svg)          | 6B                            |    MM    |    Legacy    |    1.0     |
| [Ziya](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md) ![End of Life](./docs/assets/eol.svg)                         | 13B                           |  稠密LLM   |    Legacy    |    1.0     |
| [Bert](https://atomgit.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md) ![End of Life](./docs/assets/eol.svg)                      | 4M/110M                       |  稠密LLM   |    Legacy    |    0.8     |

![End of Life](./docs/assets/eol.svg) 表示模型已经从主干分支下线，可以通过最新支持的版本进行使用。

模型维护策略跟随最新支持版本的[生命周期及版本配套策略](#四生命周期及版本配套策略)。

### 模型级别介绍

Mcore架构模型按照训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。库中各模型不同规格的级别，详见模型自述文档。

#### 训练

- `Released`（发布级）：通过测试团队验收，确定性条件下，loss 与 grad norm 精度与标杆拟合度满足标准；
- `Validated`（验证级）：通过开发团队自验证，确定性条件下，loss 与 grad norm 精度与标杆拟合度满足标准；
- `Preliminary`（初步级）：通过开发者初步自验证，功能完整可试用，训练正常收敛但精度未严格验证；
- `Untested`（未测试级）：功能可用但未经系统测试，精度和收敛性未验证，支持用户自定义开发使能；
- `Community`（社区级）：社区贡献的 MindSpore 原生模型，由社区开发维护。

#### 推理

- `Released`（发布级）：通过测试团队验收，评测精度与标杆满足对齐标准；
- `Validated`（验证级）：通过开发团队自验证，评测精度与标杆满足对齐标准；
- `Preliminary`（初步级）：通过开发者初步自验证，功能完整可试用，推理输出符合逻辑但精度未严格验证；
- `Untested`（未测试级）：功能可用但未经系统测试，精度未验证，支持用户自定义开发使能；
- `Community`（社区级）：社区贡献的 MindSpore 原生模型，由社区开发维护。

## 二、安装

### 版本匹配关系

当前支持的硬件为 Atlas 800T A2、Atlas 800I A2、Atlas 900 A3 SuperPoD。

当前套件建议使用的Python版本为3.11.4。

| MindSpore Transformers | MindSpore | CANN | 固件与驱动 |
|:----------------------:|:---------:|:----:|:-----:|
|          在研版本          |   在研版本    | 在研版本 | 在研版本  |

历史版本配套关系：

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                                      固件与驱动                                                      |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.8.0          |   [2.7.2](https://www.mindspore.cn/install)   |   [8.5.0](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)   |   [25.5.0](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)   |
|         1.7.0          |   [2.7.1](https://www.mindspore.cn/install)   | [8.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) | [25.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) |
|         1.6.0          |   [2.7.0](https://www.mindspore.cn/install)   | [8.2.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html) |  [25.2.0](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html)  |
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

### 源码编译安装

MindSpore Transformers目前支持源码编译安装，用户可以执行如下命令进行安装。

```shell
git clone -b master https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、使用指南

MindSpore Transformers支持一键启动大模型的分布式[预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html)、[SFT 微调](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/supervised_fine_tuning.html)、[推理](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/inference.html)任务，可点击[模型列表](#模型列表)中各模型的链接查看对应使用文档。

关于MindSpore Transformers的更多功能说明可参阅[MindSpore Transformers文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/index.html)。

## 四、生命周期及版本配套策略

MindSpore Transformers版本有以下五个维护阶段：

|   **状态**    | **期限** | **说明**                         |
|:-----------:|:------:|:-------------------------------|
|     计划      | 1-3 个月 | 规划功能。                          |
|     开发      |  3 个月  | 构建功能。                          |
|     维护      |  6 个月  | 合入所有已解决的问题并发布新版本。              |
|     无维护     | 0-3 个月 | 合入所有已解决的问题，没有专职维护团队，且不计划发布新版本。 |
| 生命周期终止（EOL） |  N/A   | 分支进行封闭，不再接受任何修改。               |

MindSpore Transformers已发布版本维护策略：

| **MindSpore Transformers版本** | **对应标签** | **当前状态** |  **发布时间**  |      **后续状态**       | **EOL日期**  |
|:----------------------------:|:--------:|:--------:|:----------:|:-------------------:|:----------:|
|            1.8.0             |  v1.8.0  |    维护    | 2026/01/26 |  预计2026/07/26起无维护   | 2026/10/26 |
|            1.7.0             |  v1.7.0  |    维护    | 2025/10/27 |  预计2026/04/27起无维护   | 2026/07/27 |
|            1.6.0             |  v1.6.0  |   无维护    | 2025/07/29 | 预计2026/04/29起生命周期终止 | 2026/04/29 |
|            1.5.0             |  v1.5.0  |  生命周期终止  | 2025/04/29 |          -          | 2026/01/29 |
|            1.3.2             |  v1.3.2  |  生命周期终止  | 2024/12/20 |          -          | 2025/09/20 |
|            1.2.0             |  v1.2.0  |  生命周期终止  | 2024/07/12 |          -          | 2025/04/12 |
|            1.1.0             |  v1.1.0  |  生命周期终止  | 2024/04/15 |          -          | 2025/01/15 |

## 五、免责声明

1. `scripts/examples`目录下的内容是作为参考示例提供的，并不构成商业发布产品的一部分，仅供用户参考。如需使用，需要用户自行负责将其转化为适合商业用途的产品，并确保进行安全防护，对于由此产生的安全问题，MindSpore Transformers 不承担安全责任。
2. 关于数据集， MindSpore Transformers 仅提示性地建议可用于训练的数据集， MindSpore Transformers 不提供任何数据集。用户使用任何数据集进行训练，都需确保训练数据的合法性与安全性，并自行承担以下风险：
   1. 数据投毒（Data Poisoning）：恶意篡改的训练数据可能导致模型产生偏见、安全漏洞或错误输出。
   2. 数据合规性：用户应确保数据采集、处理过程符合相关法律法规及隐私保护要求。
3. 如果您不希望您的数据集在 MindSpore Transformers 中被提及，或希望更新 MindSpore Transformers 中关于您的数据集的描述，请在AtomGit提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对 MindSpore Transformers 的理解和贡献。
4. 关于模型权重，用户下载、分发的模型权重需经可信来源验证，MindSpore Transformers 无法保证第三方权重的安全性。权重文件在传输、加载过程中可能被篡改，导致模型产生预期外的输出或安全漏洞。用户应自行承担使用第三方权重的风险，并确保在使用前对权重文件进行安全验证。
5. 关于从魔乐社区等下载的权重、词表、脚本等文件，需经可信来源验证，MindSpore Transformers 无法保证第三方文件的安全性。这些文件在使用时产生预期之外的功能问题、输出或安全漏洞，用户应自行承担风险。
6. MindSpore Transformers 根据用户设置的路径进行权重或日志的保存，用户设置时需避免使用系统文件目录。如因路径设置不当产生预期之外的系统问题等，用户应自行承担风险。

## 六、贡献

欢迎参与社区贡献，可参考[MindSpore Transformers贡献指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/contribution/mindformers_contribution.html)。

## 七、许可证

[Apache 2.0许可证](LICENSE)