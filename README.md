# MindSpore Transformers (MindFormers)

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 1. Introduction

The MindSpore Transformers suite aims to build a comprehensive development toolkit covering the entire lifecycle of large-scale models, including pre-training, fine-tuning, evaluation, inference, and deployment. It provides industry-leading large Transformer-based language models, multimodal understanding models, and omni-modal models, enabling users to easily achieve end-to-end development of large-scale models.

Based on MindSpore's built-in multi-dimensional hybrid parallelism technology and modular design, the MindSpore Transformers suite offers the following features:

- Configurable one-click launch for pre-training, fine-tuning, evaluation, inference, and deployment of large-scale models.
- Integration with mainstream ecosystems such as Hugging Face, Megatron-LM, vLLM, and OpenCompass.
- Rich multi-dimensional hybrid parallelism and debugging/tuning capabilities, supporting training for trillion-parameter models.
- System-level deep optimization for training and inference, enhancing performance for hundred-billion dense and trillion sparse large-scale models.
- High availability in training, ensuring stable operation of large models on clusters with tens of thousands of NPUs.
- Fine-grained, multi-level training monitoring to facilitate anomaly detection and analysis.
- Simplified model integration through Mcore architecture upgrades and modular design, offering broader standardization and stronger ecosystem support.

For details about MindSpore Transformers tutorials and API documents, see **[MindSpore Transformers Documentation](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/index.html)**. The following are quick jump links to some of the key content:

- 📝 [Pre-training](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/pre_training.html)
- 📝 [Supervised Fine-Tuning](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/supervised_fine_tuning.html)
- 📝 [Evaluation](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/evaluation.html)
- 📝 [Service-oriented Deployment](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/deployment.html)

If you have any suggestions on MindSpore Transformers, contact us through an issue, and we will address it promptly.

If you're interested in MindSpore Transformers technology or wish to contribute code, we welcome you to join the [MindSpore Transformers SIG](https://www.mindspore.cn/sig/MindSpore%20Transformers/en).

### Models List

The following table lists models supported by MindSpore Transformers.

| Model                                                                                                             | Specifications                |    Model Type     | Model Architecture | Latest Version |
|:------------------------------------------------------------------------------------------------------------------|:------------------------------|:-----------------:|:------------------:|:--------------:|
| [TeleChat3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/telechat3) `🔥HOT`                      | 36B                           |     Dense LLM     |       Mcore        |     1.9.0      |
| [TeleChat3-MoE](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/telechat3_moe) `🔥HOT`              | 105B-A4.7B                    |    Sparse LLM     |       Mcore        |     1.9.0      |
| [Qwen3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/qwen3) `🔥HOT`                              | 0.6B/1.7B/4B/8B/14B/32B       |     Dense LLM     |       Mcore        |     1.9.0      |
| [Qwen3-MoE](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/qwen3_moe) `🔥HOT`                      | 30B-A3B/235B-A22B             |    Sparse LLM     |       Mcore        |     1.9.0      |
| [DeepSeek-V3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/deepseek3) `🔥HOT`                   | 671B                          |    Sparse LLM     |    Mcore/Legacy    |     1.9.0      |
| [GLM4.5](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/glm4_moe) `🔥HOT`                          | 106B-A12B/355B-A32B           |    Sparse LLM     |       Mcore        |     1.9.0      |
| [GLM4](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/glm4) `🔥HOT`                                | 9B                            |     Dense LLM     |    Mcore/Legacy    |     1.9.0      |
| [Qwen2.5](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/qwen2_5) `🔥HOT`                         | 0.5B/1.5B/7B/14B/32B/72B      |     Dense LLM     |       Legacy       |     1.9.0      |
| [TeleChat2](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/telechat2) `🔥HOT`                     | 7B/35B/115B                   |     Dense LLM     |    Mcore/Legacy    |     1.9.0      |
| [Llama3.1](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/llama3_1) `⚠️EOL`                       | 8B/70B                        |     Dense LLM     |       Legacy       |     1.7.0      |
| [Mixtral](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/mixtral) `⚠️EOL`                         | 8x7B                          |    Sparse LLM     |       Legacy       |     1.7.0      |
| [CodeLlama](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md) `⚠️EOL`          | 34B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [CogVLM2-Image](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md) `⚠️EOL`  | 19B                           |        MM         |       Legacy       |     1.5.0      |
| [CogVLM2-Video](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md) `⚠️EOL`  | 13B                           |        MM         |       Legacy       |     1.5.0      |
| [DeepSeek-V2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2) `⚠️EOL`                   | 236B                          |    Sparse LLM     |       Legacy       |     1.5.0      |
| [DeepSeek-Coder-V1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5) `⚠️EOL`         | 7B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [DeepSeek-Coder](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek) `⚠️EOL`                 | 33B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [GLM3-32K](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/glm32k) `⚠️EOL`                         | 6B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [GLM3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md) `⚠️EOL`                    | 6B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [InternLM2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/internlm2) `⚠️EOL`                     | 7B/20B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama3.2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) `⚠️EOL`            | 3B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama3.2-Vision](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md) `⚠️EOL`       | 11B                           |        MM         |       Legacy       |     1.5.0      |
| [Llama3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/llama3) `⚠️EOL`                           | 8B/70B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [Qwen2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen2) `⚠️EOL`                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | Dense /Sparse LLM |       Legacy       |     1.5.0      |
| [Qwen1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5) `⚠️EOL`                         | 7B/14B/72B                    |     Dense LLM     |       Legacy       |     1.5.0      |
| [Qwen-VL](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl) `⚠️EOL`                          | 9.6B                          |        MM         |       Legacy       |     1.5.0      |
| [TeleChat](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/telechat) `⚠️EOL`                       | 7B/12B/52B                    |     Dense LLM     |       Legacy       |     1.5.0      |
| [Whisper](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md) `⚠️EOL`              | 1.5B                          |        MM         |       Legacy       |     1.5.0      |
| [Yi](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yi) `⚠️EOL`                                   | 6B/34B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [YiZhao](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yizhao) `⚠️EOL`                           | 12B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md) `⚠️EOL`                | 7B/13B/70B                    |     Dense LLM     |       Legacy       |     1.3.2      |
| [Baichuan2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md) `⚠️EOL`        | 7B/13B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [GLM2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md) `⚠️EOL`                    | 6B                            |     Dense LLM     |       Legacy       |     1.3.2      |
| [GPT2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md) `⚠️EOL`                    | 124M/13B                      |     Dense LLM     |       Legacy       |     1.3.2      |
| [InternLM](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md) `⚠️EOL`           | 7B/20B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [Qwen](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md) `⚠️EOL`                       | 7B/14B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [CodeGeex2](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md) `⚠️EOL`          | 6B                            |     Dense LLM     |       Legacy       |     1.1.0      |
| [WizardCoder](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md) `⚠️EOL`  | 15B                           |     Dense LLM     |       Legacy       |     1.1.0      |
| [Baichuan](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md) `⚠️EOL`             | 7B/13B                        |     Dense LLM     |       Legacy       |      1.0       |
| [Blip2](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md) `⚠️EOL`                    | 8.1B                          |        MM         |       Legacy       |      1.0       |
| [Bloom](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md) `⚠️EOL`                    | 560M/7.1B/65B/176B            |     Dense LLM     |       Legacy       |      1.0       |
| [Clip](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md) `⚠️EOL`                      | 149M/428M                     |        MM         |       Legacy       |      1.0       |
| [CodeGeex](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md) `⚠️EOL`             | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [GLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md) `⚠️EOL`                        | 6B                            |     Dense LLM     |       Legacy       |      1.0       |
| [iFlytekSpark](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) `⚠️EOL` | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Llama](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md) `⚠️EOL`                    | 7B/13B                        |     Dense LLM     |       Legacy       |      1.0       |
| [MAE](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md) `⚠️EOL`                        | 86M                           |        MM         |       Legacy       |      1.0       |
| [Mengzi3](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) `⚠️EOL`                | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [PanguAlpha](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md) `⚠️EOL`          | 2.6B/13B                      |     Dense LLM     |       Legacy       |      1.0       |
| [SAM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md) `⚠️EOL`                        | 91M/308M/636M                 |        MM         |       Legacy       |      1.0       |
| [Skywork](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md) `⚠️EOL`                | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Swin](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md) `⚠️EOL`                      | 88M                           |        MM         |       Legacy       |      1.0       |
| [T5](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md) `⚠️EOL`                          | 14M/60M                       |     Dense LLM     |       Legacy       |      1.0       |
| [VisualGLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md) `⚠️EOL`          | 6B                            |        MM         |       Legacy       |      1.0       |
| [Ziya](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md) `⚠️EOL`                         | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Bert](https://atomgit.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md) `⚠️EOL`                      | 4M/110M                       |     Dense LLM     |       Legacy       |      0.8       |

`⚠️EOL` indicates that the model has been offline from the main branch and can be used with the latest supported version (e.g., 1.7.0).

The model maintenance strategy follows the [Life Cycle And Version Matching Strategy](#4-life-cycle-and-version-matching-strategy) of the corresponding latest supported version.

### Model Level Introduction

The Mcore architecture model is divided into five levels for training and inference, respectively, representing different standards for model deployment. For details on the levels of different specifications of models in the library, please refer to the model documentation.

#### Training

- `Released`: Passed testing team verification, with loss and grad norm accuracy meeting benchmark alignment standards under deterministic conditions;
- `Validated`: Passed self-verification by the development team, with loss and grad norm accuracy meeting benchmark alignment standards under deterministic conditions;
- `Preliminary`: Passed preliminary self-verification by developers, with complete functionality and usability, normal convergence of training, but accuracy not strictly verified;
- `Untested`: Functionality is available but has not undergone systematic testing, with accuracy and convergence not verified, and support for user-defined development enablement;
- `Community`: Community-contributed MindSpore native models, developed and maintained by the community.

#### Inference

- `Released`: Passed testing team acceptance, with evaluation accuracy aligned with benchmark standards;
- `Validated`: Passed developer self-verification, with evaluation accuracy aligned with benchmark standards;
- `Preliminary`: Passed preliminary self-verification by developers, with complete functionality and usable for testing; inference outputs are logically consistent but accuracy has not been strictly verified;
- `Untested`: Functionality is available but has not undergone system testing; accuracy has not been verified; supports user-defined development enablement;
- `Community`: Community-contributed MindSpore native models, developed and maintained by the community.

## 2. Installation

### Version Mapping

Currently supported hardware includes Atlas 800T A2, Atlas 800I A2, and Atlas 900 A3 SuperPoD.

Python 3.11.4 is recommended for the current suite.

| MindSpore Transformers | MindSpore | CANN  | Driver/Firmware |
|:----------------------:|:---------:|:-----:|:---------------:|
|         1.9.0          |   2.9.0   | 9.0.0 |    26.0.RC1     |

Historical Version Supporting Relationships:

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                                 Driver/Firmware                                                 |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.8.0          |   [2.7.2](https://www.mindspore.cn/install)   |   [8.5.0](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)   |   [25.5.0](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)   |
|         1.7.0          |   [2.7.1](https://www.mindspore.cn/install)   | [8.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) | [25.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) |
|         1.6.0          |   [2.7.0](https://www.mindspore.cn/install)   | [8.2.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html) |  [25.2.0](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html)  |
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

### Installation Using the Source Code

Currently, MindSpore Transformers can be compiled and installed using the source code. You can run the following commands to install MindSpore Transformers:

```shell
git clone -b r1.9.0 https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 3. User Guide

MindSpore Transformers supports distributed [pre-training](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/pre_training.html), [supervised fine-tuning](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/supervised_fine_tuning.html), and [inference](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/guide/inference.html) tasks for large models with one click. You can click the link of each model in [Model List](#models-list) to see the corresponding documentation.

For more information about the functions of MindSpore Transformers, please refer to [MindSpore Transformers Documentation](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/index.html).

## 4. Life Cycle And Version Matching Strategy

MindSpore Transformers version has the following five maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                             |
|-------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------|
| Plan              | 1-3 months   | Planning function.                                                                                                          |
| Develop           | 3 months     | Build function.                                                                                                             |
| Preserve          | 6 months     | Incorporate all solved problems and release new versions.                                                                   |
| No Preserve       | 0-3 months   | Incorporate all the solved problems, there is no full-time maintenance team, and there is no plan to release a new version. |
| End of Life (EOL) | N/A          | The branch is closed and no longer accepts any modifications.                                                               |

[MindSpore Transformers released version preservation policy](https://gitcode.com/mindspore/mindformers/blob/master/README.md#4-life-cycle-and-version-matching-strategy)

## 5. Disclaimer

1. `scripts/examples` directory is provided as reference examples and do not form part of the commercially released products. They are only for users' reference. If it needs to be used, the user should be responsible for transforming it into a product suitable for commercial use and ensuring security protection. MindSpore Transformers does not assume security responsibility for the resulting security problems.
2. Regarding datasets, MindSpore Transformers only provides suggestions for datasets that can be used for training. MindSpore Transformers does not provide any datasets. Users who use any dataset for training must ensure the legality and security of the training data and assume the following risks:  
   1. Data poisoning: Maliciously tampered training data may cause the model to produce bias, security vulnerabilities, or incorrect outputs.
   2. Data compliance: Users must ensure that data collection and processing comply with relevant laws, regulations, and privacy protection requirements.
3. If you do not want your dataset to be mentioned in MindSpore Transformers, or if you want to update the description of your dataset in MindSpore Transformers, please submit an issue to AtomGit, and we will remove or update the description of your dataset according to your issue request. We sincerely appreciate your understanding and contribution to MindSpore Transformers.
4. Regarding model weights, users must verify the authenticity of downloaded and distributed model weights from trusted sources. MindSpore Transformers cannot guarantee the security of third-party weights. Weight files may be tampered with during transmission or loading, leading to unexpected model outputs or security vulnerabilities. Users should assume the risk of using third-party weights and ensure that weight files are verified for security before use.
5. Regarding weights, vocabularies, scripts, and other files downloaded from sources like openmind, users must verify the authenticity of downloaded and distributed model weights from trusted sources. MindSpore Transformers cannot guarantee the security of third-party files. Users should assume the risks arising from unexpected functional issues, outputs, or security vulnerabilities when using these files.
6. MindSpore Transformers saves weights or logs based on the path set by the user. Users should avoid using system file directories when configuring paths. If unexpected system issues arise due to improper path settings, users shall bear the risks themselves.

## 6. Contribution

We welcome contributions to the community. For details, see [MindSpore Transformers Contribution Guidelines](https://www.mindspore.cn/mindformers/docs/en/r1.9.0/contribution/mindformers_contribution.html).

## 7. License

[Apache 2.0 License](LICENSE)