## MindSpore Transformers 2.0.0 Release Notes

The following is the changelog for MindSpore Transformers 2.0.0 compared with 1.9.0, including key new features and bug fixes.

### New Features

* **Full PyNative (Dynamic Graph) Support:**

    * Added dynamic graph training capability, enabling LLM models to complete pre-training tasks in PyNative mode.

        Implemented dynamic graph multi-dimensional parallelism based on DTensor ([issue #2150](https://atomgit.com/mindspore/mindformers/issues/2150)), supporting data parallelism (FSDP), tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), and context parallelism (CP), decoupling computation from distributed parallelism to reduce development and maintenance costs;

    * Added foundational computation interfaces for LLMs in PyNative mode, including core components such as [GPTModel base class](https://atomgit.com/mindspore/mindformers/issues/2115), TransformerLayer, TransformerBlock, Attention, FlashAttention, MLA (Multi-Head Latent Attention), MoE Layer, RotaryPositionEmbedding, YarnRoPE, and VocabEmbedding, achieving a unified computation interface system across static and dynamic graphs;

    * Refactored [YAML configuration fields](https://gitcode.com/mindspore/mindformers/issues/2495) with clearer categorization between configuration modules, improving configuration usability;

    * [Checkpoint 2.0](https://atomgit.com/mindspore/mindformers/issues/2149) solution supports dynamic graph save/load and checkpoint resume workflows;

### New Models

Newly supported models:

| Model                  | Variants                                                        |
|------------------------|-----------------------------------------------------------------|
| DeepSeek-V3 (PyNative) | [1B](configs/deepseek3/pretrain_deepseek3_1b_8p_pynative.yaml)  |

### Bug Fixes

During this release cycle, we conducted bug fixes across multiple aspects including models, features, usability, and documentation. Here are some key fixes:

[!8476](https://atomgit.com/mindspore/mindformers/pull/8476): Added sqrtsoftplus support for MoE aux-loss routing.

[!8477](https://atomgit.com/mindspore/mindformers/pull/8477): Added per-branch YaRN RoPE support for DeepSeekV4 hybrid.

[!8380](https://atomgit.com/mindspore/mindformers/pull/8380): Modularized ExpertParallel and fixed EP overlap dual-thread conflict.

[!8401](https://atomgit.com/mindspore/mindformers/pull/8401): Enabled simultaneous MTP and mHC.

[!8392](https://atomgit.com/mindspore/mindformers/pull/8392): Supported force expert balance with TP.

[!8345](https://atomgit.com/mindspore/mindformers/pull/8345): Fixed MTP attention correctly applying CP under CP mode.

[!8451](https://atomgit.com/mindspore/mindformers/pull/8451): Fixed thread deadlock, resource leak, and unsafe YAML deserialization.

[!8452](https://atomgit.com/mindspore/mindformers/pull/8452): Fixed pyarrow CVE-2026-25087.

[!8379](https://atomgit.com/mindspore/mindformers/pull/8379): Fixed Pipeline Parallel training hang.

[!8443](https://atomgit.com/mindspore/mindformers/pull/8443): Fixed loss becoming NaN due to swap.

### Change Notes

This release includes changes to some historically deprecated models, code, and materials. Details:

| Change              | Description                                                                                        |
|---------------------|----------------------------------------------------------------------------------------------------|
| Sunset ckpt format  | The new version defaults to Checkpoint 2.0 format. The legacy checkpoint format (`.ckpt`) is gradually deprecated. |
| Sunset static graph | MindSpore Transformers 2.0 introduces PyNative (dynamic graph) mode. The legacy static graph mode will be gradually deprecated. |

### Contributors

Thanks to everyone who contributed during this release cycle:

[@alpha-junh](https://atomgit.com/alpha-junh) , [@bj-wang1](https://atomgit.com/bj-wang1) , [@chenrayray](https://atomgit.com/chenrayray) , [@DavidFFFan](https://atomgit.com/DavidFFFan) , [@hss-shuai](https://atomgit.com/hss-shuai) , [@husichao](https://atomgit.com/husichao) , [@JavaZeroo](https://atomgit.com/JavaZeroo) , [@jiaboxie](https://atomgit.com/jiaboxie) , [@jimmyisme1](https://atomgit.com/jimmyisme1) , [@kongziyi](https://atomgit.com/kongziyi) , [@lanshaozuishuai](https://atomgit.com/lanshaozuishuai) , [@lzy0920232](https://atomgit.com/lzy0920232) , [@niujunhao](https://atomgit.com/niujunhao) , [@renyujin](https://atomgit.com/renyujin) , [@smallsilly](https://atomgit.com/smallsilly) , [@Sunshine_Youngster](https://atomgit.com/Sunshine_Youngster) , [@wangjialin](https://atomgit.com/wangjialin) , [@wei_zhuoyi](https://atomgit.com/wei_zhuoyi) , [@wjlflyer](https://atomgit.com/wjlflyer) , [@xiaoqi-zhou](https://atomgit.com/xiaoqi-zhou) , [@xiejiabo](https://atomgit.com/xiejiabo) , [@yide12](https://atomgit.com/yide12) , [@yule100](https://atomgit.com/yule100) , [@zhangyihuiben](https://atomgit.com/zhangyihuiben) , [@zzzkeke](https://atomgit.com/zzzkeke)

Contributions in any form are welcome!