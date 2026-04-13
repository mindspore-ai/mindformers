# MindSpore Transformers Release Notes

## MindSpore Transformers 1.9.0 Release Notes

The following is the changelog for MindSpore Transformers 1.9.0 compared with 1.8.0, including key new features and bug fixes.

### New Features

* **Training:** Supports forward inference during training; when pipeline parallelism is enabled for training jobs, parameter loading information for the corresponding rank can be printed.

* **Model support:** Added inference and pre-training for TeleChat3-36B; added pre-training for TeleChat3-105B.

* **Performance monitoring:** Extended the Profile performance monitoring module with timing tracking for the cluster’s first startup phase.

* **Checkpoint solution:** Checkpoint 2.0 is adapted for fast recovery from failures; optimizes Hugging Face weight loading performance<sup id="fn1">[1]</sup>.

* **PyNative Capability (Experimental):** Supports launching the training process via Trainer; supports the construction of Qwen3 dense models.

### New Models

Newly supported models:

| Model     | Variants                                                                                                                                                                                                                                           |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TeleChat3 | [TeleChat3-36B](https://gitcode.com/mindspore/mindformers/blob/r1.9.0-beta1/configs/telechat3) (pre-training, inference), [TeleChat3-105B-A4.7B](https://gitcode.com/mindspore/mindformers/blob/r1.9.0-beta1/configs/telechat3_moe) (pre-training) |

### Bug Fixes

During this release cycle we fixed issues across models, features, usability, documentation, and more. Key fixes include:

* [!8006](https://gitcode.com/mindspore/mindformers/pull/8006): Fixed incorrect TFLOPs printing for MoE models.

* [!7874](https://gitee.com/mindspore/mindformers/pulls/7874): Fixed `pad_token_id` not taking effect in MCore networks.

* [!7818](https://gitee.com/mindspore/mindformers/pulls/7818): Fixed hostname retrieval failures in some environments.

* [!7793](https://gitee.com/mindspore/mindformers/pulls/7793) [!7713](https://gitee.com/mindspore/mindformers/pulls/7713): Fixed Hugging Face dataset-related issues.

* [!7630](https://gitee.com/mindspore/mindformers/pulls/7630): Fixed safetensors weight conversion and loading when changing parallel strategies.

* [!7620](https://gitee.com/mindspore/mindformers/pulls/7620): Fixed accuracy issues caused by communication for VocabEmbedding under certain configurations.

### Change Notes

This release includes changes to some historically deprecated models, code, and materials. Details:

| Change | Description                      |
|--------|----------------------------------|
| None   | No change notes for this version |

### Contributors

Thanks to everyone who contributed during this release cycle:

[@lanshaozuishuai](https://gitcode.com/lanshaozuishuai), [@zyw-hw](https://gitcode.com/zyw-hw), [@smallsilly](https://gitcode.com/smallsilly), [@wei_zhuoyi](https://gitcode.com/wei_zhuoyi), [@yule100](https://gitcode.com/yule100), [@zzzkeke](https://gitcode.com/zzzkeke), [@sunyu-xuan](https://gitcode.com/sunyu-xuan), [@alpha-junh](https://gitcode.com/alpha-junh), [@zhangyihuiben](https://gitcode.com/zhangyihuiben), [@jimmyisme1](https://gitcode.com/jimmyisme1), [@yiyison](https://gitcode.com/yiyison), [@huangjingwei](https://gitcode.com/huangjingwei), [@chenrayray](https://gitcode.com/chenrayray), [@Sunshine_Youngster](https://gitcode.com/Sunshine_Youngster), [@suhaibo](https://gitcode.com/suhaibo), [@minghu111](https://gitcode.com/minghu111), [@senzhen-town](https://gitcode.com/senzhen-town), [@limuan](https://gitcode.com/limuan), [@husichao](https://gitcode.com/husichao), [@xiaoqi-zhou](https://gitcode.com/xiaoqi-zhou), [@silkage_jiajia](https://gitcode.com/silkage_jiajia), [@hss-shuai](https://gitcode.com/hss-shuai), [@pengjingyou](https://gitcode.com/pengjingyou), [@wjlflyer](https://gitcode.com/wjlflyer), [@shen_haochen](https://gitcode.com/shen_haochen), [@wujinyuan1](https://gitcode.com/wujinyuan1), [@yyyyrf](https://gitcode.com/yyyyrf), [@Somnus2020](https://gitcode.com/Somnus2020), [@renyujin](https://gitcode.com/renyujin), [@qsc97](https://gitcode.com/qsc97), [@yinanf](https://gitcode.com/yinanf), [@hangangqiang](https://gitcode.com/hangangqiang), [@lzy0920232](https://gitcode.com/lzy0920232)

Contributions in any form are welcome!

<hr />

<ol>
<li id="fn1">Experimental tests show that loading time for a hundred-billion-parameter model on a hundred-NPU cluster has been reduced by 80%.</li>
</ol>
