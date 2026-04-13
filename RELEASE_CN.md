# MindSpore Transformers Release Notes

## MindSpore Transformers 1.9.0 Release Notes

以下为MindSpore Transformers套件1.9.0版本的变更日志，相较于1.8.0版本有以下关键新特性和bugfix。

### 新特性

* **训练功能：** 支持训练前向推理；开启流水线并行进行任务训练时，支持打印对应rank加载的参数信息。

* **模型支持：** 新增支持TeleChat3\-36B推理和预训练；新增支持TeleChat3\-105B预训练。

* **权重方案：** 权重2.0方案适配故障快恢功能；Hugging Face权重加载性能优化<sup id="fn1">[1]</sup>。

* **动态图能力（实验性）：** 支持Trainer拉起训练流程；支持Qwen3稠密模型搭建。

### 新模型

以下为新支持模型：

| 模型        | 规格                                                                                                                                                                                                                       |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TeleChat3 | [TeleChat3\-36B](https://gitcode.com/mindspore/mindformers/blob/r1.9.0-beta1/configs/telechat3)（预训练、推理）、[TeleChat3\-105B\-A4.7B](https://gitcode.com/mindspore/mindformers/blob/r1.9.0-beta1/configs/telechat3_moe)（预训练） |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的bugfix，在此列举部分关键修复内容：

* [\!8006](https://gitcode.com/mindspore/mindformers/pull/8006)：修复MOE模型Tflops值打印不正确的问题。

* [\!7874](https://gitee.com/mindspore/mindformers/pulls/7874)：修复MCore网络中pad\_token\_id不生效问题。

* [\!7818](https://gitee.com/mindspore/mindformers/pulls/7818)：修复部分环境下hostname获取失败问题。

* [\!7793](https://gitee.com/mindspore/mindformers/pulls/7793) [\!7713](https://gitee.com/mindspore/mindformers/pulls/7713)：修复Hugging Face数据集相关问题。

* [\!7630](https://gitee.com/mindspore/mindformers/pulls/7630)：修复变换并行策略时safetensors权重转换加载问题。

* [\!7620](https://gitee.com/mindspore/mindformers/pulls/7620)：修复VocabEmbedding在特定配置下通信引起的精度问题。

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容 | 变更说明      |
|------|-----------|
| 无    | 本版本暂无变更说明 |

### 贡献者

感谢以下所有在版本周期内参与贡献的开发者：

[@lanshaozuishuai](https://gitcode.com/lanshaozuishuai)、[@zyw\-hw](https://gitcode.com/zyw-hw)、[@smallsilly](https://gitcode.com/smallsilly)、[@wei\_zhuoyi](https://gitcode.com/wei_zhuoyi)、[@yule100](https://gitcode.com/yule100)、[@zzzkeke](https://gitcode.com/zzzkeke)、[@sunyu\-xuan](https://gitcode.com/sunyu-xuan)、[@alpha-junh](https://gitcode.com/alpha-junh)、[@zhangyihuiben](https://gitcode.com/zhangyihuiben)、[@jimmyisme1](https://gitcode.com/jimmyisme1)、[@yiyison](https://gitcode.com/yiyison)、[@huangjingwei](https://gitcode.com/huangjingwei)、[@chenrayray](https://gitcode.com/chenrayray)、[@Sunshine\_Youngster](https://gitcode.com/Sunshine_Youngster)、[@suhaibo](https://gitcode.com/suhaibo)、[@minghu111](https://gitcode.com/minghu111)、[@senzhen\-town](https://gitcode.com/senzhen-town)、[@limuan](https://gitcode.com/limuan)、[@husichao](https://gitcode.com/husichao)、[@xiaoqi\-zhou](https://gitcode.com/xiaoqi-zhou)、[@silkage_jiajia](https://gitcode.com/silkage_jiajia)、[@hss\-shuai](https://gitcode.com/hss-shuai)、[@pengjingyou](https://gitcode.com/pengjingyou)、[@wjlflyer](https://gitcode.com/wjlflyer)、[@shen\_haochen](https://gitcode.com/shen_haochen)、[@wujinyuan1](https://gitcode.com/wujinyuan1)、[@yyyyrf](https://gitcode.com/yyyyrf)、[@Somnus2020](https://gitcode.com/Somnus2020)、[@renyujin](https://gitcode.com/renyujin)、[@qsc97](https://gitcode.com/qsc97)、[@yinanf](https://gitcode.com/yinanf)、[@hangangqiang](https://gitcode.com/hangangqiang)、[@lzy0920232](https://gitcode.com/lzy0920232)

欢迎以任何形式对项目提供贡献！

<hr />

<ol>
<li id="fn1">实验测试千亿模型百卡集群权重加载时间缩短80%。</li>
</ol>