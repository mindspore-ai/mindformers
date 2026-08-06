## MindSpore Transformers 2.0.0 Release Notes

以下为MindSpore Transformers套件2.0.0版本的变更日志，相较于1.9.0版本有以下关键新特性和bugfix。

### 新特性

* **动态图（PyNative）全流程支持：**

    * 新增动态图训练能力，支持LLM模型在动态图模式下完成预训练任务。

        基于 DTensor 实现动态图[多维并行](https://atomgit.com/mindspore/mindformers/issues/2150)，支持数据并行（FSDP）、张量并行（TP）、流水线并行（PP）、专家并行（EP）、上下文并行（CP）等多维分布式并行策略，达成计算与分布式并行解耦，减少开发和维护成本；

    * 新增动态图模式下大语言模型基础计算接口，包括[GPTModel基类](https://atomgit.com/mindspore/mindformers/issues/2115)、TransformerLayer、TransformerBlock、Attention、FlashAttention、MLA（Multi-Head Latent Attention）、MoE Layer、RotaryPositionEmbedding、YarnRoPE、VocabEmbedding等核心组件，实现动静一致的计算接口体系；

    * 重构 [YAML 配置字段](https://gitcode.com/mindspore/mindformers/issues/2495)，各个配置模块之间分类更加清晰，提高配置易用性；

    * [权重 2.0](https://atomgit.com/mindspore/mindformers/issues/2149) 方案支持动态图保存加载与断点续训流程；

### 新模型

以下为新支持模型：

| 模型                     | 规格                                                              |
|------------------------|-----------------------------------------------------------------|
| DeepSeek-V3（PyNative）  | [1B](configs/deepseek3/pretrain_deepseek3_1b_8p_pynative.yaml)  |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的bugfix，在此列举部分关键修复内容：

[!8476](https://atomgit.com/mindspore/mindformers/pull/8476)：MoE aux-loss 路由支持 sqrtsoftplus。

[!8477](https://atomgit.com/mindspore/mindformers/pull/8477)：DeepSeekV4 hybrid 支持 per-branch YaRN RoPE。

[!8380](https://atomgit.com/mindspore/mindformers/pull/8380)：ExpertParallel 模块化，并修复 EP overlap 双线程冲突。

[!8401](https://atomgit.com/mindspore/mindformers/pull/8401)：支持 MTP 与 mHC 同时开启。

[!8392](https://atomgit.com/mindspore/mindformers/pull/8392)：支持 force expert balance + TP。

[!8345](https://atomgit.com/mindspore/mindformers/pull/8345)：CP 下 MTP 注意力正确应用 CP。

[!8451](https://atomgit.com/mindspore/mindformers/pull/8451)：修复线程死锁、资源泄露及不安全 YAML 反序列化。

[!8452](https://atomgit.com/mindspore/mindformers/pull/8452)：修复 pyarrow 的 CVE-2026-25087。

[!8379](https://atomgit.com/mindspore/mindformers/pull/8379)：修复 Pipeline Parallel 训练卡死。

[!8443](https://atomgit.com/mindspore/mindformers/pull/8443)：修复 swap 导致 loss 为 NaN。

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容         | 变更说明                                                        |
|--------------|-------------------------------------------------------------|
| 日落 ckpt 权重格式 | 新版本默认使用权重2.0格式，旧版权重格式（`.ckpt`）逐步废弃。                         |
| 日落静态图模式      | MindSpore Transformers 2.0新增支持动态图（PyNative）模式，原有静态图模式将逐步废弃。 |

### 贡献者

感谢以下所有在版本周期内参与贡献的开发者：

[@alpha-junh](https://atomgit.com/alpha-junh) 、 [@bj-wang1](https://atomgit.com/bj-wang1) 、 [@chenrayray](https://atomgit.com/chenrayray) 、 [@DavidFFFan](https://atomgit.com/DavidFFFan) 、 [@hss-shuai](https://atomgit.com/hss-shuai) 、 [@husichao](https://atomgit.com/husichao) 、 [@JavaZeroo](https://atomgit.com/JavaZeroo) 、 [@jiaboxie](https://atomgit.com/jiaboxie) 、 [@jimmyisme1](https://atomgit.com/jimmyisme1) 、 [@kongziyi](https://atomgit.com/kongziyi) 、 [@lanshaozuishuai](https://atomgit.com/lanshaozuishuai) 、 [@lzy0920232](https://atomgit.com/lzy0920232) 、 [@niujunhao](https://atomgit.com/niujunhao) 、 [@renyujin](https://atomgit.com/renyujin) 、 [@smallsilly](https://atomgit.com/smallsilly) 、 [@Sunshine_Youngster](https://atomgit.com/Sunshine_Youngster) 、 [@wangjialin](https://atomgit.com/wangjialin) 、 [@wei_zhuoyi](https://atomgit.com/wei_zhuoyi) 、 [@wjlflyer](https://atomgit.com/wjlflyer) 、 [@xiaoqi-zhou](https://atomgit.com/xiaoqi-zhou) 、 [@xiejiabo](https://atomgit.com/xiejiabo) 、 [@yide12](https://atomgit.com/yide12) 、 [@yule100](https://atomgit.com/yule100) 、 [@zhangyihuiben](https://atomgit.com/zhangyihuiben) 、 [@zzzkeke](https://atomgit.com/zzzkeke)

欢迎以任何形式对项目提供贡献！