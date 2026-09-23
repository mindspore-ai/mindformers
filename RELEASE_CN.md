## MindSpore Transformers 2.1.0 Release Notes

以下为MindSpore Transformers套件2.1.0版本的变更日志，相较于2.0.0版本有以下关键新特性和bugfix。

### 新特性

* **动态图训练与推理能力增强：**

    * 新增动态图 [Trainer 前向推理接口](https://atomgit.com/mindspore/mindformers/issues/2536)，支持批量贪心解码，复用训练的多维并行与权重加载流程，通过 `run_mode: predict` 启动，支持文本及 JSONL 等输入和结果保存（[!8749](https://atomgit.com/mindspore/mindformers/pull/8749)）；
    * 新增 [MoE LoRA 微调](https://atomgit.com/mindspore/mindformers/issues/2596)，将 LoRA 适配扩展至路由专家，支持仅稠密层、仅专家层及两者联合微调，并适配分布式并行、断点续训和权重合并（[!8813](https://atomgit.com/mindspore/mindformers/pull/8813)）；
    * 支持动态调整全局 batch size，通过调整梯度累积步数逐步增大训练 batch，适配流水线并行及断点续训（[!8509](https://atomgit.com/mindspore/mindformers/pull/8509)）；新增可扩展的 TensorBoard 训练监控，统一记录训练指标与配置（[!8542](https://atomgit.com/mindspore/mindformers/pull/8542)）；
    * 新增 [NaN/Inf 跳步保护](https://atomgit.com/mindspore/mindformers/issues/2549)，可在全局梯度范数为 NaN/Inf 时跳过优化器更新，并设置连续跳步上限，避免无效训练持续运行；该能力默认关闭（[!8748](https://atomgit.com/mindspore/mindformers/pull/8748)）。

* **DSA 长序列训练：**

    * 新增动态图 DSA（DeepSeek Sparse Attention）核心组件，支持 Indexer 稠密预热与稀疏训练两个阶段，并接入模型并行及重计算（[!8505](https://atomgit.com/mindspore/mindformers/pull/8505)、[!8575](https://atomgit.com/mindspore/mindformers/pull/8575)）；
    * 新增 [IndexShare](https://atomgit.com/mindspore/mindformers/issues/2580)，支持跨层共享 Indexer 和 Top-K 索引，减少重复计算；支持 `leader` / `served` 监督方式及可选 Top-K 换出（[!8785](https://atomgit.com/mindspore/mindformers/pull/8785)、[!8803](https://atomgit.com/mindspore/mindformers/pull/8803)）；
    * 优化 DSA 稠密预热阶段的激活管理，新增可选的逐层反向能力，并适配流水线并行和 IndexShare，缓解长序列训练的显存压力（[!8796](https://atomgit.com/mindspore/mindformers/pull/8796)）。

* **并行与数据性能优化：**

    * 新增基于本地 Tensor 计算的 TP/SP 路径，参数仍由 DTensor 管理，减少激活重分布开销（[!8525](https://atomgit.com/mindspore/mindformers/pull/8525)）；优化 HSDP 梯度累积，将 replicate 维的 all-reduce 延迟至最后一个 micro-batch（[!8770](https://atomgit.com/mindspore/mindformers/pull/8770)）；
    * 新增 OrderedIndexDataLoader，并支持 [TND 长序列数据负载均衡](https://atomgit.com/mindspore/mindformers/issues/2491)，通过样本索引重排改善数据并行各 rank 的注意力计算负载（[!8569](https://atomgit.com/mindspore/mindformers/pull/8569)、[!8668](https://atomgit.com/mindspore/mindformers/pull/8668)）；
    * 优化 Muon 去冗余计算的负载分配、BF16 Newton-Schulz 计算及 HSDP 通信范围，降低热点 rank 的显存压力（[!8687](https://atomgit.com/mindspore/mindformers/pull/8687)）；重构 `exclude_op`，支持按模块、函数属性及 TP/EP 通信路径排除重计算（[!8577](https://atomgit.com/mindspore/mindformers/pull/8577)）。

* **权重方案：**

    * 补齐 [MindFormers→Hugging Face 权重转换](https://atomgit.com/mindspore/mindformers/issues/2526)，新增离线导出工具，支持基于模型声明的转换规则导出 safetensors，并从带有 metadata 的 rank 分片权重重建参数；采用按层处理方式控制内存开销（[!8709](https://atomgit.com/mindspore/mindformers/pull/8709)、[!8761](https://atomgit.com/mindspore/mindformers/pull/8761)）；
    * 优化动态图大规模集群的 [Checkpoint 保存与加载](https://atomgit.com/mindspore/mindformers/issues/2533)，通过 layout 去重汇聚、rank 0 集中构建与分发保存计划、缓存复用及统一写入 metadata，减少重复通信和 CPU 内存开销（[!8741](https://atomgit.com/mindspore/mindformers/pull/8741)、[!8745](https://atomgit.com/mindspore/mindformers/pull/8745)、[!8746](https://atomgit.com/mindspore/mindformers/pull/8746)）；
    * 新增 LoRA 微调后 safetensors 权重合并工具（[!8757](https://atomgit.com/mindspore/mindformers/pull/8757)）。

### 新模型

本版本主要扩展已有模型的能力，以下为新增支持的运行场景：

| 模型 | 规格 |
|------|------|
| DeepSeek-V4（PyNative） | Flash：新增 Trainer 前向推理（[!8749](https://atomgit.com/mindspore/mindformers/pull/8749)） |
| Qwen3（PyNative） | 8B：新增 Trainer 前向推理，并完成 Hugging Face 权重加载验证（[!8749](https://atomgit.com/mindspore/mindformers/pull/8749)） |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的bugfix，在此列举部分关键修复内容：

[!8836](https://atomgit.com/mindspore/mindformers/pull/8836)：修复静态图平衡存盘首次保存崩溃、分片文件冲突及 metadata 与实际权重内容不一致等问题。

[!8623](https://atomgit.com/mindspore/mindformers/pull/8623)、[!8717](https://atomgit.com/mindspore/mindformers/pull/8717)：修复冻结参数缺少优化器状态、PP 续训时 FP32 主权重被错误覆盖导致的精度偏差。

[!8816](https://atomgit.com/mindspore/mindformers/pull/8816)、[!8819](https://atomgit.com/mindspore/mindformers/pull/8819)：修复 `no_load_optim: true` 时仍校验优化器文件，以及 PP 下无 layout 参数续训找不到权重文件的问题。

[!8621](https://atomgit.com/mindspore/mindformers/pull/8621)、[!8639](https://atomgit.com/mindspore/mindformers/pull/8639)：修复静态图开启 `calculate_per_token_loss` 与 MTP 后编译失败，以及流水线 stage 数不少于 4 时执行死锁的问题。

[!8778](https://atomgit.com/mindspore/mindformers/pull/8778)：修复整层重计算 replay 提前停止引起的确定性问题。

[!8766](https://atomgit.com/mindspore/mindformers/pull/8766)：修复按注意力头维度切分 CP 时 QK clipping 写越界与死锁问题。

[!8769](https://atomgit.com/mindspore/mindformers/pull/8769)：修复 `linear_fc1` 融合权重布局未跟随模型配置，导致权重转换错误的问题。

[!8815](https://atomgit.com/mindspore/mindformers/pull/8815)：修复 Muon 去冗余 P2P scatter 在压缩 rank 域下的分片错位问题。

[!8821](https://atomgit.com/mindspore/mindformers/pull/8821)、[!8817](https://atomgit.com/mindspore/mindformers/pull/8817)：修复 SwiGLU 配置引起的 MLP 宽度不匹配，以及 dryrun 下 TND 序列长度传递引起的 FlashAttention 异常。

[!8786](https://atomgit.com/mindspore/mindformers/pull/8786)、[!8824](https://atomgit.com/mindspore/mindformers/pull/8824)、[!8825](https://atomgit.com/mindspore/mindformers/pull/8825)：更新第三方依赖并移除核心 BLEU 指标对 nltk 的依赖；补齐 YAML 深度检查，调整安装脚本的子进程调用方式，并恢复容器安装源的 TLS 校验。

### 变更说明

当前版本包含以下配置、依赖及使用方式变更：

| 变更内容 | 变更说明 |
|----------|----------|
| 第三方依赖 | `transformers` 由 `4.57.1` 调整为 `>=5.3.0`，`datasets` 由 `>=4.0.0` 调整为 `>=5.0.1`；移除默认依赖中的 `nltk`，ADGEN BLEU 指标改用内置实现。 |
| IndexShare 开发期配置迁移 | 使用过本周期早期实现的用户，需将 `dsa_index_share_size` 改为 `dsa_index_topk_freq`，将 `dsa_index_share_pattern` 改为 `dsa_indexer_types`；后者采用由 `full` / `shared` 组成、长度等于模型层数的列表，不再接受 `FSSS` 等紧凑字符串（[!8803](https://atomgit.com/mindspore/mindformers/pull/8803)）。 |
| 动态图推理 | 通过 `run_mode: predict` 和 `inference` 配置启用；权重与 tokenizer 从 `checkpoint.load_path` 加载。当前为批量贪心解码，PP 推理不支持 interleave、`overlap_b_f` 和 swap 组合（[!8749](https://atomgit.com/mindspore/mindformers/pull/8749)）。 |
| MF→HF 离线导出 | 按模型声明的 `weight_converters` 导出，当前可发现 DeepSeek-V4、Qwen3、Qwen3-MoE 的转换规则。生成的 `config.json` 是配置子集，使用下游工具前需核对并补齐缺失的模型或推理字段（[!8761](https://atomgit.com/mindspore/mindformers/pull/8761)）。 |
| Muon 运行环境 | 优化路径依赖 MindSpore 的 `op_precision.cube_math_type` 接口；缺少该接口时会明确报错，请确认所用 MindSpore 构建提供该能力（[!8687](https://atomgit.com/mindspore/mindformers/pull/8687)）。 |

### 贡献者

感谢以下在本次版本增量中提交合入 PR 的开发者：

[@alpha-junh](https://atomgit.com/alpha-junh) 、 [@husichao](https://atomgit.com/husichao) 、 [@jimmyisme1](https://atomgit.com/jimmyisme1) 、 [@lanshaozuishuai](https://atomgit.com/lanshaozuishuai) 、 [@lzy0920232](https://atomgit.com/lzy0920232) 、 [@qhzhuang11111111](https://atomgit.com/qhzhuang11111111) 、 [@senzhen-town](https://atomgit.com/senzhen-town) 、 [@smallsilly](https://atomgit.com/smallsilly) 、 [@Sunshine_Youngster](https://atomgit.com/Sunshine_Youngster) 、 [@wei_zhuoyi](https://atomgit.com/wei_zhuoyi) 、 [@xu-xianliang](https://atomgit.com/xu-xianliang) 、 [@zhangyihuiben](https://atomgit.com/zhangyihuiben) 、 [@zzzkeke](https://atomgit.com/zzzkeke)

欢迎以任何形式对项目提供贡献！
