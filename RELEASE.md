## MindSpore Transformers 2.1.0 Release Notes

The following is the changelog for MindSpore Transformers 2.1.0 compared with 2.0.0, including key new features and bug fixes.

### New Features

* **PyNative Training and Inference Enhancements:**

    * Added a [Trainer inference interface](https://atomgit.com/mindspore/mindformers/issues/2536) with batched greedy decoding, reusing the training parallelization and checkpoint loading paths. Enable it with `run_mode: predict`; text and JSONL inputs and result export are supported ([!8749](https://atomgit.com/mindspore/mindformers/pull/8749));
    * Added [MoE LoRA fine-tuning](https://atomgit.com/mindspore/mindformers/issues/2596), extending adapters to routed experts. Supports dense-only, expert-only, and combined fine-tuning, with distributed parallelism, checkpoint resume, and weight merging ([!8813](https://atomgit.com/mindspore/mindformers/pull/8813));
    * Added global batch size ramp-up through dynamic gradient accumulation, with pipeline parallelism and checkpoint resume support ([!8509](https://atomgit.com/mindspore/mindformers/pull/8509)); added extensible TensorBoard monitoring for training metrics and configurations ([!8542](https://atomgit.com/mindspore/mindformers/pull/8542));
    * Added optional [NaN/Inf step skipping](https://atomgit.com/mindspore/mindformers/issues/2549). Optimizer updates can be skipped when the global gradient norm is non-finite, with a configurable limit on consecutive skipped steps. Disabled by default ([!8748](https://atomgit.com/mindspore/mindformers/pull/8748)).

* **DSA Long-Sequence Training:**

    * Added PyNative DeepSeek Sparse Attention (DSA) components for dense Indexer warm-up and sparse training, integrated with model parallelism and activation recomputation ([!8505](https://atomgit.com/mindspore/mindformers/pull/8505), [!8575](https://atomgit.com/mindspore/mindformers/pull/8575));
    * Added [IndexShare](https://atomgit.com/mindspore/mindformers/issues/2580) to share Indexers and Top-K indices across layers, reducing redundant computation. Supports `leader` / `served` supervision and optional Top-K offloading ([!8785](https://atomgit.com/mindspore/mindformers/pull/8785), [!8803](https://atomgit.com/mindspore/mindformers/pull/8803));
    * Improved activation management during dense warm-up and added optional layerwise backward execution, compatible with pipeline parallelism and IndexShare, to reduce memory pressure during long-sequence training ([!8796](https://atomgit.com/mindspore/mindformers/pull/8796)).

* **Parallelism and Data Performance:**

    * Added native local Tensor TP/SP execution while retaining DTensor-managed parameters, reducing activation redistribution overhead ([!8525](https://atomgit.com/mindspore/mindformers/pull/8525)); deferred HSDP replicate-dimension all-reduce to the final micro-batch during gradient accumulation ([!8770](https://atomgit.com/mindspore/mindformers/pull/8770));
    * Added OrderedIndexDataLoader and [TND long-sequence load balancing](https://atomgit.com/mindspore/mindformers/issues/2491), reordering sample indices to balance attention workloads across data-parallel ranks ([!8569](https://atomgit.com/mindspore/mindformers/pull/8569), [!8668](https://atomgit.com/mindspore/mindformers/pull/8668));
    * Improved Muon computation ownership, BF16 Newton-Schulz computation, and HSDP communication scope to reduce memory pressure on overloaded ranks ([!8687](https://atomgit.com/mindspore/mindformers/pull/8687)); refactored `exclude_op` to exclude modules, function attributes, and TP/EP communication paths from recomputation ([!8577](https://atomgit.com/mindspore/mindformers/pull/8577)).

* **Checkpoint and Weight Conversion:**

    * Implemented [MindFormers-to-Hugging Face weight conversion](https://atomgit.com/mindspore/mindformers/issues/2526) and an offline safetensors export tool. Uses model-declared conversion rules, reconstructs parameters from rank-sharded checkpoints with metadata, and processes weights by layer to control memory use ([!8709](https://atomgit.com/mindspore/mindformers/pull/8709), [!8761](https://atomgit.com/mindspore/mindformers/pull/8761));
    * Optimized PyNative [checkpoint saving and loading at scale](https://atomgit.com/mindspore/mindformers/issues/2533) through deduplicated layout collection, centralized save-plan construction and distribution on rank 0, cached plans, and consolidated metadata writes, reducing repeated communication and CPU memory use ([!8741](https://atomgit.com/mindspore/mindformers/pull/8741), [!8745](https://atomgit.com/mindspore/mindformers/pull/8745), [!8746](https://atomgit.com/mindspore/mindformers/pull/8746));
    * Added a tool for merging LoRA fine-tuned safetensors weights ([!8757](https://atomgit.com/mindspore/mindformers/pull/8757)).

### New Models

This release primarily extends existing models with the following newly supported scenarios:

| Model | Variants |
|-------|----------|
| DeepSeek-V4 (PyNative) | Flash: added Trainer inference ([!8749](https://atomgit.com/mindspore/mindformers/pull/8749)) |
| Qwen3 (PyNative) | 8B: added Trainer inference with validated Hugging Face checkpoint loading ([!8749](https://atomgit.com/mindspore/mindformers/pull/8749)) |

### Bug Fixes

During this release cycle, we conducted bug fixes across multiple aspects including models, features, usability, and documentation. Here are some key fixes:

[!8836](https://atomgit.com/mindspore/mindformers/pull/8836): Fixed Graph-mode balanced checkpoint failures on the first save, conflicting shard filenames, and inconsistencies between metadata and saved weights.

[!8623](https://atomgit.com/mindspore/mindformers/pull/8623), [!8717](https://atomgit.com/mindspore/mindformers/pull/8717): Fixed resume accuracy issues caused by missing optimizer states for frozen parameters and incorrectly overwritten FP32 master weights under pipeline parallelism.

[!8816](https://atomgit.com/mindspore/mindformers/pull/8816), [!8819](https://atomgit.com/mindspore/mindformers/pull/8819): Fixed unnecessary optimizer-file validation with `no_load_optim: true` and missing checkpoint-file lookup for parameters without layouts under pipeline parallelism.

[!8621](https://atomgit.com/mindspore/mindformers/pull/8621), [!8639](https://atomgit.com/mindspore/mindformers/pull/8639): Fixed Graph-mode compilation failures with `calculate_per_token_loss` and MTP, and execution deadlocks with four or more pipeline stages.

[!8778](https://atomgit.com/mindspore/mindformers/pull/8778): Fixed determinism issues caused by early termination of full-layer recomputation replay.

[!8766](https://atomgit.com/mindspore/mindformers/pull/8766): Fixed out-of-bounds writes and deadlocks in QK clipping with head-dimension context parallelism.

[!8769](https://atomgit.com/mindspore/mindformers/pull/8769): Fixed fused `linear_fc1` weight layouts not following model configuration during weight conversion.

[!8815](https://atomgit.com/mindspore/mindformers/pull/8815): Fixed incorrect Muon P2P scatter shard placement within compressed rank domains.

[!8821](https://atomgit.com/mindspore/mindformers/pull/8821), [!8817](https://atomgit.com/mindspore/mindformers/pull/8817): Fixed SwiGLU configuration causing MLP width mismatches and incorrect TND sequence-length handling for FlashAttention during dryrun.

[!8786](https://atomgit.com/mindspore/mindformers/pull/8786), [!8824](https://atomgit.com/mindspore/mindformers/pull/8824), [!8825](https://atomgit.com/mindspore/mindformers/pull/8825): Updated dependencies and removed the core BLEU metric's nltk dependency; added YAML depth checks, revised subprocess invocation in setup, and restored TLS verification for container package installation.

### Change Notes

This release includes the following configuration, dependency, and usage changes:

| Change | Description |
|--------|-------------|
| Third-party dependencies | Changed `transformers` from `4.57.1` to `>=5.3.0` and `datasets` from `>=4.0.0` to `>=5.0.1`. Removed `nltk` from default dependencies; ADGEN BLEU now uses a built-in implementation. |
| IndexShare development configuration migration | Users of the early implementation in this cycle must replace `dsa_index_share_size` with `dsa_index_topk_freq`, and `dsa_index_share_pattern` with `dsa_indexer_types`. The latter requires a list of `full` / `shared` entries matching the model layer count; compact strings such as `FSSS` are no longer accepted ([!8803](https://atomgit.com/mindspore/mindformers/pull/8803)). |
| PyNative inference | Enable through `run_mode: predict` and the `inference` configuration. Weights and tokenizer are loaded from `checkpoint.load_path`. Currently uses batched greedy decoding; PP inference does not support interleave, `overlap_b_f`, or swap ([!8749](https://atomgit.com/mindspore/mindformers/pull/8749)). |
| MF-to-HF offline export | Uses model-declared `weight_converters`; currently discovers rules for DeepSeek-V4, Qwen3, and Qwen3-MoE. The generated `config.json` is a subset of the full configuration. Check and supply missing model or inference fields before using downstream tools ([!8761](https://atomgit.com/mindspore/mindformers/pull/8761)). |
| Muon runtime requirement | The optimized path requires MindSpore's `op_precision.cube_math_type` API and raises an explicit error when it is unavailable. Ensure the installed MindSpore build provides this capability ([!8687](https://atomgit.com/mindspore/mindformers/pull/8687)). |

### Contributors

Thanks to the developers whose PRs were merged in this release increment:

[@alpha-junh](https://atomgit.com/alpha-junh) , [@husichao](https://atomgit.com/husichao) , [@jimmyisme1](https://atomgit.com/jimmyisme1) , [@lanshaozuishuai](https://atomgit.com/lanshaozuishuai) , [@lzy0920232](https://atomgit.com/lzy0920232) , [@qhzhuang11111111](https://atomgit.com/qhzhuang11111111) , [@senzhen-town](https://atomgit.com/senzhen-town) , [@smallsilly](https://atomgit.com/smallsilly) , [@Sunshine_Youngster](https://atomgit.com/Sunshine_Youngster) , [@wei_zhuoyi](https://atomgit.com/wei_zhuoyi) , [@xu-xianliang](https://atomgit.com/xu-xianliang) , [@zhangyihuiben](https://atomgit.com/zhangyihuiben) , [@zzzkeke](https://atomgit.com/zzzkeke)

Contributions in any form are welcome!
