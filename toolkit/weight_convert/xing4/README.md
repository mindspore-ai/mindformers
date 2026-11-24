# Xing4 权重转换

29B 和 285B 的离线转换复用 `mindformers/models/xing4` 的权重模板，覆盖 MLA、DSA、MoE、mHC 和追加的 MTP 层。转换使用 CPU，不需要 PyTorch。模型 YAML 与权重应使用相同型号及结构；285B Xing4.1-Preview 的 MTP 层数为 0。

在仓库根目录执行以下命令。

## HuggingFace → MindSpore Transformers

```bash
python -m toolkit.weight_convert.xing4.convert_xing4_hf_weight \
  --yaml_config /path/to/model.yaml \
  --input_path /path/to/original_hf \
  --output_path /path/to/converted_mf \
  --dtype source \
  --max_worker 1
```

输入是带索引的 HF safetensors 目录，或直接包含 safetensors 文件的目录。工具逐层处理，并输出完整 MF 权重、`param_name_map.json`、`metadata.json` 和从零开始的 `common.json`。微调时将 `checkpoint.load_path` 指向输出目录，并设置 `checkpoint.no_load_optim: true`。

`--max_worker` 控制按层转换的进程数，默认 `1`。多个进程各自读取和写入独立分片，主进程在全部完成后统一写出索引与元信息；每个进程持有一层完整权重，按主机内存容量设置并发数。

`--dtype source` 是默认值，保留来源精度；也可指定 `bf16`、`fp16` 或 `fp32`，其中 router correction bias 保留 FP32。缺少一组转换所需的权重时默认警告并跳过该组；`--strict_load` 可将其改为错误。已有 HF 权重也可以直接通过训练的 `checkpoint.load_path` 加载，无需预先离线转换。

## MindSpore Transformers → HuggingFace

```bash
python -m toolkit.weight_convert.convert_mf_to_hf \
  --yaml_config /path/to/model.yaml \
  --input_path /path/to/mf_checkpoint/iteration_00000200 \
  --output_path /path/to/exported_hf \
  --hf_config_path /path/to/original_hf/config.json \
  --dtype source \
  --max_worker 1
```

输入既可以是完整 MF 权重，也可以是包含 `metadata.json` 及其引用分片的分布式 checkpoint。工具按元数据恢复各轴的规则切分，忽略优化器状态，并输出 HF safetensors、索引和原生 `config.json`。29B 使用 `xing4_0` 身份，285B 使用 `deepseek_v32` 身份；MTP 的共享 embedding/head 也会恢复对应的 HF 权重条目。

`--dtype source` 保留微调后的各参数精度。`--hf_config_path` 用于保留 `auto_map` 等原始元数据，模型结构字段以 YAML 及型号默认值为准。Tokenizer 文件和 `auto_map` 引用的自定义 Python 文件应从原 HF 目录复制到输出目录。

每个进程会处理一层的完整权重。先使用 `--max_worker 1`，再根据主机内存容量增加并发数；输出目录应位于输入目录之外。
