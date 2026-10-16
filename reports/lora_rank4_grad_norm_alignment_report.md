# LoRA rank=4 八卡并行泛化 grad_norm 对齐报告

## 1. 问题边界

- 八卡 PyNative 训练，对比 `configs/0001.yaml`（DP/FSDP8）与
  `configs/0006.yaml`（TP2/EP2/PP4）。
- 数据、基础 checkpoint、global batch、优化器和 LoRA 配置相同。
- 唯一关键变量是 `lora_rank`：rank=8 可对齐，rank=4 不对齐。
- 基础 checkpoint 不包含 LoRA 参数。
- 本次只修改 MindFormers；Hyper-Parallel 无代码修改。

修改前 rank=4 的首步 loss 基本一致，但 grad_norm 为 `0.090059` 对
`0.096178`，相对误差 6.794%。第 50 步相对误差达到 23.403%。

## 2. rank=4 与 rank=8 实际改变的代码路径

Dense LoRA A 的形状为 `(lora_rank, input_size)`。0001 的 FSDP shard size
是 8，而原 `_build_fsdp_policy` 对没有显式规则的参数只尝试 `Shard(0)`：

- rank=8：`8 % 8 == 0`，LoRA A 可以分片，进入 shard-domain Muon 路径；
- rank=4：`4 % 8 != 0`，LoRA A 被回退为 Replicate，进入 replica-local
  Muon 路径；
- 0006 的 DP=1，不存在上述 FSDP8 回退。

运行态元数据直接体现了这个分界：

| 场景 | 2D Muon layout decision | max batch/owner |
|---|---|---:|
| rank=4，修复前 0001 | `p2p_replica_local=54, p2p_shard_domain=54` | 18 |
| rank=8 的 shard 路径 | `p2p_shard_domain=108` | 3 |
| rank=4，修复后 0001 | `p2p_shard_domain=108` | 3 |

因此 rank=4 并非只改变矩阵宽度，还让一半 dense LoRA 2D 参数切换了
FSDP 存储布局和 Muon 更新域。这是 rank=4 独有、rank=8 不出现的实质差异。

## 3. 两个问题及其作用范围

### 3.1 独立存在的小问题：LoRA 初始化依赖进程 RNG

原 `init_states()` 按每个进程的模型构造顺序调用 `normal_()`。PP 与 FSDP
构造和初始化参数的顺序不同，同名 LoRA A 可能获得不同初值。修复是在并行
切分和 delayed init 后、checkpoint 加载前，使用 `training.seed + 参数完整名`
生成唯一全局值，再按 DTensor layout 取本地分片。

该修复是必要的，但不是 rank=4 后续发散的主修复。单独应用后的结果为：

| step | 0001 norm | 0006 norm | 相对误差 |
|---:|---:|---:|---:|
| 1 | 0.056490 | 0.056507 | 0.030% |
| 5 | 0.090310 | 0.095342 | 5.572% |
| 20 | 0.135829 | 0.172157 | 26.745% |
| 50 | 0.109223 | 0.139293 | 27.531% |

它修复了首步初值差异，但没有解决首次 optimizer 更新后的持续分叉。

### 3.2 主问题：dense LoRA A 的 FSDP fallback 选错布局

LoRA A 的 dim0 是低秩维，不应因为低秩维小于 FSDP degree 就把整个大矩阵
复制。其 dim1 (`input_size`) 可被 8 整除，应该作为安全的 fallback shard 维。
expert LoRA policy 已有“避免切低秩维、改切矩阵维”的规则，dense LoRA 缺失
同类处理。

修复规则严格限定为：

1. 参数名是 `.lora_a` 或 `_lora_a`；
2. 参数为 2D；
3. 没有调用方显式 `shard_dim` 规则；
4. 默认 dim0 不可整除、dim1 可整除 shard size。

满足时使用 `Shard(1)`，否则完全保留原行为。这样 rank=8、非 LoRA 参数、
显式 placement、推理和 checkpoint 语义都不受影响。

## 4. 排除项

- Muon singleton `mm/bmm` 修改已回退，不在最终代码中。
- 对实际 rank=4/8 窄矩阵执行 batch=3/6/18 的 Ascend 定向实验，所有输出
  逐元素相等，排除了 BMM batch cardinality。
- 3D routed-expert batch=64/32 的定向实验也逐元素相等，排除了 expert BMM。
- 没有修改 dropout、grad norm 算法、Muon Newton-Schulz 或 Hyper-Parallel。

## 5. 最终 50 步验证

代码更新到 PR 8861 的最新 beta1 基线 `7ab340402` 后，两套配置均重新执行
50 步：

- 0001：`logs/pr8860_8861_combined_0001_50step`；
- 0006：`logs/pr8860_8861_combined_0006_50step`。

两个日志均由 `worker_7.log` 提取到完整的 50 个 loss/grad_norm 记录。第 50 步
结果如下：

| step | 0001 loss | 0006 loss | loss 相对误差 | 0001 norm | 0006 norm | norm 相对误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 6.837346 | 6.837439 | 0.001360% | 0.138786 | 0.139623 | 0.603087% |

汇总：

- 50 步最大 grad_norm 相对误差：`0.603087%`（step 50）；
- 50 步平均 grad_norm 相对误差：`0.189911%`；
- 50 步最大 loss 相对误差：`0.001986%`（step 26）；
- 50 步平均 loss 相对误差：`0.000607%`；
- 两个任务均完成 50/50 步并以退出码 0 结束；
- 相比只修初始化时 step 50 的 `27.531%`，主问题已消除。

## 6. 回归与最终改动

- 现有 FSDP policy 与 LoRA 初始化定向测试：`12 passed`。LoRA 用例覆盖
  参数名派生 seed 及 pipeline partition independence。
- `git diff --check` 通过。
- PR 8860（master）与 PR 8861（r2.1.0-beta1）包含相同的确定性初始化修复；
  当前 beta1 工作分支在最新基线之上将其与 FSDP fallback 修复合并为一个提交。
- 最终生产代码仅涉及：
  - `mindformers/pynative/base_models/gpt/parallelize.py`；
  - `mindformers/pynative/pet/utils.py`；
  - `mindformers/pynative/trainer/trainer.py`。

结论：rank=4 不能对齐的本质原因是低秩维不可被 DP8 整除时，dense LoRA A
错误地从 FSDP shard 路径退化为 Replicate，进而切换到与 rank=8 不同的
replica-local Muon 更新路径。将其安全地 fallback 到非低秩 dim1 后，rank=4
恢复到稳定的 shard-domain 路径。更新到 PR 最新基线后重新执行的两组 50 步
实验仍满足精度对齐要求。
