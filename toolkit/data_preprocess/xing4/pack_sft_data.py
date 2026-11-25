# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""极简 SFT 数据处理: 对话 jsonl -> 定长 packed Arrow 数据集。

两个阶段, 都是多进程并行, 内存和数据规模解耦 —— 不需要先把原始数据切片:

  阶段1  tokenize   按配比表权重采样, 流式逐行读 jsonl, 逐条 tokenize, 随机写进
                    NUM_SHARDS 个分片之一 (随机打散, 让不同来源的数据混进同一个桶,
                    装桶时长短搭配, 填充率更高)。
                    每个进程同时只持有一条样本, 内存 O(1), 和数据总量无关。

  阶段2  pack       每个分片各自贪心装桶到 MAX_LEN, 补 pad、造 labels, 直接落成一个
                    Arrow 分片。单进程内存 ≈ 总token数 / NUM_SHARDS * 36 字节,
                    总内存再乘以 --pack-workers。

内存怎么调: NUM_SHARDS 调大 => 单个装桶进程更省内存 (代价是尾桶变多, 每个分片末尾
            都会留一个装不满的桶, 所以 padding 浪费 ≈ NUM_SHARDS / 总桶数)。

输入 jsonl, 每行一条样本:
    {"system": "你是一个助手",
     "dialog": [{"role": "user", "content": "..."},
                {"role": "bot",  "content": "<think>...</think>...", "select": true},
                {"role": "tool", "content": "..."}]}
    role 取值: user / bot (或 assistant) / tool;  select 省略时默认 true。

输出 (datasets.load_from_disk 直接可读):
    <output>/summary/dataset_dict.json
    <output>/summary/train/{dataset_info.json, state.json, shard-*.arrow}

用法:
    python pack_sft_data.py --weights weights.xlsx --tokenizer /path/to/tokenizer \\
        --output ./work --max-len 262144 --shards 64

    python pack_sft_data.py --validate ./work/summary --tokenizer /path/to/tokenizer
"""

import argparse
import glob
import json
import math
import os
import random
import shutil
from contextlib import ExitStack
from multiprocessing import Pool

IGNORE_ID = -100
R_USER, R_BOT, R_TOOL = 0, 1, 2
_ROLE_CODE = {"user": R_USER, "bot": R_BOT, "assistant": R_BOT, "tool": R_TOOL}

_WORKER = {}


class Special:
    """模板里用到的特殊 token id, 从 tokenizer 查出来, 不硬编码。"""

    def __init__(self, tok):
        get = tok.get_vocab().get
        self.system = get("<_system>")
        self.user = get("<_user>")
        self.bot = get("<_bot>")
        self.observation = get("<_observation>")
        self.end = get("<_end>")
        self.pad = get("<_pad>")
        self.newline = tok("\n")["input_ids"]
        self.newline_len = len(self.newline)
        self.newline_ignores = [IGNORE_ID] * self.newline_len
        self.think = (get("<think>"), get("</think>"))

        missing = [n for n in ("system", "user", "bot", "observation", "end", "pad")
                   if getattr(self, n) is None]
        if missing:
            raise RuntimeError(f"tokenizer 里查不到这些特殊 token: {missing}")

    def describe(self):
        """Describe the tokenizer markers used by the conversation template."""
        return (f"<_system>={self.system} <_user>={self.user} <_bot>={self.bot} "
                f"<_observation>={self.observation} <_end>={self.end} <_pad>={self.pad} "
                f"<think>/</think>={self.think} newline={self.newline}")


def _init_worker(tk_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tk_path, trust_remote_code=True)
    _WORKER.update(tokenizer=tokenizer, special=Special(tokenizer))


# --------------------------------------------------------------------------
# 阶段1: 流式 tokenize + 按权重采样 + 随机分片
# --------------------------------------------------------------------------

def tokenize_sample(sample, sp):
    """一条样本 -> {"n": 拼接后长度, "s": system ids, "d": [[role, ids, select], ...]}

    n 必须和阶段2真正写进序列的 token 数严格一致, 否则会装出超过 MAX_LEN 的桶:
        system :  <_system> + ids                  -> len + 1
        user   :  <_user>   + ids                  -> len + 1
        tool   :  <_observation> + ids             -> len + 1
        bot    :  <_bot> + ids + <_end> + "\\n"     -> len + 2 + newline_len
    """
    sys_text = sample.get("system") or ""
    sys_ids = _WORKER["tokenizer"](sys_text)["input_ids"] if sys_text else []
    length = len(sys_ids) + 1

    turns = []
    for turn in sample.get("dialog") or []:
        role = _ROLE_CODE.get(turn.get("role"))
        if role is None:
            return None                      # 非法 role, 整条丢弃
        ids = _WORKER["tokenizer"](turn.get("content") or "")["input_ids"]
        select = bool(turn.get("select", True))
        turns.append([role, ids, select])
        length += len(ids) + (2 + sp.newline_len if role == R_BOT else 1)

    if not turns or turns[-1][0] != R_BOT:
        # 最后一轮必须是 bot: 模板靠结尾的 <_end> 定位样本边界, 而且一条
        # 不以回复结尾的样本没有任何 loss, 留着只会白占位置
        return None
    return {"n": length, "s": sys_ids, "d": turns}


def tokenize_file(task):
    """处理一个输入 jsonl: 采样 -> tokenize -> 随机写进 NUM_SHARDS 个分片文件。

    权重语义和原流程一致:
        w <  1 : 每条按概率 w 保留 (先丢弃再 tokenize, 省算力)
        w >= 1 : 整数部分复制 floor(w) 份, 小数部分按概率再来一份
    复制出的多份会落到不同的随机分片, 避免同一个桶里塞进重复样本。
    """
    path, weight, idx, cfg = task
    rng = random.Random(f"{cfg['seed']}:{idx}")
    sp = _WORKER["special"]
    shards = cfg["shards"]

    kept = read = bad = toolong = 0
    with ExitStack() as stack:
        writers = [stack.enter_context(open(
            os.path.join(cfg["tmp_dir"], f"{idx:05d}.s{j:04d}.jsonl"), "w", encoding="utf-8"))
                   for j in range(shards)]
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                read += 1

                # 先定复制几份, 0 份就完全不用 tokenize
                if weight < 1.0:
                    reps = 1 if rng.random() < weight else 0
                else:
                    reps = int(math.floor(weight))
                    if rng.random() < weight - reps:
                        reps += 1
                if reps == 0:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                try:
                    tok = tokenize_sample(sample, sp)
                except (TypeError, ValueError, KeyError, OverflowError):
                    bad += 1
                    continue
                if tok is None:
                    bad += 1
                    continue
                if tok["n"] >= cfg["max_len"]:
                    toolong += 1          # 单条就超过 MAX_LEN, 装不进任何桶
                    continue

                blob = json.dumps(tok, ensure_ascii=False)
                for _ in range(reps):
                    writers[rng.randrange(shards)].write(blob + "\n")
                    kept += 1

    return {"file": os.path.basename(path), "read": read, "kept": kept,
            "bad": bad, "toolong": toolong}


# --------------------------------------------------------------------------
# 阶段2: 贪心装桶 + 造 labels + 写 Arrow
# --------------------------------------------------------------------------

def greedy_pack(samples, max_len, threshold, window):
    """从头扫, 能塞下就塞; 填到 max_len*threshold 或连续 window 条塞不下就封桶。"""
    buckets = []
    while samples:
        ptr, used, bucket, miss = 0, 0, [], 0
        while ptr < len(samples) and used < max_len * threshold and miss < window:
            if used + samples[ptr]["n"] < max_len:
                used += samples[ptr]["n"]
                bucket.append(samples.pop(ptr))
                miss = 0
            else:
                miss += 1
                ptr += 1
        if not bucket:                    # 剩下的都塞不进去了
            break
        buckets.append(bucket)
    return buckets


def build_row(bucket, sp, max_len, label_shift, use_hybrid):
    """一个桶 -> 一条定长训练样本。

    labels 规则:
      * 非 bot 轮、以及 select=False 的 bot 轮, 全部 -100
      * bot 轮计 loss 时: <_bot> 不计, <_end> 计, 结尾换行不计
      * use_hybrid=True : bot 回复首 token (<think>/</think>) 不计 loss,
                          不让模型去学"这一轮要不要思考"
      * 两种模式均检查 <think>/</think> 开头; 不符合格式的轮不计 loss
      * label_shift=True: labels 整体左移一位 (MindSpore 训练框架需要;
                          Megatron 系框架自己会移位, 要置 False)
    """
    ids, labels, bounds = [], [], []
    skipped = 0

    for s in bucket:
        ids.append(sp.system)
        ids.extend(s["s"])
        labels.extend([IGNORE_ID] * (len(s["s"]) + 1))

        for role, content, select in s["d"]:
            if role == R_USER:
                ids.append(sp.user)
                ids.extend(content)
                labels.extend([IGNORE_ID] * (len(content) + 1))
            elif role == R_TOOL:
                ids.append(sp.observation)
                ids.extend(content)
                labels.extend([IGNORE_ID] * (len(content) + 1))
            else:
                ids.append(sp.bot)
                ids.extend(content)
                ids.append(sp.end)
                ids.extend(sp.newline)
                if select and content and content[0] in sp.think:
                    labels.append(IGNORE_ID)                 # <_bot>
                    if use_hybrid:
                        labels.append(IGNORE_ID)             # <think> / </think>
                        labels.extend(content[1:])
                    else:
                        labels.extend(content)
                    labels.append(sp.end)
                    labels.extend(sp.newline_ignores)
                else:
                    if select:
                        skipped += 1
                    labels.extend([IGNORE_ID] * (len(content) + 2 + sp.newline_len))

        assert ids[-sp.newline_len - 1] == sp.end, "<_end> 位置未对齐, 模板拼错了"
        # 每条样本在拼接序列里的结束位置(不含结尾换行), 训练时据此切分 attention,
        # 同一个桶里的样本互相看不见
        bounds.append(len(ids) - sp.newline_len)

    assert len(ids) == len(labels) <= max_len, f"长度不一致 ids={len(ids)} labels={len(labels)}"
    ids.extend([sp.pad] * (max_len - len(ids)))
    labels.extend([IGNORE_ID] * (max_len - len(labels)))
    if label_shift:
        labels.append(IGNORE_ID)
        del labels[0]

    return {"input_ids": ids, "labels": labels, "actual_seq_len": bounds}, skipped


def pack_shard(task):
    """一个分片: 读回它的所有 tokenize 结果 -> 装桶 -> 写成一个 Arrow 分片。"""
    shard_id, cfg = task
    import datasets
    datasets.disable_progress_bars()

    sp = _WORKER["special"]
    max_len = cfg["max_len"]
    rng = random.Random(f"{cfg['seed']}:pack:{shard_id}")

    samples = []
    for path in sorted(glob.glob(os.path.join(cfg["tmp_dir"], f"*.s{shard_id:04d}.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                samples.append(json.loads(line))
    if not samples:
        return {"shard": shard_id, "rows": 0, "tokens": 0, "samples": 0, "skipped": 0}

    n_samples = len(samples)
    rng.shuffle(samples)
    buckets = greedy_pack(samples, max_len, cfg["threshold"], cfg["window"])
    del samples

    # loss_mask / position_ids 每行都一样, 共享同一个只读 list, 不要原地改
    loss_mask = [1] * max_len
    position_ids = list(range(max_len))

    rows, tokens, skipped = [], 0, 0
    for bucket in buckets:
        row, sk = build_row(bucket, sp, max_len, cfg["label_shift"], cfg["use_hybrid"])
        tokens += row["actual_seq_len"][-1]
        skipped += sk
        row["loss_mask"] = loss_mask
        row["position_ids"] = position_ids
        rows.append(row)
    del buckets

    if not rows:
        return {"shard": shard_id, "rows": 0, "tokens": 0, "samples": n_samples, "skipped": 0}

    n_rows = len(rows)
    stage = os.path.join(cfg["tmp_dir"], f"_arrow_{shard_id:04d}")
    datasets.Dataset.from_list(rows).save_to_disk(stage, max_shard_size="20000MB")
    del rows

    produced = sorted(f for f in os.listdir(stage) if f.endswith(".arrow"))
    assert len(produced) == 1, f"分片 {shard_id} 产出了 {len(produced)} 个 arrow, 预期 1 个"
    shutil.move(os.path.join(stage, produced[0]),
                os.path.join(cfg["train_dir"], f"shard-{shard_id:05d}.arrow"))
    # dataset_info.json / state.json 留着, 合并元数据时要从第一个分片抄一份
    return {"shard": shard_id, "rows": n_rows, "tokens": tokens,
            "samples": n_samples, "skipped": skipped, "stage": stage}


# --------------------------------------------------------------------------
# 元数据合并 / 校验
# --------------------------------------------------------------------------

def write_metadata(summary_dir, train_dir, stage_dir):
    """把所有 arrow 分片登记进 state.json, 让 load_from_disk 一次读到全部数据。"""
    arrows = sorted(f for f in os.listdir(train_dir) if f.endswith(".arrow"))
    if not arrows:
        raise RuntimeError("没有任何 Arrow 分片, 检查输入数据和权重")

    shutil.copy2(os.path.join(stage_dir, "dataset_info.json"),
                 os.path.join(train_dir, "dataset_info.json"))
    with open(os.path.join(stage_dir, "state.json"), "r", encoding="utf-8") as f:
        state = json.load(f)
    state["_data_files"] = [{"filename": name} for name in arrows]
    state["_fingerprint"] = "packed_sft"
    with open(os.path.join(train_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(os.path.join(summary_dir, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": ["train"]}, f)
    return len(arrows)


def validate(summary_dir, tokenizer_path, max_len, label_shift, preview):
    """结构校验 + Arrow 完整性 + 填充率统计 + 解码预览。"""
    import datasets
    import pyarrow as pa

    print(f"数据集: {summary_dir}")
    train_dir = os.path.join(summary_dir, "train")
    for name in ("dataset_dict.json", "train/dataset_info.json", "train/state.json"):
        path = os.path.join(summary_dir, name)
        assert os.path.exists(path), f"缺元数据: {name}"
    print("[OK  ] 元数据齐全")

    with open(os.path.join(train_dir, "state.json"), "r", encoding="utf-8") as f:
        listed = [d["filename"] for d in json.load(f)["_data_files"]]
    assert listed, "state.json 里没有登记任何 arrow"
    for name in listed:
        path = os.path.join(train_dir, name)
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"arrow 缺失或为空: {name}"
    print(f"[OK  ] {len(listed)} 个 Arrow 分片齐全且非空")

    # 只读 actual_seq_len 这一列 (Arrow 是列存, 不会把整个数据集读进来)
    rows = tokens = 0
    for name in listed:
        with pa.memory_map(os.path.join(train_dir, name), "rb") as src:
            col = pa.ipc.open_stream(src).read_all().column("actual_seq_len")
            for v in col:
                bounds = v.as_py()
                assert bounds, "actual_seq_len 为空"
                assert bounds == sorted(bounds), "actual_seq_len 不递增"
                assert bounds[-1] <= max_len, f"样本边界 {bounds[-1]} 超过 max_len"
                tokens += bounds[-1]
                rows += 1
    print(f"[OK  ] {rows} 行, 样本边界全部合法")
    print(f"[INFO] 有效 token {tokens / 1e9:.3f} B, 平均填充率 "
          f"{100 * tokens / (rows * max_len):.2f}% (padding 浪费 "
          f"{100 * (1 - tokens / (rows * max_len)):.2f}%)")

    ds = datasets.load_from_disk(summary_dir)
    assert ds["train"].num_rows == rows, "load_from_disk 行数和 arrow 对不上"
    expect = {"input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"}
    assert set(ds["train"].column_names) == expect, f"列不对: {ds['train'].column_names}"
    print(f"[OK  ] load_from_disk 成功: {rows} 行 x {max_len}")

    row = ds["train"][0]
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        assert len(row[key]) == max_len, f"{key} 长度 {len(row[key])} != {max_len}"
    print(f"[OK  ] 定长校验通过 ({max_len})")

    # labels 必须和 input_ids 对得齐: 非 -100 的位置, 要么等于同位 token,
    # 要么(左移时)等于下一位 token。第 0 行可能整行都不计 loss, 往后找一行有的
    checked = aligned = 0
    probe = -1
    for k in range(min(rows, 20)):
        cand = ds["train"][k]
        if any(x != IGNORE_ID for x in cand["labels"]):
            probe, row = k, cand
            break
    assert probe >= 0, "前 20 行都没有计 loss 的位置, 检查 select 字段和 <think> 开头"

    ids, labels = row["input_ids"], row["labels"]
    for i, lab in enumerate(labels):
        if lab == IGNORE_ID:
            continue
        checked += 1
        ref = ids[i + 1] if label_shift and i + 1 < max_len else ids[i]
        if lab == ref:
            aligned += 1
    assert aligned == checked, (
        f"labels 和 input_ids 对不齐: {aligned}/{checked} (label_shift={label_shift})")
    print(f"[OK  ] labels 对齐校验通过 (第 {probe} 行, {checked} 个计 loss 的位置, "
          f"label_shift={label_shift})")

    if preview and tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        bounds = row["actual_seq_len"]
        valid = bounds[-1]
        loss_n = sum(1 for x in labels if x != IGNORE_ID)
        print(f"\n[预览] 第 {probe} 行: 桶内 {len(bounds)} 条样本, 有效 token {valid} "
              f"({100 * valid / max_len:.1f}%), 计 loss {loss_n} "
              f"({100 * loss_n / valid:.1f}% of 有效)")
        print("-" * 70)
        print(tok.decode(ids[:bounds[0]])[:1200])
        print("-" * 70)
    print("\n[SUMMARY] 全部通过 ✓")


# --------------------------------------------------------------------------

def load_weights(weight_file, data_root):
    """Read source paths and sampling weights from the input workbook."""
    import pandas as pd
    df = pd.read_excel(weight_file, sheet_name="Sheet1")
    missing = {"文件名", "权重"} - set(df.columns)
    assert not missing, f"配比表缺少列 {missing}, 实际列: {list(df.columns)}"

    items = []
    for name, weight in zip(df["文件名"], df["权重"]):
        path = os.path.join(data_root, str(name)) if data_root else str(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"配比表里的数据文件不存在: {path}")
        weight = float(weight)
        if weight > 0:
            items.append((path, weight))
    assert items, "配比表里没有权重大于 0 的文件"
    return items


def build_parser():
    """Build the processing and validation command-line interface."""
    p = argparse.ArgumentParser(
        description="极简 SFT 数据处理: 对话 jsonl -> 定长 packed Arrow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="tokenizer 目录, 必须和训练用的一致")
    p.add_argument("--validate", metavar="SUMMARY_DIR",
                   help="只做校验: 给定 <output>/summary 目录, 不跑处理")
    p.add_argument("--weights", help="配比表 xlsx (Sheet1, 列: 文件名 / 权重)")
    p.add_argument("--data-root", default="", help="配比表里写相对路径时的前缀目录")
    p.add_argument("--output", default="./work", help="输出目录")
    p.add_argument("--max-len", type=int, default=262144, help="拼接后的序列长度")
    p.add_argument("--shards", type=int, default=64,
                   help="分片数 = Arrow 分片数。调大省内存, 代价是尾桶变多")
    p.add_argument("--tokenize-workers", type=int, default=32, help="阶段1 并发进程数")
    p.add_argument("--pack-workers", type=int, default=8,
                   help="阶段2 并发进程数, 最吃内存, 按内存给")
    p.add_argument("--threshold", type=float, default=0.95, help="桶填到这个比例就封桶")
    p.add_argument("--window", type=int, default=10, help="连续多少条塞不下就封桶")
    p.add_argument("--no-label-shift", action="store_true",
                   help="labels 不左移 (Megatron 系框架用; 默认左移, MindSpore 用)")
    p.add_argument("--no-hybrid", action="store_true",
                   help="首个 <think>/</think> 也计 loss; 仍要求思考数据格式")
    p.add_argument("--seed", type=int, default=0, help="采样/打散随机种子")
    p.add_argument("--keep-temp", action="store_true", help="保留中间文件, 便于排查")
    return p


def main():
    """Tokenize and pack conversations, or validate an existing dataset."""
    args = build_parser().parse_args()
    label_shift = not args.no_label_shift

    if args.validate:
        validate(args.validate, args.tokenizer, args.max_len, label_shift, preview=True)
        return

    if not args.weights:
        build_parser().error("跑处理流程必须给 --weights (或用 --validate 只做校验)")
    assert args.shards >= 1 and args.max_len > 0

    files = load_weights(args.weights, args.data_root)
    summary_dir = os.path.join(args.output, "summary")
    train_dir = os.path.join(summary_dir, "train")
    tmp_dir = os.path.join(args.output, "tmp")
    for d in (train_dir, tmp_dir):
        os.makedirs(d, exist_ok=True)

    cfg = {"shards": args.shards, "max_len": args.max_len, "tmp_dir": tmp_dir,
           "train_dir": train_dir, "threshold": args.threshold, "window": args.window,
           "label_shift": label_shift, "use_hybrid": not args.no_hybrid,
           "seed": args.seed}

    print("=" * 66)
    print(f"配比表    : {args.weights}  ({len(files)} 个文件)")
    print(f"tokenizer : {args.tokenizer}")
    print(f"序列长度  : {args.max_len}        分片数: {args.shards}")
    print(f"并发      : tokenize={args.tokenize_workers}  pack={args.pack_workers}")
    print(f"标签策略  : label_shift={label_shift} (True=MindSpore)  "
          f"hybrid={not args.no_hybrid}")
    print(f"输出      : {summary_dir}")
    print("=" * 66)

    # ---------- 阶段1 ----------
    _init_worker(args.tokenizer)          # 父进程也加载一次, 用来打印特殊 token
    print(f"\n特殊 token: {_WORKER['special'].describe()}")
    print(f"\n阶段1: tokenize {len(files)} 个文件 (流式, 并发 {args.tokenize_workers})")
    tasks = [(path, weight, i, cfg) for i, (path, weight) in enumerate(files)]
    with Pool(args.tokenize_workers, initializer=_init_worker,
              initargs=(args.tokenizer,)) as pool:
        stats = []
        for i, st in enumerate(pool.imap_unordered(tokenize_file, tasks), 1):
            stats.append(st)
            print(f"  [{i}/{len(files)}] {st['file']}: 读 {st['read']} 条 -> "
                  f"留 {st['kept']} 条"
                  + (f", 坏 {st['bad']}" if st["bad"] else "")
                  + (f", 超长丢弃 {st['toolong']}" if st["toolong"] else ""), flush=True)

    total_read = sum(s["read"] for s in stats)
    total_kept = sum(s["kept"] for s in stats)
    total_long = sum(s["toolong"] for s in stats)
    print(f"阶段1 完成: 读 {total_read} 条, 采样后 {total_kept} 条"
          + (f", 单条超过 max_len 丢弃 {total_long} 条" if total_long else ""))
    if total_kept == 0:
        raise RuntimeError("采样后一条样本都不剩, 检查配比表权重")

    # ---------- 阶段2 ----------
    print(f"\n阶段2: 装桶 + 写 Arrow, {args.shards} 个分片 (并发 {args.pack_workers})")
    with Pool(args.pack_workers, initializer=_init_worker,
              initargs=(args.tokenizer,)) as pool:
        results = list(pool.imap_unordered(
            pack_shard, [(j, cfg) for j in range(args.shards)]))

    results = [r for r in results if r["rows"] > 0]
    if not results:
        raise RuntimeError("没有装出任何桶, 检查数据和 max_len")
    rows = sum(r["rows"] for r in results)
    tokens = sum(r["tokens"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    print(f"阶段2 完成: {rows} 行 (packed 序列), 有效 token {tokens / 1e9:.3f} B, "
          f"平均填充率 {100 * tokens / (rows * args.max_len):.2f}%")
    if skipped:
        print(f"  注意: {skipped} 个 bot 轮因首 token 不是 <think>/</think> 未计 loss")

    n_arrow = write_metadata(summary_dir, train_dir, results[0]["stage"])
    print(f"元数据合并完成: {n_arrow} 个 Arrow 分片")

    if not args.keep_temp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 66)
    print(f"完成: {summary_dir}")
    print(f"训练侧加载: datasets.load_from_disk('{os.path.abspath(summary_dir)}')")
    print("=" * 66)


if __name__ == "__main__":
    main()
