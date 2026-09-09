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
"""Greedy text generation for the PyNative Trainer.

Prompts are plain text; chat-format rendering is out of scope (message
templates are applied in offline preprocessing, per repo convention).
"""

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from mindspore import Tensor
from mindspore.graph.api import _no_grad
from mindspore.mint.distributed import all_gather_object, broadcast_object_list

from mindformers.models.build_tokenizer import build_tokenizer as _graph_build_tokenizer
from mindformers.pynative.distributed.style import _all_gather_dim
from mindformers.tools.logger import logger
from mindformers.tools.utils import FILE_PERMISSION


@dataclass
class PipelineInferenceContext:
    """State the decode loop needs to drive a pipeline-parallel forward.

    Every rank calls ``schedule.run`` with the same kwargs; only the last
    stage receives logits and broadcasts the per-batch token ids over its
    own PP line's group. ``src`` is that stage's GLOBAL rank: MindSpore's
    broadcast resolves ``src`` against the world group even when a sub
    group is given.
    """

    schedule: Any
    is_last_stage: bool
    group: Any
    src: int

    def broadcast_tokens(self, token_ids):
        """Broadcast the last stage's per-sample token ids over the PP group."""
        buf = list(token_ids)
        broadcast_object_list(buf, group=self.group, src=self.src)
        return [int(token_id) for token_id in buf]


def configure_inference(config):
    """Remove training-only branches before the model is instantiated.

    Drops the MTP layers (and their trailing ``compress_ratios`` entries).
    """
    num_layers = config.model.num_hidden_layers
    compress_ratios = getattr(config.model, "compress_ratios", None)
    if compress_ratios is not None:
        config.model.compress_ratios = list(compress_ratios[:num_layers])
    config.model.num_nextn_predict_layers = 0


_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "tokenizer.model", "spiece.model",
)


def build_inference_tokenizer(config):
    """Load the tokenizer from the checkpoint directory and record the pad
    token id on the model config.

    Weights and tokenizer share the single ``checkpoint.load_path`` knob.
    HF checkpoints colocate both; MindFormers ``iteration_*`` checkpoints
    store weights only, so the tokenizer files must be present in the
    checkpoint directory as well (checked here with a clear error).
    """
    tokenizer_dir = config.checkpoint.load_path
    if not tokenizer_dir:
        raise ValueError(
            "Inference requires checkpoint.load_path pointing at the weights "
            "directory; the tokenizer is loaded from the same path."
        )
    if not any(
        os.path.isfile(os.path.join(tokenizer_dir, name))
        for name in _TOKENIZER_FILES
    ):
        raise ValueError(
            f"No tokenizer files found in the checkpoint directory "
            f"{tokenizer_dir!r}. MindFormers checkpoints store weights only; "
            f"copy the tokenizer files (e.g. tokenizer.json, "
            f"tokenizer_config.json) into that directory before inference."
        )
    tokenizer = _graph_build_tokenizer(
        use_legacy=False, pretrained_model_dir=tokenizer_dir
    )
    config.model.pad_token_id = _resolve_pad_token_id(tokenizer)
    return tokenizer


def _resolve_pad_token_id(tokenizer):
    """Return the pad token id, falling back to eos, raising when neither exists."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id or eos_token_id.")
    return pad_token_id


def right_pad_token_ids(token_ids, pad_token_id, sequence_divisor):
    """Right-pad logical token IDs for tensor/context-parallel sequence slicing."""
    padded_length = math.ceil(len(token_ids) / sequence_divisor) * sequence_divisor
    return token_ids + [pad_token_id] * (padded_length - len(token_ids))


def tokenize_input(tokenizer, text, sequence_divisor):
    """Tokenize one string and return logical and parallel-padded token IDs."""
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("The tokenizer produced an empty token sequence.")

    pad_token_id = _resolve_pad_token_id(tokenizer)
    padded_ids = right_pad_token_ids(token_ids, pad_token_id, sequence_divisor)
    return token_ids, padded_ids, pad_token_id


def iter_jsonl_records(path):
    """Yield successive non-empty JSON records from a jsonl file.

    Raises ``ValueError`` with the 1-based line number on a malformed line.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: expected a JSON object per line, got {e.msg}."
                ) from e
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_no}: expected a JSON object, got {type(record).__name__}."
                )
            yield record


def iter_text_records(path):
    """Yield one ``{"instruction": line}`` record per non-blank line."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield {"instruction": stripped}


def normalize_input(input_data):
    """Coerce any accepted input shape into an iterable of records.

    ``str``: literal prompt, or an existing file (``.jsonl`` streams one
    JSON object per line, any other suffix is plain text); ``dict``: single
    record; ``list[dict]`` / ``list[str]``: batch. Mixed-type lists and
    other types raise ``TypeError``.
    """
    if input_data is None:
        return []
    if isinstance(input_data, str):
        path = input_data
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() == ".jsonl":
                return iter_jsonl_records(path)
            return iter_text_records(path)
        return [{"instruction": path}]
    if isinstance(input_data, dict):
        return [input_data]
    if isinstance(input_data, list):
        if all(isinstance(item, dict) for item in input_data):
            return list(input_data)
        if all(isinstance(item, str) for item in input_data):
            return [{"instruction": item} for item in input_data]
        raise TypeError(
            "input_data list must be uniformly list[dict] or list[str]; "
            f"got mixed types ({[type(i).__name__ for i in input_data]})."
        )
    raise TypeError(
        f"input_data must be str, dict, or list; got {type(input_data).__name__}."
    )


def build_dataset_prompt(record, instruction_field):
    """Render one record into a prompt string.

    ``record[instruction_field]`` (with optional ``input`` appended) forms
    the prompt; raises ``ValueError`` when both are empty.
    """
    instruction = record.get(instruction_field, "") if isinstance(record, dict) else ""
    extra_input = record.get("input", "") if isinstance(record, dict) else ""
    if extra_input:
        instruction = f"{instruction}\n{extra_input}" if instruction else extra_input
    if not instruction:
        raise ValueError(
            f"jsonl record has no usable prompt (missing "
            f"{instruction_field!r}/'input'): {record!r}."
        )
    return instruction


def gather_vocab_parallel_logits(logits, parallel_dims):
    """Gather local LM-head vocabulary shards over the tensor-parallel group."""
    if hasattr(logits, "to_local"):
        logits = logits.to_local()
    if parallel_dims is None or not parallel_dims.tp_enabled:
        return logits

    return _all_gather_dim(logits, -1, parallel_dims.get_mesh("tp").get_group())


def select_last_logits_row(logits, last_index, parallel_dims):
    """Select the global ``last_index`` logits row under context parallel.

    CP shards logits along the sequence dimension, so a global row index
    is only valid on its owner rank. Every rank slices its in-chunk offset
    (always in bounds), the row is all-gathered over the CP group, and the
    owner's contribution is selected (the others' rows are never read);
    the collective stays symmetric. Without CP the row is sliced globally.
    """
    if parallel_dims is None or not parallel_dims.cp_enabled:
        return logits[last_index:last_index + 1]

    if hasattr(logits, "to_local"):
        logits = logits.to_local()
    cp_group = parallel_dims.get_mesh("cp").get_group()
    chunk = logits.shape[0]
    owner = last_index // chunk
    local_offset = last_index % chunk
    row = logits[local_offset:local_offset + 1]
    gathered = _all_gather_dim(row, 0, cp_group)
    return gathered[owner:owner + 1]


def generate_batch(model, tokenizer, prompts, pad_token_id, sequence_divisor,
                   parallel_dims, max_new_tokens, max_seq_length, rank,
                   pp_ctx: Optional[PipelineInferenceContext] = None):
    """Batched greedy generation with per-sample eos freezing.

    All ranks of one decode unit (TP/CP siblings, a pure-FSDP dp group, or
    the whole PP world) pass the same ``prompts`` and execute this loop
    collectively, so every rank issues the same number of forwards by
    construction. Each step forwards one ``[bs, L]`` batch; the model
    returns ``[bs * L, vocab]`` logits (sequence-sharded under CP,
    sample-major). A sample that hits eos (or overflows ``max_seq_length``)
    freezes: it keeps its slot as a pad-filled row whose outputs are
    discarded. The finished state derives from the same logits on every
    rank, so the break is collective-safe.
    """
    batch_size = len(prompts)
    generated_ids = [list(prompt) for prompt in prompts]
    new_token_ids = [[] for _ in prompts]
    eos_token_id = tokenizer.eos_token_id
    eos_token_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    finished = [False] * batch_size

    for _ in range(1, max_new_tokens + 1):
        # Freeze samples whose padded length no longer fits max_seq_length.
        for i in range(batch_size):
            if not finished[i]:
                padded = right_pad_token_ids(generated_ids[i], pad_token_id, sequence_divisor)
                if len(padded) > max_seq_length:
                    finished[i] = True
        if all(finished):
            break

        # Rebuild padding every step: appending after a previous padded
        # sequence would give the generated token an incorrect position ID.
        # The batch length is driven by the active rows only; frozen rows
        # are pad-filled (their outputs are discarded).
        active_rows = {
            i: right_pad_token_ids(generated_ids[i], pad_token_id, sequence_divisor)
            for i in range(batch_size) if not finished[i]
        }
        batch_len = max(len(row) for row in active_rows.values())
        input_array = np.full((batch_size, batch_len), pad_token_id, dtype=np.int32)
        for i, row in active_rows.items():
            input_array[i, :len(row)] = row
        input_ids = Tensor(input_array)
        position_ids = Tensor(np.tile(
            np.arange(batch_len, dtype=np.int32), (batch_size, 1)
        ))

        with _no_grad():
            if pp_ctx is None:
                logits = model[0](input_ids=input_ids, position_ids=position_ids)
            else:
                # The schedule consumes the inputs only on the first stage;
                # ``run`` returns a one-element list of last-stage outputs.
                logits = pp_ctx.schedule.run(
                    input_ids=input_ids, position_ids=position_ids
                )
                if logits:
                    logits = logits[0]

        next_token_ids = [0] * batch_size
        if pp_ctx is None or pp_ctx.is_last_stage:
            if hasattr(logits, "to_local"):
                logits = logits.to_local()
            # The model flattens logits to [bs * seq, vocab] (seq = the
            # local chunk under CP); sample i owns rows
            # [i * rows_per_sample, (i + 1) * rows_per_sample).
            rows_per_sample = logits.shape[0] // batch_size
            for i in range(batch_size):
                if finished[i]:
                    # Frozen: every rank skips the same samples, so the
                    # per-sample collectives stay paired.
                    continue
                # Slice the last logical row before the TP all-gather so
                # per-step communication is O(vocab) per sample.
                sample_logits = logits[i * rows_per_sample:(i + 1) * rows_per_sample]
                last_row = select_last_logits_row(
                    sample_logits, len(generated_ids[i]) - 1, parallel_dims
                )
                last_row = gather_vocab_parallel_logits(last_row, parallel_dims)
                next_token_ids[i] = int(np.argmax(last_row[0].asnumpy()))
        if pp_ctx is not None:
            next_token_ids = pp_ctx.broadcast_tokens(next_token_ids)

        for i in range(batch_size):
            if finished[i]:
                continue
            token_id = next_token_ids[i]
            generated_ids[i].append(token_id)
            new_token_ids[i].append(token_id)
            if token_id in eos_token_ids:
                finished[i] = True
        if all(finished):
            break

    if rank == 0:
        for i, tokens in enumerate(new_token_ids):
            generated_text = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"[generate] sample {i} generated_token_ids: {tokens}", flush=True)
            print(f"[generate] sample {i} generated_text: {generated_text!r}", flush=True)
    return generated_ids


def _resolve_dataset_meta(record, instruction_field):
    """Echo the original instruction/input pair so outputs join back to inputs."""
    echo = {instruction_field: record.get(instruction_field, "")}
    if "input" in record:
        echo["input"] = record["input"]
    return echo


def _resolve_output_format(output_path):
    """Map an ``output_path`` to a writer strategy by file extension.

    Returns ``"jsonl"``, ``"json"``, ``"txt"``, or
    ``None`` when unset. Raises ``ValueError`` for unknown extensions.
    """
    if output_path is None:
        return None
    lower = output_path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".txt") or Path(output_path).suffix == "":
        return "txt"
    raise ValueError(
        f"Unsupported inference output extension {Path(output_path).suffix!r}. Supported: "
        ".jsonl, .json, .txt."
    )


def _open_output_file(path):
    """Open a text output file with explicit ``FILE_PERMISSION`` (0o640)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    return os.fdopen(os.open(path, flags, FILE_PERMISSION), "w", encoding="utf-8")


def _open_jsonl_writer(path, rank):
    """Open a jsonl file for streaming writes. Rank 0 only; other ranks get None."""
    if rank != 0:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return _open_output_file(path)


def _record_result(fmt, writer, collected, payload, text_line):
    """Persist one sample result per the chosen output format (rank 0 only)."""
    if fmt == "jsonl":
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        writer.flush()
    elif fmt == "json":
        collected.append(payload)
    elif fmt == "txt":
        collected.append(text_line)


def _sample_shard_info(parallel_dims, pp_ctx):
    """Sample-level sharding across independent model replicas.

    Each ``dp_replicate`` coordinate owns a full model copy whose forward
    shares no collectives with the other copies, so replicas may decode
    different stripes of records at their own pace. Returns
    ``(shard_id, num_shards)``; ``(0, 1)`` disables sharding (single
    card, pure FSDP, or PP -- every PP line decodes the full stream).
    """
    if pp_ctx is not None or parallel_dims is None:
        return 0, 1
    if parallel_dims.dp_replicate <= 1:
        return 0, 1
    mesh = parallel_dims.get_optional_mesh("dp_replicate")
    if mesh is None:
        return 0, 1
    coord = mesh.get_coordinate()
    shard_id = int(coord[0]) if coord else 0
    return shard_id, int(mesh.size())


def _merge_shard_entries(shard_entries):
    """Merge per-rank ``(payload, text_line)`` entries into one ordered list.

    tp/cp siblings of a replica decode the same records and report
    identical entries; duplicates collapse by ``sample_idx``. The result is
    ordered by ``sample_idx`` so the output file preserves the input order.
    """
    merged = {}
    for entries in shard_entries:
        for payload, text_line in entries:
            merged.setdefault(payload["sample_idx"], (payload, text_line))
    return [merged[idx] for idx in sorted(merged)]


def generate_from_records(model, tokenizer, records, instruction_field,
                          sequence_divisor, parallel_dims,
                          max_new_tokens, batch_size,
                          max_seq_length, rank, output_path=None,
                          pp_ctx: Optional[PipelineInferenceContext] = None):
    """Run batched generation over a sequence of records and stream results.

    Records are tokenized and decoded in ``batch_size`` batches by
    ``generate_batch``. All ranks of a decode unit share the same batch
    stream; with ``dp_replicate > 1`` each replica decodes its own stripe
    (see ``_sample_shard_info``) and results are all-gathered at the end
    and written in input order. A record that fails at tokenize time is
    reported per sample and skipped; a forward failure propagates.
    Returns the ``(sample_idx, generated_ids)`` list.
    """
    fmt = _resolve_output_format(output_path)
    output_path = os.path.realpath(output_path) if output_path else None
    writer = _open_jsonl_writer(output_path, rank) if fmt == "jsonl" else None
    sample_results = []
    collected = []  # used by json / txt -- flushed in the finally block.
    shard_id, num_shards = _sample_shard_info(parallel_dims, pp_ctx)
    shard_entries = []  # sharded mode: this rank's (payload, text_line) pairs.
    if num_shards > 1:
        # Striped sharding needs the global sample index, so materialize
        # streams up front; every rank builds the identical list.
        records = list(records)
        if rank == 0:
            logger.info(
                "[generate] DP sample sharding: %s replicas, this rank decodes "
                "every %s-th record starting at index %s",
                num_shards, num_shards, shard_id,
            )

    def _report(payload, text_line):
        if num_shards > 1:
            shard_entries.append((payload, text_line))
        else:
            _record_result(fmt, writer, collected, payload, text_line)

    try:
        # Stage 1: tokenize + preflight every record. A bad record is
        # dropped so the rest of its batch still decodes.
        batch = []  # (sample_idx, record, token_ids, pad_token_id)
        sample_idx = -1
        for record in records:
            sample_idx += 1
            if num_shards > 1 and sample_idx % num_shards != shard_id:
                continue
            try:
                prompt = build_dataset_prompt(record, instruction_field)
                token_ids, padded_ids, pad_token_id = tokenize_input(
                    tokenizer, prompt, sequence_divisor
                )
                if len(padded_ids) > max_seq_length:
                    raise ValueError(
                        f"Padded input length ({len(padded_ids)}) exceeds "
                        f"model.seq_length ({max_seq_length})."
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Log on every rank: a rank-0-only log would hide the
                # original error behind a desync timeout cascade.
                logger.error(
                    "[generate][rank %s] sample_idx=%s failed: %s: %s",
                    rank, sample_idx, type(exc).__name__, exc,
                )
                meta = _resolve_dataset_meta(
                    record, instruction_field
                ) if isinstance(record, dict) else {"raw": record}
                payload = {"sample_idx": sample_idx, "error": str(exc), **meta}
                _report(payload, json.dumps(payload, ensure_ascii=False))
                continue
            batch.append((sample_idx, record, token_ids, pad_token_id))

        # Stage 2: decode per batch. All ranks of the decode unit agree
        # on the batch boundaries, so every forward stays paired.
        for start in range(0, len(batch), batch_size):
            chunk = batch[start:start + batch_size]
            generated = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=[item[2] for item in chunk],
                pad_token_id=chunk[0][3],
                sequence_divisor=sequence_divisor,
                parallel_dims=parallel_dims,
                max_new_tokens=max_new_tokens,
                max_seq_length=max_seq_length,
                rank=rank,
                pp_ctx=pp_ctx,
            )
            for (sample_idx_ok, record_ok, token_ids_ok, _), generated_ids in zip(
                    chunk, generated):
                sample_results.append((sample_idx_ok, generated_ids))
                if rank == 0 or num_shards > 1:
                    generated_text = tokenizer.decode(
                        generated_ids[len(token_ids_ok):], skip_special_tokens=False
                    )
                    meta = _resolve_dataset_meta(
                        record_ok, instruction_field
                    ) if isinstance(record_ok, dict) else {"raw": record_ok}
                    _report({
                        "sample_idx": sample_idx_ok,
                        "generated_ids": generated_ids,
                        "generated_text": generated_text,
                        **meta,
                    }, generated_text)
    finally:
        if num_shards > 1:
            # Every rank contributes its entries (empty for idle shards) so
            # the return value stays identical on all ranks.
            gathered = [None] * parallel_dims.world_size
            all_gather_object(gathered, shard_entries)
            merged = _merge_shard_entries(gathered)
            sample_results = [
                (payload["sample_idx"], payload["generated_ids"])
                for payload, _ in merged
                if "generated_ids" in payload
            ]
            if rank == 0:
                for payload, text_line in merged:
                    _record_result(fmt, writer, collected, payload, text_line)
        if writer is not None:
            writer.close()
        if rank == 0 and fmt in ("json", "txt"):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with _open_output_file(output_path) as f:
                if fmt == "json":
                    json.dump(collected, f, ensure_ascii=False, indent=2)
                else:
                    f.write("\n".join(collected) + ("\n" if collected else ""))
    return sample_results
