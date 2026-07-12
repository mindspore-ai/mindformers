# Copyright 2025 Huawei Technologies Co., Ltd
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
"""loss function"""

from mindspore import nn, mint, ops
from mindspore.common import dtype as mstype
from mindspore.common._grad_function import _Function
from mindspore.ops.function import comm_func
from mindspore import log as logger

from mindformers.pynative.dtensor_compat import inplace_copy
from mindformers.tools.logger import _LogActionOnce


def _tp_all_reduce(tensor, op, group):
    """Plain (non-autograd) all-reduce over a mesh dimension's process group.

    Used inside ``_VocabParallelCrossEntropy`` whose manual backward already supplies
    the gradient, so the collective must not register its own bprop.
    """
    output, _ = comm_func.all_reduce(tensor.contiguous(), op, group)
    return output


def _vocab_parallel_ce_terms(local_logits, target, group, vocab_start, vocab_per_rank):
    """Vocab-parallel cross entropy for one block of ``[N, V_local]`` float32 logits.

    ``local_logits`` is this rank's vocab slice; ``target`` holds global vocab indices.
    The max / sum-exp / target logit are reduced across the TP ``group`` so the full-vocab
    logits are never materialised. Returns the per-token loss plus the terms the backward
    needs (the softmax is rebuilt from ``exp_vals / global_sum_exp``).
    """
    # Global max over the full vocab (max-reduce the per-rank local maxima).
    local_max = mint.max(local_logits, -1, True)[0]
    global_max = _tp_all_reduce(local_max, 'max', group)
    exp_vals = mint.exp(mint.sub(local_logits, global_max))
    global_sum_exp = _tp_all_reduce(mint.sum(exp_vals, -1, True), 'sum', group)
    log_z = mint.log(global_sum_exp)

    # Target logit: each rank contributes its slice; out-of-range targets are zeroed and
    # all-reduce-summed so the owning rank's value survives.
    in_range = mint.logical_and(mint.ge(target, vocab_start),
                                mint.lt(target, vocab_start + vocab_per_rank))
    in_range_f = ops.cast(in_range, mstype.float32)
    local_target = mint.reshape(
        mint.mul(mint.sub(target, vocab_start), ops.cast(in_range, target.dtype)), (-1, 1))
    target_logit = mint.mul(mint.gather(local_logits, 1, local_target),
                            mint.reshape(in_range_f, (-1, 1)))
    target_logit = _tp_all_reduce(target_logit, 'sum', group)

    # CE = logsumexp(z) - z_target = (global_max + log_z) - z_target.
    loss = mint.sub(mint.reshape(mint.add(global_max, log_z), (-1,)),
                    mint.reshape(target_logit, (-1,)))
    return loss, exp_vals, global_sum_exp, local_target, in_range_f


def _vocab_parallel_ce_grad(exp_vals, global_sum_exp, local_target, in_range_f, upstream, tp_size):
    """Gradient w.r.t. the local logits: ``(softmax_local - onehot) * upstream``, scaled
    by ``tp_size``.

    The trainer pre-scales the loss gradient by sense = 1/(dp*tp*cp) (see
    distributed/utils.py). In the replicated-logits path that 1/tp is undone by the
    output_layer all-gather's reduce-scatter (SUM over tp) in backward; the vocab-sharded
    logits are never gathered, so we restore the missing tp factor here instead -- without
    it the whole model's gradients come out 1/tp_size too small.
    """
    softmax = mint.div(exp_vals, global_sum_exp)
    grad = mint.scatter_add(softmax, 1, local_target, mint.neg(mint.reshape(in_range_f, (-1, 1))))
    grad = mint.mul(grad, mint.unsqueeze(upstream, -1))
    return mint.mul(grad, tp_size)


class _VocabParallelCrossEntropy(_Function):
    """Cross entropy over 2D ``[N, V_local]`` local logits holding this rank's vocab slice.

    TP group, rank, and size are cached on the loss module instead of read from a DTensor.
    The max / sum-exp / target logit are reduced so the
    full-vocab logits are never materialised; the gradient stays a plain local tensor.
    """

    @staticmethod
    def forward(ctx, logits, labels, group, tp_rank, tp_size, compensate_tp=True):
        """Forward pass for vocab-parallel cross entropy on local logits."""
        local_logits = logits
        ctx.orig_dtype = local_logits.dtype
        ctx.tp_grad_factor = tp_size if compensate_tp else 1

        local_logits = ops.cast(local_logits, mstype.float32)
        local_logits = mint.reshape(local_logits, (-1, local_logits.shape[-1]))
        vocab_per_rank = local_logits.shape[-1]
        vocab_start = tp_rank * vocab_per_rank
        target = mint.reshape(labels, (-1,))

        loss, exp_vals, global_sum_exp, local_target, in_range_f = _vocab_parallel_ce_terms(
            local_logits, target, group, vocab_start, vocab_per_rank)

        ctx.exp_vals = exp_vals
        ctx.global_sum_exp = global_sum_exp
        ctx.local_target = local_target
        ctx.in_range_f = in_range_f
        ctx.labels = labels
        return loss

    @staticmethod
    def backward(ctx, grads):
        """Backward pass: grad = (softmax - onehot) * upstream, scaled by tp_grad_factor."""
        grad = _vocab_parallel_ce_grad(ctx.exp_vals, ctx.global_sum_exp, ctx.local_target,
                                       ctx.in_range_f, grads, ctx.tp_grad_factor)
        grad = ops.cast(grad, ctx.orig_dtype)
        return grad, mint.zeros_like(ctx.labels), None, None, None, None


class _LogSoftmax(_Function):
    """
    Custom LogSoftmax function with manual backward implementation.
    Computes log(softmax(logits)) in a numerically stable way.
    """

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass for LogSoftmax.
        """
        ctx.logits = logits
        logits = ops.cast(logits, mstype.float32)
        max_val, _ = mint.max(logits, 1, True)
        shifted = mint.sub(logits, max_val)
        exp_vals = mint.exp(shifted)
        sum_exp = mint.sum(exp_vals, -1, True)
        log_sum = mint.log(sum_exp)
        log_softmax = mint.sub(log_sum, shifted)
        return log_softmax

    @staticmethod
    def backward(ctx, grads):
        """
        Backward pass for LogSoftmax.
        """
        logits = ctx.logits
        return ops.cast(grads, logits.dtype)


class _NLLLoss(_Function):
    """
    Custom Negative Log Likelihood Loss function with manual backward implementation.
    Computes the negative log likelihood loss for classification tasks.
    """

    @staticmethod
    def forward(ctx, log_softmax, labels):
        """
        Forward pass for NLLLoss.
        """
        ctx.log_softmax = log_softmax
        ctx.labels = labels
        indices = mint.reshape(labels, (-1, 1))
        loss = mint.reshape(mint.gather(log_softmax, 1, indices), (-1,))
        return loss

    @staticmethod
    def backward(ctx, grads):
        """
        Backward pass for NLLLoss.
        """
        log_softmax = ctx.log_softmax
        labels = ctx.labels
        indices = mint.reshape(labels, (-1, 1))
        probs = mint.exp(mint.neg(log_softmax))
        vals = mint.zeros_like(indices, dtype=probs.dtype) - 1
        grad = mint.scatter_add(probs, 1, indices, vals)
        grad = grad * mint.unsqueeze(grads, -1)
        return grad, mint.zeros_like(labels)


class _LogSoftmaxModule(nn.Cell):
    """
    Module wrapper for the custom LogSoftmax function.
    """

    @staticmethod
    def construct(logits):
        """
        Forward pass for LogSoftmax.
        """
        return _LogSoftmax.apply(logits)


class _NLLLossModule(nn.Cell):
    """
    Module wrapper for the custom NLLLoss function.
    """

    @staticmethod
    def construct(log_softmax, label):
        """
        Forward pass for _NLLLoss.
        """
        return _NLLLoss.apply(log_softmax, label)


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss.

    CrossEntropyLoss supports two different types of targets:

    - Class indices (int), where the range of values is :math:`[0, C)` with :math:`C` being the number of classes.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
          l_n = - w_{y_n} \\log \\frac{\\exp(x_{n,y_n})}{\\sum_{c=1}^C \\exp(x_{n,c})}
          \\cdot \\mathbb{1}\\{y_n \\not= \\text{ignore_index}\\}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \\ell(x, y) = \\begin{cases}
              \\sum_{n=1}^N \\frac{1}{\\sum_{n=1}^N w_{y_n} \\cdot \\mathbb{1}\\{y_n \\not= \\text{ignore_index}\\}}
              l_n, &
              \\text{if reduction} = \\text{'mean',}\\\\
              \\sum_{n=1}^N l_n,  &
              \\text{if reduction} = \\text{'sum'.}
              \\end{cases}

    - Class probabilities (float), used when the target is a probability distribution over multiple class labels.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
          l_n = - \\sum_{c=1}^C w_c \\log \\frac{\\exp(x_{n,c})}{\\sum_{i=1}^C \\exp(x_{n,i})} y_{n,c}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \\ell(x, y) = \\begin{cases}
              \\frac{\\sum_{n=1}^N l_n}{N}, &
              \\text{if reduction} = \\text{'mean',}\\
              \\sum_{n=1}^N l_n,  &
              \\text{if reduction} = \\text{'sum'.}
              \\end{cases}

    Args:
        calculate_per_token_loss (bool): Whether to calculate the loss of each token. Default: ``False``.
        loss_tag (str, optional): Distinguish different types of loss. Default: 'lm'.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        Tensor, the computed cross entropy loss value.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.pynative.loss import CrossEntropyLoss
        >>> loss = CrossEntropyLoss()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    @_LogActionOnce(m_logger=logger, key='CrossEntropyLoss')
    def __init__(self, calculate_per_token_loss=False, loss_tag='lm',
                 compensate_loss_sense_tp=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        # The trainer's non-PP loss sense is 1/(dp*tp*cp), so the vocab-parallel gradient must
        # multiply tp back (see _vocab_parallel_ce_grad). Under PP the last-stage loss_scale
        # already carries that tp factor, so the compensation must be turned off there.
        self.compensate_loss_sense_tp = compensate_loss_sense_tp
        self.loss_tag = loss_tag
        self.sum = mint.sum
        self.mul = mint.mul
        self.add = mint.add
        self.div = mint.div
        self.relu = mint.nn.functional.relu
        self.reshape = mint.reshape
        self.cast = ops.cast
        self.tuple_to_array = ops.tuple_to_array

        self.log_softmax = _LogSoftmaxModule()
        self.nll_loss = _NLLLossModule()

        self.calculate_per_token_loss = calculate_per_token_loss
        self._tp_group = None
        self._tp_rank = 0
        self._tp_size = 1

    def enable_vocab_parallel(self, group, rank, size):
        """Enable vocab-parallel cross entropy for plain local logits."""
        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = size

    def construct(self, logits, label, input_mask=None):
        """Forward process"""
        if self._tp_group is not None:
            # Logits are the local vocab slice (TP > 1): cross entropy with a vocab-parallel
            # reduction over the TP group, no logits all-gather.
            loss_reduce = _VocabParallelCrossEntropy.apply(
                logits, label, self._tp_group, self._tp_rank, self._tp_size,
                self.compensate_loss_sense_tp)
        else:
            log_softmax = self.log_softmax(logits)
            loss_reduce = self.nll_loss(log_softmax, label)

        # Using input_mask to mask the loss
        if input_mask is None:
            return loss_reduce

        input_mask = self.reshape(input_mask, (-1,))
        input_mask = self.cast(input_mask, mstype.float32)
        numerator = self.sum(self.mul(loss_reduce, input_mask))
        denominator = self.add(
            self.sum(input_mask),
            self.cast(self.tuple_to_array((1e-8,)),mstype.float32))
        if not self.calculate_per_token_loss:
            return self.div(numerator, denominator)
        return numerator, denominator


class ChunkCrossEntropyLoss(CrossEntropyLoss):
    """Chunked cross entropy loss for 3D ``[batch, seq, vocab]`` logits."""

    def __init__(self, calculate_per_token_loss=False, loss_tag='lm',
                 chunk_loss_num=1, **kwargs):
        super().__init__(calculate_per_token_loss=calculate_per_token_loss, loss_tag=loss_tag, **kwargs)
        self.chunk_loss_num = chunk_loss_num

    def construct(self, logits, label, input_mask=None):
        """Forward process."""
        if logits.ndim == 3:
            if input_mask is None:
                input_mask = mint.ones_like(label)
            if self._tp_group is not None:
                # local vocab slice (TP > 1): fuse sequence-dim chunking with the
                # vocab-parallel reduction over the TP group.
                return _ChunkVocabParallelCrossEntropy.apply(
                    logits, label, input_mask, self.chunk_loss_num,
                    self._tp_group, self._tp_rank, self._tp_size, self.compensate_loss_sense_tp)
            return _ChunkCrossEntropyLoss.apply(logits, label, input_mask, self.chunk_loss_num)
        return super().construct(logits, label, input_mask)


class _ChunkCrossEntropyLoss(_Function):
    """
    Chunked cross entropy with a fused backward.

    The forward computes the loss per sequence chunk but only saves the original logits, labels and mask.
    The backward recomputes chunk softmax and concatenates chunk gradients, avoiding autograd slice backward
    materializing multiple full-size logits gradient buffers.
    """

    @staticmethod
    def _chunk_sizes(seq_length, chunk_loss_num):
        active_chunk_num = min(chunk_loss_num, seq_length) if seq_length > 0 else 1
        chunk_size = seq_length // active_chunk_num
        chunk_check = seq_length % active_chunk_num
        if chunk_check == 0:
            return [chunk_size for _ in range(active_chunk_num)]
        chunk_sizes = [chunk_size for _ in range(active_chunk_num - 1)]
        chunk_sizes.append(seq_length - chunk_size * (active_chunk_num - 1))
        return chunk_sizes

    @staticmethod
    def _neg_log_softmax(logits):
        max_val, _ = mint.max(logits, 1, True)
        shifted = mint.sub(logits, max_val)
        exp_vals = mint.exp(shifted)
        sum_exp = mint.sum(exp_vals, -1, True)
        log_sum = mint.log(sum_exp)
        return mint.sub(log_sum, shifted)

    @staticmethod
    def forward(ctx, logits, labels, input_mask, chunk_loss_num):
        """Forward pass for fused chunked cross entropy."""
        ctx.logits = logits
        ctx.labels = labels
        ctx.input_mask = input_mask
        ctx.logits_dtype = logits.dtype
        ctx.chunk_sizes = _ChunkCrossEntropyLoss._chunk_sizes(labels.shape[1], chunk_loss_num)

        denominator = mint.sum(mint.reshape(input_mask, (-1,)))
        denominator = mint.add(denominator, ops.cast(ops.tuple_to_array((1e-8,)), mstype.float32))
        ctx.denominator = denominator

        loss = None
        start_idx = 0
        for chunk_size in ctx.chunk_sizes:
            end_idx = start_idx + chunk_size
            seq_logits = logits[:, start_idx:end_idx, :]
            seq_logits = ops.cast(mint.reshape(seq_logits, (-1, seq_logits.shape[-1])), mstype.float32)
            seq_labels = mint.reshape(labels[:, start_idx:end_idx], (-1,))
            seq_mask = ops.cast(mint.reshape(input_mask[:, start_idx:end_idx], (-1,)), mstype.float32)

            neg_log_softmax = _ChunkCrossEntropyLoss._neg_log_softmax(seq_logits)
            indices = mint.reshape(seq_labels, (-1, 1))
            loss_reduce = mint.reshape(mint.gather(neg_log_softmax, 1, indices), (-1,))
            numerator = mint.sum(mint.mul(loss_reduce, seq_mask))
            seq_loss = mint.div(numerator, denominator)
            loss = seq_loss if loss is None else loss + seq_loss
            start_idx = end_idx
        return loss

    @staticmethod
    def backward(ctx, grads):
        """Backward pass for fused chunked cross entropy."""
        logits = ctx.logits
        labels = ctx.labels
        input_mask = ctx.input_mask
        grad_logits_chunks = []
        start_idx = 0
        for chunk_size in ctx.chunk_sizes:
            end_idx = start_idx + chunk_size
            seq_logits = logits[:, start_idx:end_idx, :]
            seq_shape = seq_logits.shape
            seq_logits = ops.cast(mint.reshape(seq_logits, (-1, seq_logits.shape[-1])), mstype.float32)
            seq_labels = mint.reshape(labels[:, start_idx:end_idx], (-1,))
            seq_mask = ops.cast(mint.reshape(input_mask[:, start_idx:end_idx], (-1,)), mstype.float32)

            max_val, _ = mint.max(seq_logits, 1, True)
            shifted = mint.sub(seq_logits, max_val)
            exp_vals = mint.exp(shifted)
            probs = mint.div(exp_vals, mint.sum(exp_vals, -1, True))
            indices = mint.reshape(seq_labels, (-1, 1))
            vals = mint.zeros_like(indices, dtype=probs.dtype) - 1
            grad = mint.scatter_add(probs, 1, indices, vals)
            grad_scale = mint.div(mint.mul(seq_mask, grads), ctx.denominator)
            grad = mint.mul(grad, mint.unsqueeze(grad_scale, -1))
            grad = ops.cast(mint.reshape(grad, seq_shape), ctx.logits_dtype)
            grad_logits_chunks.append(grad)
            start_idx = end_idx
        grad_logits = mint.cat(grad_logits_chunks, dim=1)
        return grad_logits, mint.zeros_like(labels), mint.zeros_like(input_mask), None


class _ChunkVocabParallelCrossEntropy(_Function):
    """Chunked cross entropy for 3D ``[batch, seq, V_local]`` logits sharded on the vocab
    dimension across the TP mesh (enable_loss_parallel together with chunk_loss_num > 1).

    Combines two memory savings: sequence-dim chunking (one chunk's softmax is materialised
    at a time, the fused backward recomputes per chunk) and the vocab-parallel reduction
    (max / sum-exp / target logit all-reduced over the TP group, full-vocab logits never
    gathered). Forward saves only the local logits slice, labels and mask; the backward
    recomputes each chunk's softmax (re-running the TP all-reduce) and concatenates the
    chunk gradients along the sequence dimension.
    """

    @staticmethod
    def forward(ctx, logits, labels, input_mask, chunk_loss_num, group, tp_rank, tp_size,
                compensate_tp=True):
        """Forward pass for fused chunked vocab-parallel cross entropy on local logits."""
        local_logits = logits  # [b, s, V_local]
        ctx.local_logits = local_logits
        ctx.labels = labels
        ctx.input_mask = input_mask
        ctx.logits_dtype = local_logits.dtype
        ctx.group = group
        ctx.tp_grad_factor = tp_size if compensate_tp else 1
        ctx.vocab_per_rank = local_logits.shape[-1]
        ctx.vocab_start = tp_rank * local_logits.shape[-1]
        ctx.chunk_sizes = _ChunkCrossEntropyLoss._chunk_sizes(labels.shape[1], chunk_loss_num)

        denominator = mint.sum(mint.reshape(input_mask, (-1,)))
        denominator = mint.add(denominator, ops.cast(ops.tuple_to_array((1e-8,)), mstype.float32))
        ctx.denominator = denominator

        loss = None
        start_idx = 0
        for chunk_size in ctx.chunk_sizes:
            end_idx = start_idx + chunk_size
            seq_logits = ops.cast(
                mint.reshape(local_logits[:, start_idx:end_idx, :], (-1, ctx.vocab_per_rank)),
                mstype.float32)
            seq_labels = mint.reshape(labels[:, start_idx:end_idx], (-1,))
            seq_mask = ops.cast(mint.reshape(input_mask[:, start_idx:end_idx], (-1,)), mstype.float32)

            loss_reduce = _vocab_parallel_ce_terms(
                seq_logits, seq_labels, group, ctx.vocab_start, ctx.vocab_per_rank)[0]
            numerator = mint.sum(mint.mul(loss_reduce, seq_mask))
            seq_loss = mint.div(numerator, denominator)
            loss = seq_loss if loss is None else loss + seq_loss
            start_idx = end_idx
        return loss

    @staticmethod
    def backward(ctx, grads):
        """Backward pass: recompute each chunk into one preallocated gradient buffer."""
        local_logits = ctx.local_logits
        labels = ctx.labels
        input_mask = ctx.input_mask
        grad_logits = mint.empty_like(local_logits)
        start_idx = 0
        for chunk_size in ctx.chunk_sizes:
            end_idx = start_idx + chunk_size
            seq_slice = local_logits[:, start_idx:end_idx, :]
            seq_shape = seq_slice.shape
            seq_logits = ops.cast(mint.reshape(seq_slice, (-1, ctx.vocab_per_rank)), mstype.float32)
            seq_labels = mint.reshape(labels[:, start_idx:end_idx], (-1,))
            seq_mask = ops.cast(mint.reshape(input_mask[:, start_idx:end_idx], (-1,)), mstype.float32)

            _, exp_vals, global_sum_exp, local_target, in_range_f = _vocab_parallel_ce_terms(
                seq_logits, seq_labels, ctx.group, ctx.vocab_start, ctx.vocab_per_rank)
            upstream = mint.div(mint.mul(seq_mask, grads), ctx.denominator)
            grad = _vocab_parallel_ce_grad(exp_vals, global_sum_exp, local_target, in_range_f,
                                           upstream, ctx.tp_grad_factor)
            grad = ops.cast(mint.reshape(grad, seq_shape), ctx.logits_dtype)
            inplace_copy(grad_logits[:, start_idx:end_idx, :], grad)
            start_idx = end_idx
        return (grad_logits, mint.zeros_like(labels), mint.zeros_like(input_mask),
                None, None, None, None, None)
