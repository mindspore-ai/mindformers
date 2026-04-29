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
from mindspore import log as logger

from mindformers.tools.logger import _LogActionOnce


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
    def __init__(self, calculate_per_token_loss=False, loss_tag='lm', **kwargs):
        super().__init__()
        self.kwargs = kwargs
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

    def construct(self, logits, label, input_mask):
        """Forward process"""
        log_softmax = self.log_softmax(logits)
        loss_reduce = self.nll_loss(log_softmax, label)

        # Using input_mask to mask the loss
        input_mask = self.reshape(input_mask, (-1,))
        input_mask = self.cast(input_mask, mstype.float32)
        numerator = self.sum(self.mul(loss_reduce, input_mask))
        denominator = self.add(
            self.sum(input_mask),
            self.cast(self.tuple_to_array((1e-8,)),mstype.float32))
        if not self.calculate_per_token_loss:
            return self.div(numerator, denominator)
        return numerator, denominator
