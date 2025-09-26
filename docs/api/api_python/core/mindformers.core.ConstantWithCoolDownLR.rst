mindformers.core.ConstantWithCoolDownLR
=======================================

.. py:class:: mindformers.core.ConstantWithCoolDownLR(learning_rate, warmup_steps=None, warmup_lr_init=0., warmup_ratio=None, keep_steps=0, decay_steps=None, decay_ratio=None, total_steps=None, num_cycles=0.5, lr_end1=0, final_steps=0, lr_end2=None, **kwargs)

    分段式学习率。

    根据论文 `DeepSeek-V3 Technical Report <https://arxiv.org/pdf/2412.19437>`_ 第23页的描述实现。

    分段式学习率首先采用线性预热策略逐渐增加每个参数组的学习率，并在一定步数中保持不变，然后遵循余弦函数逐渐衰减。最后，在维持一定步数后切换至一个新的常数学习率。

    在预热阶段，学习率从一个较小的初始值增加到基准学习率，公式如下：

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    其中， :math:`\eta_{\text{warmup}}` 是初始学习率， :math:`\eta_{\text{base}}` 是预热阶段结束后的学习率。

    在衰减阶段，学习率将按照余弦衰减公式进行衰减：

    .. math::
        \eta_t = \eta_{\text{end}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{end}})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    其中， :math:`T_{cur}` 是自预热阶段结束以来的epoch数量， :math:`T_{max}` 是下次重启前的总epoch数。

    参数：
        - **learning_rate** (float) - 预热阶段结束后的学习率。该学习率会在保持阶段中维持不变。
        - **warmup_steps** (int, 可选) - 预热阶段的步数。默认值： ``None`` 。
        - **warmup_lr_init** (float, 可选) - 预热阶段的初始学习率。默认值： ``0.`` 。
        - **warmup_ratio** (float, 可选) - 预热阶段占总训练步数的比例。默认值： ``None`` 。
        - **keep_steps** (int, 可选) - 保持阶段的步数。默认值： ``0`` 。
        - **decay_steps** (int, 可选) - 衰减步骤的数量。默认值： ``None`` 。
        - **decay_ratio** (float, 可选) - 衰减阶段占总训练步骤的比例。默认值： ``None`` 。
        - **total_steps** (int, 可选) - 总的训练步骤数。默认值： ``None`` 。
        - **num_cycles** (float, 可选) - 余弦调度中的周期数量（默认情况下为半个周期，从最大值递减至 0）。默认值： ``0.5`` 。
        - **lr_end1** (float, 可选) - 衰减阶段结束后的学习率。默认值： ``0.0`` 。
        - **final_steps** (int, 可选) - 学习率保持在 `lr_end1` 的步数。默认值： ``0`` 。
        - **lr_end2** (float, 可选) - 学习率的最终值。设置为 ``None`` 时，与 `lr_end1` 保持一致。默认值： ``None`` 。

    输入：
        - **global_step** (int) - 全局步数。

    输出：
        学习率。
