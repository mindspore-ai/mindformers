FROM swr.cn-north-4.myhuaweicloud.com/atlas-ci/pr-mindformers:a2-1017

USER root

# 删除基础镜像自带的 CANN 8.5.0，避免与待安装的 9.1.0 冲突
# /usr/local/Ascend/driver 为驱动挂载目录，保留；其余（cann、cann-8.5.0、ascend-toolkit、nnal、atb、opp 等）全部删除
RUN find /usr/local/Ascend -mindepth 1 -maxdepth 1 ! -name driver -exec rm -rf {} + \
    && rm -f /etc/ascend_install.info \
    && (dpkg -l 2>/dev/null | awk '/^ii[[:space:]]+ascend-cann/{print $2}' | xargs -r dpkg --purge --force-all >/dev/null 2>&1 || true)

COPY install_a2_cann.sh /tmp/install_a2_cann.sh

# 通过 install_a2_cann.sh 安装 CANN 9.1.0（toolkit + 910b-ops + nnal）
RUN cd /tmp && bash install_a2_cann.sh && rm -rf /tmp/cann /tmp/install_a2_cann.sh

# 基础镜像中的 CANN 环境变量指向已删除的 8.5.0 目录，这里改为 9.1.0 安装路径
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
ENV ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_OPS_NN_PATH=/usr/local/Ascend/ascend-toolkit/latest/share/info/ops_nn
ENV PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:${PATH}
ENV PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:${LD_LIBRARY_PATH}
