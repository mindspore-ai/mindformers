# ===================== 构建参数 =====================
ARG CANN_VERSION
ARG CHIP_ARCH
ARG OS_SYSTEM
ARG PY_VERSION

# ===================== CANN 官方基础镜像 =====================

FROM quay.io/ascend/cann:${CANN_VERSION}-${CHIP_ARCH}-${OS_SYSTEM}-${PY_VERSION} AS base

# ===================== 安装阶段 =====================

## set pip source
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
RUN pip3 config set global.index-url ${PIP_INDEX_URL}

##  install pip package for mindspore
RUN pip3 install wheel \
 && pip3 install -U setuptools \
 && pip3 install --no-cache-dir pyyaml \
 && pip3 install  --no-cache-dir "numpy>=1.19.3,<=1.26.4" \
 && pip3 install  --no-cache-dir sympy \
 && pip3 install  --no-cache-dir decorator \
 && pip3 install  --no-cache-dir protobuf \
 && pip3 install  --no-cache-dir attrs

# install mindspore pkg
ARG MINDSPORE_VERSION
RUN pip3 install --no-cache-dir mindspore==${MINDSPORE_VERSION} -i https://repo.mindspore.cn/pypi/simple --extra-index-url ${PIP_INDEX_URL}
ARG HYPER_PARALLEL_VERSION
RUN pip3 install --no-cache-dir hyper-parallel==${HYPER_PARALLEL_VERSION} -i https://repo.mindspore.cn/pypi/simple --extra-index-url ${PIP_INDEX_URL}
ARG MINDFORMERS_VERSION
RUN pip3 install --no-cache-dir mindformers==${MINDFORMERS_VERSION}

# ===================== 健康检查 =====================

## 探活：校验容器内 MindSpore Transformers 可正常导入。
## 实测一次导入约 7s、峰值内存约 800MB，因此把探测间隔放宽到 5 分钟，避免打扰容器内的训练任务；
## 如不需要周期性探活，可在 docker run 时加 --no-healthcheck 关闭。
HEALTHCHECK --interval=5m --timeout=60s --start-period=30s --retries=3 \
 CMD python3 -c "import mindformers" || exit 1
