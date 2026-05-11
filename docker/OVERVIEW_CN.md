# MindSpore
 	 
> [English](./OVERVIEW.md) | 中文

## 快速参考

- MindSpore Transformers 由 [MindSpore Transformers 社区](https://atomgit.com/mindspore/mindformers) 维护

- 从哪里获取帮助

    - [MindSpore Transformers 文档](https://www.mindspore.cn/mindformers/docs/zh-CN/stable/index.html)
    - [MindSpore Transformers 社区](https://www.mindspore.cn/sig/MindSpore%20Transformers)
    - [问题反馈](https://atomgit.com/mindspore/mindformers/issues)

## MindSpore Transformers

MindSpore Transformers套件的目标是构建一个大模型预训练、微调、推理、部署的全流程开发套件，提供业内主流的Transformer类大语言模型（Large Language Models, LLMs）和多模态理解模型（Multimodal Models, MMs）。期望帮助用户轻松地实现大模型全流程开发。

## 支持的 Tags 及 Dockerfile 使用方法

### Tag 规范

Tag 遵循以下格式：

```text
<MindSpore Transformers 版本号>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
```

| 字段                         | 示例值                          | 说明                                        |
|----------------------------|------------------------------|-------------------------------------------|
| MindSpore Transformers 版本号 | 1.8.0                        | 对应 MindSpore Transformers 官方发布 Tag 中的版本标识 |
| 硬件信息（芯片）                   | 910b / a3                    | 昇腾芯片型号标识                                  |
| 操作系统                       | ubuntu22.04 / openeuler24.03 | 基础镜像所使用的操作系统发行版及版本号                       |
| Python 版本                  | py3.11                       | 镜像内置 Python 大版本号                          |

> Tips: 系统架构通过 Docker Manifest 自动识别，无需在 Tag 中指定。

### 镜像仓库地址

MindSpore Transformers Ascend 镜像托管在华为云 SWR 镜像仓库：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers
```

**完整镜像示例：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers:1.8.0-910b-ubuntu22.04-py3.11
```

### 构建参数

| 参数                  | 说明                         | 必填 | 参考来源                         | 示例值                                                    |
|---------------------|----------------------------|----|------------------------------|--------------------------------------------------------|
| CANN_VERSION        | 昇腾 CANN 工具包版本              | 是  | CANN 镜像标签                    | 8.5.0                                                  |
| CHIP_ARCH           | 昇腾芯片架构标识                   | 是  | Tag 规范                       | 910b / a3                                              |
| OS_SYSTEM           | 基础镜像操作系统及版本                | 是  | Tag 规范                       | ubuntu22.04 / openeuler24.03                           |
| PY_VERSION          | 基础镜像内置 Python 版本           | 是  | Tag 规范                       | py3.11                                                 |
| MINDSPORE_VERSION   | MindSpore 版本号              | 是  | MindSpore 仓库发行版              | 2.7.2                                                  |
| MINDFORMERS_VERSION | MindSpore Transformers 版本号 | 是  | MindSpore Transformers 仓库发行版 | 1.8.0                                                  |
| PIP_INDEX_URL       | pip 安装源地址（默认华为云源）          | 否  | PyPI 镜像源                     | https://mirrors.huaweicloud.com/repository/pypi/simple |

## 快速开始

### 构建 MindSpore Transformers 镜像

```bash
docker build \
--build-arg CANN_VERSION=8.5.0 \
--build-arg CHIP_ARCH=910b \
--build-arg OS_SYSTEM=ubuntu22.04 \
--build-arg PY_VERSION=py3.11 \
--build-arg MINDSPORE_VERSION=2.7.2 \
--build-arg MINDFORMERS_VERSION=1.8.0 \
--build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
-t mindformers:1.8.0-910b-ubuntu22.04-py3.11 \
-f Dockerfile .
```

### 运行 MindSpore Transformers 容器

```bash
docker run \
    --privileged \
    --name mindformers_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspore:tag bash
```

### 安全风险

在使用 Docker 容器运行 MindSpore Transformers 时，需要注意以下安全风险：

- **使用 root 用户运行**：  
  容器默认以 root 用户身份运行，可能带来安全隐患。建议在生产环境中创建非特权用户来运行应用程序。

- **使用 `--privileged` 特权模式运行**：  
  为保证 Ascend NPU 功能正常，容器可能需要启用 `--privileged`。该模式会提升容器对宿主机设备和系统资源的访问权限，增加潜在安全风险。建议仅在可信环境中使用，并结合设备白名单、资源限制及网络隔离等方式降低风险。

- **缺少 CPU 和内存资源限制**：  
  未设置资源限制可能导致容器消耗过多系统资源，影响宿主机性能。建议使用 `--cpus` 和 `--memory` 参数限制资源使用。

## 支持的硬件

| 芯片系列    | 产品示例                             | 架构                    |
|---------|----------------------------------|-----------------------|
| 昇腾 910B | Atlas 800T A2、Atlas 900 A2 PoD   | 自动识别 (ARM64/x86_64)   |
| 昇腾 A3   | Atlas 800T A3                    | 自动识别 (ARM64/x86_64)   |

> Tips: 使用 `docker manifest inspect` 命令可以查看镜像支持的系统架构。

## 许可证

查看这些镜像中包含的 MindSpore 的[许可证信息](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE) 和 MindSpore Transformers 的[许可证信息](https://atomgit.com/mindspore/mindformers/blob/master/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。