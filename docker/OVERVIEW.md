# MindSpore
 	 
> English | [中文](./OVERVIEW_CN.md)

## Quick Reference

- MindSpore Transformers is maintained by the [MindSpore Transformers Community](https://atomgit.com/mindspore/mindformers) 维护

- Where to get help

    - [MindSpore Transformers Documentation](https://www.mindspore.cn/mindformers/docs/zh-CN/stable/index.html)
    - [MindSpore Transformers Community](https://www.mindspore.cn/sig/MindSpore%20Transformers)
    - [Issue Feedback](https://atomgit.com/mindspore/mindformers/issues)

## MindSpore Transformers

The MindSpore Transformers suite aims to build a comprehensive development toolkit covering the entire lifecycle of large-scale models, including pre-training, fine-tuning, evaluation, inference, and deployment. It provides industry-leading large Transformer-based language models, multimodal understanding models, and omni-modal models, enabling users to easily achieve end-to-end development of large-scale models.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow this format:

```text
 <MindSpore Transformers Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

| Field                          | Example Values               | Description                                                                           |
|--------------------------------|------------------------------|---------------------------------------------------------------------------------------|
| MindSpore Transformers Version | 1.8.0                        | Corresponds to the version identifier in MindSpore Transformers official release tags |
| Hardware Info (Chip)           | 910b / a3                    | Ascend chip model identifier                                                          |
| Operating System               | ubuntu22.04 / openeuler24.03 | Operating system distribution and version used in the base image                      |
| Python Version                 | py3.11                       | Major Python version built into the image                                             |

> Tips: System architecture is automatically detected via Docker Manifest, no need to specify in the tag.

### Image Repository Address

MindSpore Transformers Ascend images are hosted on Huawei Cloud SWR image repository:

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers
```

**Full Image Example：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers:1.8.0-910b-ubuntu22.04-py3.11
```

### 构建参数

| Parameter                      | Description                                                | Required | Source                        | Example Values                                         |
|--------------------------------|------------------------------------------------------------|----------|-------------------------------|--------------------------------------------------------|
| CANN_VERSION                   | Ascend CANN toolkit version                                | Yes      | CANN image tag                | 8.5.0                                                  |
| CHIP_ARCH                      | Ascend chip architecture identifier                        | Yes      | Tag specification             | 910b / a3                                              |
| OS_SYSTEM                      | Base image operating system and version                    | Yes      | Tag specification             | ubuntu22.04 / openeuler24.03                           |
| PY_VERSION                     | Python version built into the base image                   | Yes      | Tag specification             | py3.11                                                 |
| MINDSPORE_VERSION              | MindSpore version number                                   | Yes      | MindSpore repository releases | 2.7.2                                                  |
| MINDSPORE_TRANSFORMERS_VERSION | MindSpore version number                                   | Yes      | MindSpore repository releases | 1.8.0                                                  |
| PIP_INDEX_URL                  | pip installation source URL (default: Huawei Cloud mirror) | No       | PyPI mirror source            | https://mirrors.huaweicloud.com/repository/pypi/simple |

## Quick Start

### Building MindSpore Image

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

### Security Risks

When running MindSpore Transformers in Docker containers, pay attention to the following security risks:

- **Running as the root user**:  
  Containers run as the root user by default, which may introduce security risks. It is recommended to create and use a non-privileged user in production environments.

- **Running with `--privileged` mode**:  
  To ensure Ascend NPU functionality works properly, the container may require the `--privileged` option. This mode grants the container elevated access to host devices and system resources, increasing potential security risks. It is recommended to use it only in trusted environments and reduce risks through device whitelisting, resource limits, and network isolation.

- **Missing CPU and memory resource limits**:  
  Without resource limits, containers may consume excessive system resources and impact host performance. It is recommended to use the `--cpus` and `--memory` options to limit resource usage.

## Supported Hardware

| Chip Series | Product Examples                | Architecture                  |
|-------------|---------------------------------|-------------------------------|
| Ascend 910B | Atlas 800T A2, Atlas 900 A2 PoD | Auto-detected (ARM64/x86_64)  |
| Ascend A3   | Atlas 800T A3                   | Auto-detected (ARM64/x86_64)  |

> Tips: Use the `docker manifest inspect` command to view the system architectures supported by an image.

## License

View the [license information](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE) for MindSpore and [license information](https://atomgit.com/mindspore/mindformers/blob/master/LICENSE)included in these images.
 	 
As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their respective licenses.