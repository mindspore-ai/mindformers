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
<MindSpore Transformers Version>-cann-<CANN Version>-mindspore<MindSpore Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

| Field                          | Values                       | Description                                                                           |
|--------------------------------|------------------------------|---------------------------------------------------------------------------------------|
| MindSpore Transformers Version | 2.0.0                        | Corresponds to the version identifier in MindSpore Transformers official release tags |
| CANN Version                    | 9.1.0                        | CANN version built into the image                                                      |
| MindSpore Version               | 2.10.0                       | MindSpore version built into the image                                                 |
| Hardware Info (Chip)           | 910b / a3                    | NPU chip model identifier                                                             |
| Operating System               | ubuntu22.04 / openeuler24.03 | Operating system distribution and version used in the base image                      |
| Python Version                 | py3.12                       | Major Python version built into the image                                             |

> Tips: System architecture is automatically detected via Docker Manifest, no need to specify in the tag.

### Available Tags

- `2.0.0-cann-9.1.0-mindspore2.10.0-910b-ubuntu22.04-py3.12`
- `2.0.0-cann-9.1.0-mindspore2.10.0-910b-openeuler24.03-py3.12`
- `2.0.0-cann-9.1.0-mindspore2.10.0-a3-ubuntu22.04-py3.12`
- `2.0.0-cann-9.1.0-mindspore2.10.0-a3-openeuler24.03-py3.12`

### Image Repository Address

MindSpore Transformers images are hosted on the Huawei Cloud SWR image repository:

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers
```

**Full Image Example：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers:2.0.0-cann-9.1.0-mindspore2.10.0-910b-ubuntu22.04-py3.12
```

### 构建参数

See dockerfile: [dockerfile](https://gitcode.com/mindspore/mindformers/blob/master/docker/Dockerfile)

| Parameter                      | Description                                                | Required | Source                        | Values                                                 |
|--------------------------------|------------------------------------------------------------|----------|-------------------------------|--------------------------------------------------------|
| CANN_VERSION                   | CANN toolkit version                                       | Yes      | CANN image tag                | 9.1.0                                                  |
| CHIP_ARCH                      | NPU chip architecture identifier                           | Yes      | Tag specification             | 910b / a3                                              |
| OS_SYSTEM                      | Base image operating system and version                    | Yes      | Tag specification             | ubuntu22.04 / openeuler24.03                           |
| PY_VERSION                     | Python version built into the base image                   | Yes      | Tag specification             | py3.12                                                 |
| MINDSPORE_VERSION              | MindSpore version number                                   | Yes      | MindSpore repository releases | 2.10.0                                                 |
| MINDFORMERS_VERSION | MindSpore Transformers version number                                   | Yes      | MindSpore Transformers repository releases | 2.0.0                                                  |
| PIP_INDEX_URL                  | pip installation source URL (default: Huawei Cloud mirror) | No       | PyPI mirror source            | https://mirrors.huaweicloud.com/repository/pypi/simple |

## Quick Start

### Building MindSpore Image

```bash
docker build \
--build-arg CANN_VERSION=9.1.0 \
--build-arg CHIP_ARCH=910b \
--build-arg OS_SYSTEM=ubuntu22.04 \
--build-arg PY_VERSION=py3.12 \
--build-arg MINDSPORE_VERSION=2.10.0 \
--build-arg MINDFORMERS_VERSION=2.0.0 \
--build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
-t mindformers:2.0.0-cann-9.1.0-mindspore2.10.0-910b-ubuntu22.04-py3.12 \
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
  To ensure NPU functionality works properly, the container may require the `--privileged` option. This mode grants the container elevated access to host devices and system resources, increasing potential security risks. It is recommended to use it only in trusted environments and reduce risks through device whitelisting, resource limits, and network isolation.

- **Missing CPU and memory resource limits**:  
  Without resource limits, containers may consume excessive system resources and impact host performance. It is recommended to use the `--cpus` and `--memory` options to limit resource usage.

## Supported Hardware

| Chip Series | Product Examples                | Architecture                  |
|-------------|---------------------------------|-------------------------------|
| 910B        | Atlas 800T A2, Atlas 900 A2 PoD | Auto-detected (ARM64/x86_64)  |
| A3          | Atlas 800T A3                   | Auto-detected (ARM64/x86_64)  |

> Tips: Use the `docker manifest inspect` command to view the system architectures supported by an image.

## Image Usage Responsibility Statement

The released software images are community editions and are not intended for commercial use. They are provided solely as references for production practices. The responsibility statement is displayed in the image startup information and on the image hosting platform.

## License

View the [license information](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE) for MindSpore and [license information](https://atomgit.com/mindspore/mindformers/blob/master/LICENSE)included in these images.
 	 
As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their respective licenses.
