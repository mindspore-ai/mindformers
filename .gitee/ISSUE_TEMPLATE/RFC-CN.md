---
name: RFC 提案 (中文)
about: 提交一个新的 RFC (Request for Comments) 提案 (中文版)
title: "[RFC] "
labels: kind/rfc
---

# [RFC] [标题]

**作者:**
* @nickname
* @nickname

**状态:** 草稿 (Draft) / 评审中 (Reviewing) / 已接受 (Accepted) / 已拒绝 (Rejected)
**创建日期:** YYYY-MM-DD

## **摘要 (Summary)**
用一段简短的文字或要点列表，快速说明项目背景以及你打算做什么。
*   这个提案的背景是什么？
*   高层目标是什么？

## **动机 (Motivation)**
为什么提出这个提案？它为什么重要？
*   **问题陈述**: 我们要解决什么问题？
*   **影响**: 它将如何影响 `MindFormers` 的用户和开发者？
*   **价值**: 为什么现在需要做这件事？

## **建议实现方案 (Proposed Implementation)**
这是 RFC 的核心部分。请详细解释设计方案，足以让熟悉 MindSpore/MindFormers 的人进行实现。

### **设计细节**
*   描述架构和逻辑流。
*   如果通过图表能更好地说明，请附上图表。

### **API 与接口**
*   提供新接口的代码片段（例如：新的 `Cell` 类, `Ops` 算子, 或工具函数）。
*   用户将如何调用或使用此功能？

### **兼容性 (Compatibility)**
*   **API 兼容性**: 是否引入了 Breaking Changes？是否保持了现有接口的兼容性？

## **验收标准与指标 (Metrics & Verification)**
我们如何衡量此功能是否成功？验收标准是什么？

*   **性能指标**:
    *   吞吐量 (tokens/sec), TFLOPS, 或 Step Time。
    *   Ascend 设备上的显存占用 (Peak Memory)。
*   **精度/收敛性**:
    *   与 PyTorch/Megatron-LM 基线的对齐情况（例如：Loss 曲线对比，相对误差 < 1e-5）。
*   **测试计划**:
    *   将添加哪些单元测试或集成测试？
    *   将使用哪些模型配置（例如：Llama3-8B, Qwen2-72B）进行验证？

## **约束与缺点 (Drawbacks & Constraints)**
评估风险和限制条件。

*   **破坏性变更**: 现有的模型或配置会受到影响吗？
*   **复杂度**: 是否引入了显著的代码复杂度或编译开销？
*   **依赖项**: 是否依赖特定的 MindSpore 版本、CANN 包版本或新的第三方库？

## **替代方案 (Alternatives)**
还考虑过哪些其他设计？
*   为什么选择这个方案而不是其他方案？
*   如果不做这件事会有什么后果？

## **文档与易用性 (Documentation & Usability)**
如何教用户使用此功能？

*   **配置 (`yaml`)**:
    *   需要在模型配置 `yaml` 文件中添加哪些新参数？
    *   默认值应该是什么？
*   **指南**:
    *   是否需要更新 README 或提供教程？
*   **示例**:
    *   是否会在 `scripts/examples` 中添加新的启动脚本？

## **待解决问题 (Unresolved Questions)**
*   设计的哪些部分仍需讨论？

