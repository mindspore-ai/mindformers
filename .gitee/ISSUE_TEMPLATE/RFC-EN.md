---
name: RFC Proposal (English)
about: Create a new Request for Comments (RFC) proposal (English version)
title: "[RFC] "
labels: kind/rfc
---

# [RFC] [Title]

**Authors:**
* @nickname
* @nickname

**Status:** Draft / Reviewing / Accepted / Rejected
**Created:** YYYY-MM-DD

## **Summary**
A short paragraph or bullet list that quickly explains the background and what you're trying to do.
*   What is the context of this proposal?
*   What is the high-level goal?

## **Motivation**
What motivates this proposal and why is it important?
*   **Problem Statement**: What problem are we solving?
*   **Impact**: How does it impact `MindFormers` users and developers?
*   **Value**: Why is this the right thing to do now?

## **Proposed Implementation**
This is the core of the RFC. Explain the design in sufficient detail for someone familiar with MindSpore/MindFormers to implement it.

### **Design Details**
*   Describe the architecture and logic flow.
*   Use diagrams if helpful.

### **API & Interfaces**
*   Provide code snippets for new Interfaces (e.g., new `Cell`, `Ops`, or Utility functions).
*   How will the user interact with this feature?

### **Compatibility**
*   **API Compatibility**: Does it introduce breaking changes? Is it compatible with existing interfaces?

## **Metrics & Verification**
How do we know this feature is successful? What are the acceptance criteria?

*   **Performance Metrics**:
    *   Throughput (tokens/sec), TFLOPS, or Step Time.
    *   Memory usage (Peak Memory) on Ascend devices.
*   **Accuracy/Convergence**:
    *   Alignment with PyTorch/Megatron-LM baselines (e.g., Loss curve comparison, relative error < 1e-5).
*   **Test Plan**:
    *   What unit tests or integration tests will be added?
    *   Which model configurations (e.g., Llama3-8B, Qwen2-72B) will be used for validation?

## **Drawbacks & Constraints**
Evaluate risks and limitations.

*   **Breaking Changes**: Will existing models or configs break?
*   **Complexity**: Does this introduce significant code complexity or compilation overhead?
*   **Dependencies**: Does it depend on a specific version of MindSpore, CANN, or new third-party libraries?

## **Alternatives**
What other designs have been considered?
*   Why was this approach chosen over others?
*   What is the impact of *not* doing this?

## **Documentation & Usability**
How will users learn and use this feature?

*   **Configuration (`yaml`)**:
    *   What new parameters need to be added to the model config `yaml` files?
    *   What are the default values?
*   **Guides**:
    *   Do we need to update the README or provide a tutorial?
*   **Examples**:
    *   Will a new script be added to `scripts/examples`?

## **Unresolved Questions**
*   What parts of the design are still open for discussion?

