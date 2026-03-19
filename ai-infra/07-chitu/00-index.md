# 赤兔（Chitu）推理引擎专题

> **目标**：把 Chitu 作为独立专题来复习，重点讲清它在动态批处理、KV 管理、量化、硬件适配与云平台协同上的差异化价值。

---

## 专题导读

Chitu 适合放在“推理引擎通用主线”之外单独看，因为它更像一个**带平台生态语境的专项框架**：

- 技术内核上，要能讲清 Continuous Batching、Paged KV Cache、算子融合与量化策略；
- 架构定位上，要能说明它与阿里云 PAI / EAS 的协同关系；
- 对比表达上，要能把它和 vLLM、TensorRT-LLM 放到同一张表里讲场景差异。

---

## 当前内容

<div class="grid cards" markdown>

-   **01 · 赤兔推理引擎面试笔记**

    ---
    从定位、核心机制、量化与硬件支持，到 Troubleshooting 与高频问答，适合做面试冲刺。

    [:octicons-arrow-right-24: 阅读](09-chitu-inference-engine.md)

</div>

---

## 前置知识

| 主题 | 链接 |
|------|------|
| 推理栈全景 | [推理栈全景](../02-inference-engine/01-inference-stack-overview.md) |
| LLM Serving | [LLM Serving](../02-inference-engine/04-llm-serving.md) |
| Paged KV 与 Allocator | [Paged KV & Allocator](../02-inference-engine/07-paged-kv-and-allocator.md) |
| 量化基础 | [量化 Basics](../01-operator-optimization/05-quantization-basics.md) |
| vLLM 深度解析 | [vLLM 系列导读](../06-vllm/00-index.md) |

---

[← 返回 AI Infra 总索引](../00-index.md)
