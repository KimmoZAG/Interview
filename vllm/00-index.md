# vLLM 系列：从模型加载到推理加速

> **目标**：深入理解 vLLM 内部机制，能对"为什么 vLLM 快、怎么支持新模型、怎么换算子"做出精确的系统性回答。

---

## 系列导读

vLLM 是目前工业界最广泛使用的 LLM 推理框架之一。理解它不只是"会用 API"，而是要能回答：

- 权重文件是怎么被加载进 GPU 显存的？
- 为什么新模型加进来只需要注册一行？
- FlashAttention、FlashInfer 是怎么被热插拔替换的？
- PagedAttention、CUDA Graphs、Prefix Cache 各自解决什么瓶颈？

本系列从**模型加载**出发，逐层深入到算子替换与推理加速，每篇聚焦一个核心问题。

---

## 知识地图

```
vLLM 架构总览
    ├── 模型加载机制          ← 权重从磁盘到 GPU 显存的全路径
    │       └── 多模型支持    ← 不同架构如何注册与实例化
    ├── 算子替换与定制        ← 注意力后端、量化、自定义核
    └── 推理加速技术          ← PagedAttention、连续批处理、前缀缓存、投机解码
```

---

## 章节列表

<div class="grid cards" markdown>

-   **01 · 架构总览**

    ---
    LLMEngine / Worker / Scheduler / BlockManager 四层分工，请求从入队到 token 输出的完整链路。

    [:octicons-arrow-right-24: 阅读](01-architecture-overview.md)

-   **02 · 模型加载机制**

    ---
    ModelConfig 解析 → ModelRegistry 查找 → 权重分片 → Tensor Parallel 分布 → KV Cache 预分配。

    [:octicons-arrow-right-24: 阅读](02-model-loading.md)

-   **03 · 多模型支持与注册**

    ---
    vLLM 如何用模型注册表支持 LLaMA、Mistral、Qwen、Gemma 等数十种架构，以及如何接入新模型。

    [:octicons-arrow-right-24: 阅读](03-model-registry.md)

-   **04 · 算子替换与定制**

    ---
    注意力后端（FlashAttention / FlashInfer / xFormers）热插拔原理，量化线性层替换，自定义 CUDA 核注入。

    [:octicons-arrow-right-24: 阅读](04-custom-ops-and-operator-replacement.md)

-   **05 · 推理加速技术**

    ---
    PagedAttention 解决碎片化；连续批处理提升 GPU 利用率；CUDA Graphs 消除 kernel launch 开销；前缀缓存与投机解码进一步压低延迟。

    [:octicons-arrow-right-24: 阅读](05-inference-acceleration.md)

</div>

---

## 前置知识

| 主题 | 链接 |
|------|------|
| KV Cache 基础 | [Attention & KV Cache](../ai-infra/03-llm-architecture/02-attention-kv-cache.md) |
| Paged KV 与 Allocator | [Paged KV & Allocator](../ai-infra/02-inference-engine/07-paged-kv-and-allocator.md) |
| LLM Serving 框架 | [LLM Serving](../ai-infra/02-inference-engine/04-llm-serving.md) |
| FlashAttention IO-aware | [FlashAttention](../ai-infra/01-operator-optimization/06-flashattention-io-aware.md) |
| 量化基础 | [量化 Basics](../ai-infra/01-operator-optimization/05-quantization-basics.md) |
| 训练并行策略 | [训练并行](../ai-infra/04-communication/01-training-parallelism.md) |

---

[← 返回 AI Infra 总索引](../ai-infra/00-index.md)
