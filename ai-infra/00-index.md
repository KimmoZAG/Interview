---
tags:
  - AI Infra
  - 推理引擎
  - 算子优化
  - GPU
  - 分布式训练
description: AI 基础设施工程师视角，涵盖推理栈、算子优化、LLM 架构、分布式通信。
---

# AI Infra

工程师视角的 AI 基础设施知识体系，以**"能做出正确系统判断"**为核心目标。

---

<div class="grid cards" markdown>

- :material-speedometer: **算子优化**

    ---

    单算子 / 子图 / Kernel 为什么快、为什么慢、如何定位和优化。

    [:octicons-arrow-right-24: 进入](01-operator-optimization/00-index.md)

- :material-server-network: **推理引擎**

    ---

    模型如何从请求进入，经过 runtime / compiler / scheduler / KV cache，最终稳定产生 token。

    [:octicons-arrow-right-24: 进入](02-inference-engine/00-index.md)

- :material-brain: **LLM 架构**

    ---

    推理优化工程师需要懂到什么程度的模型知识，才能做出正确的系统判断。

    [:octicons-arrow-right-24: 进入](03-llm-architecture/00-index.md)

- :material-network: **分布式通信**

    ---

    多卡系统为什么慢、慢在什么通信模式、为何通信会比计算更早成为瓶颈。

    [:octicons-arrow-right-24: 进入](04-communication/00-index.md)

- :material-bookshelf: **附录**

    ---

    术语表、坑位清单，以及 Scaling Laws、数据工程、Post-training 等扩展专题。

    [:octicons-arrow-right-24: 进入](05-appendix/00-index.md)

</div>

---

## 各节概览

### 算子优化

| 篇目 | 核心问题 |
|---|---|
| [张量与内存布局](01-operator-optimization/01-tensors-shapes-layout.md) | stride、contiguous、view 与 copy 的代价 |
| [Kernel 执行模型](01-operator-optimization/02-kernel-execution-model.md) | SIMD / SIMT / warp / block 和 occupancy |
| [内存层级与 Roofline](01-operator-optimization/03-memory-hierarchy-and-roofline.md) | arithmetic intensity，memory-bound vs compute-bound |
| [计算图融合与调度](01-operator-optimization/04-graph-fusion-scheduling.md) | op fusion、kernel launch overhead、调度策略 |
| [量化基础](01-operator-optimization/05-quantization-basics.md) | INT8/INT4、calibration、量化误差 |
| [FlashAttention 与 IO-aware](01-operator-optimization/06-flashattention-io-aware.md) | tiling、重计算、IO 感知的 Attention 实现 |

### 推理引擎

| 篇目 | 核心问题 |
|---|---|
| [推理栈全景](02-inference-engine/01-inference-stack-overview.md) | 前端 → IR → 编译 → Runtime 全流程 |
| [ONNX Runtime / TensorRT](02-inference-engine/02-runtime-onnxruntime-tensorrt.md) | session、execution provider、TRT plan |
| [图编译 TVM/MLIR/XLA](02-inference-engine/03-graph-compiler-tvm-mlir-xla.md) | IR 设计、pass 流水线、代码生成 |
| [LLM Serving](02-inference-engine/04-llm-serving.md) | continuous batching、KV cache 管理、常见框架 |
| [推理优化 Playbook](02-inference-engine/05-optimization-playbook.md) | 定位 → 动作 → 验证的系统方法 |
| [可观测性与调试](02-inference-engine/06-observability-and-debugging.md) | TTFT / TPOT / p95、tracing、profiling |
| [Paged KV 与 Allocator](02-inference-engine/07-paged-kv-and-allocator.md) | PagedAttention、块管理、显存稳定性 |
| [长上下文推理](02-inference-engine/08-long-context-serving.md) | chunked prefill、context 切分、扩展策略 |

### LLM 架构

| 篇目 | 核心问题 |
|---|---|
| [Transformer 最小知识](03-llm-architecture/01-transformer-minimum.md) | 推理视角下必须懂的 Transformer |
| [Attention 与 KV Cache](03-llm-architecture/02-attention-kv-cache.md) | MHA/GQA/MLA，KV Cache 的成本来源 |
| [Norm/激活/数值稳定性](03-llm-architecture/03-norm-activation-stability.md) | LayerNorm/RMSNorm、SwiGLU、梯度问题 |
| [Tokenizer 与采样](03-llm-architecture/04-tokenization-and-sampling.md) | BPE、temperature、top-p、speculative decoding |
| [评测与基准](03-llm-architecture/05-evaluation-and-benchmarking.md) | perplexity、MMLU、GPQA、latency / throughput |
| [MoE 最小导读](03-llm-architecture/06-moe-minimum.md) | 稀疏激活、路由、系统代价 |
| [训练资源核算](03-llm-architecture/07-training-resource-accounting.md) | FLOPs、显存、tokens 的账怎么算 |

### 分布式通信

| 篇目 | 核心问题 |
|---|---|
| [并行训练策略](04-communication/01-training-parallelism.md) | DP / TP / PP / SP / FSDP / ZeRO |
| [通信基础](04-communication/02-communication-foundations.md) | latency / bandwidth / topology |
| [互联与拓扑](04-communication/03-interconnects-and-topology.md) | PCIe / NVLink / NVSwitch / InfiniBand / RDMA |
| [集合通信原语](04-communication/04-collectives.md) | all-reduce / all-gather / reduce-scatter / all-to-all |
| [并行到通信映射](04-communication/05-parallelism-to-communication.md) | 每种并行策略引入的通信体积与频次 |

---

## 迁移说明

- 重构路线图：[`restructure-roadmap.md`](restructure-roadmap.md)
- 章节模板：[`05-appendix/chapter-template.md`](05-appendix/chapter-template.md)
