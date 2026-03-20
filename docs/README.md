---
description: 工程师视角的技术面试备考笔记，聚焦 AI Infra 系统工程与 C++ 核心知识
---

# Interview Notes

工程师视角的技术面试准备笔记，聚焦 **AI Infra 系统工程** 与 **C++ 核心知识**，以"能做出正确系统判断"为目标。

---

<div class="grid cards" markdown>

- :material-memory: **LMCache**

    ---

    基于 LMCache 源码的面试导向学习笔记，覆盖 KV Cache 复用、P2P、PD、控制平面、可观测性与 Operator。

    [:octicons-arrow-right-24: 进入 LMCache](../LMCache/00-index.md)

- :material-chip: **AI Infra**

    ---

    推理栈、算子优化、LLM 架构、分布式通信，以及训练与数据工程的工程师视角。

    [:octicons-arrow-right-24: 进入 AI Infra](ai-infra/00-index.md)

- :material-rocket-launch: **vLLM**

    ---

    从模型加载、多模型注册、算子替换到推理加速，按框架主线理解 vLLM 内部机制。

    [:octicons-arrow-right-24: 进入 vLLM](vllm/00-index.md)

- :material-rabbit: **Chitu**

    ---

    聚焦赤兔推理引擎的动态批处理、KV 管理、量化、硬件适配与平台协同。

    [:octicons-arrow-right-24: 进入 Chitu](chitu/00-index.md)

- :material-language-cpp: **C++**

    ---

    工具链、对象生命周期、模板、STL、并发基础，以及 C++11 到 C++23 各版本特性。

    [:octicons-arrow-right-24: 进入 C++](c++/00-index.md)

- :material-school: **CS336**

    ---

    Stanford CS336《从零构建语言模型》课程笔记，覆盖训练、推理、对齐全链路。

    [:octicons-arrow-right-24: 进入 CS336](cs336/00-index.md)

</div>

## 内容索引

=== "AI Infra"

    **算子优化**

    - [索引页](ai-infra/01-operator-optimization/00-index.md)
    - [张量与内存布局](ai-infra/01-operator-optimization/01-tensors-shapes-layout.md)
    - [Kernel 执行模型](ai-infra/01-operator-optimization/02-kernel-execution-model.md)
    - [内存层级与 Roofline](ai-infra/01-operator-optimization/03-memory-hierarchy-and-roofline.md)
    - [计算图融合与调度](ai-infra/01-operator-optimization/04-graph-fusion-scheduling.md)
    - [量化基础](ai-infra/01-operator-optimization/05-quantization-basics.md)
    - [FlashAttention 与 IO-aware](ai-infra/01-operator-optimization/06-flashattention-io-aware.md)

    **推理引擎**

    - [索引页](ai-infra/02-inference-engine/00-index.md)
    - [推理栈全景](ai-infra/02-inference-engine/01-inference-stack-overview.md)
    - [ONNX Runtime / TensorRT](ai-infra/02-inference-engine/02-runtime-onnxruntime-tensorrt.md)
    - [图编译 TVM/MLIR/XLA](ai-infra/02-inference-engine/03-graph-compiler-tvm-mlir-xla.md)
    - [LLM Serving](ai-infra/02-inference-engine/04-llm-serving.md)
    - [推理优化 Playbook](ai-infra/02-inference-engine/05-optimization-playbook.md)
    - [可观测性与调试](ai-infra/02-inference-engine/06-observability-and-debugging.md)
    - [Paged KV 与 Allocator](ai-infra/02-inference-engine/07-paged-kv-and-allocator.md)
    - [长上下文推理](ai-infra/02-inference-engine/08-long-context-serving.md)

    **LLM 架构**

    - [索引页](ai-infra/03-llm-architecture/00-index.md)
    - [Transformer 最小知识](ai-infra/03-llm-architecture/01-transformer-minimum.md)
    - [Attention 与 KV Cache](ai-infra/03-llm-architecture/02-attention-kv-cache.md)
    - [Norm / 激活 / 数值稳定性](ai-infra/03-llm-architecture/03-norm-activation-stability.md)
    - [Tokenizer 与采样](ai-infra/03-llm-architecture/04-tokenization-and-sampling.md)
    - [评测与基准](ai-infra/03-llm-architecture/05-evaluation-and-benchmarking.md)
    - [MoE 最小导读](ai-infra/03-llm-architecture/06-moe-minimum.md)
    - [训练资源核算](ai-infra/03-llm-architecture/07-training-resource-accounting.md)

    **分布式通信**

    - [索引页](ai-infra/04-communication/00-index.md)
    - [并行训练策略](ai-infra/04-communication/01-training-parallelism.md)
    - [通信基础](ai-infra/04-communication/02-communication-foundations.md)
    - [互联与拓扑](ai-infra/04-communication/03-interconnects-and-topology.md)
    - [集合通信原语](ai-infra/04-communication/04-collectives.md)
    - [并行到通信映射](ai-infra/04-communication/05-parallelism-to-communication.md)

=== "CS336"

    - [学习路线图](cs336/study-roadmap.md)
    - [面试准备指南](cs336/interview-prep-guide.md)
    - [01 课程总览与 Tokenization](cs336/01-overview-and-tokenization.md)
    - [02 PyTorch 与资源核算](cs336/02-pytorch-and-resource-accounting.md)
    - [03 架构与超参数](cs336/03-architectures-and-hyperparameters.md)
    - [04 Mixture of Experts](cs336/04-mixture-of-experts.md)
    - [05 GPU 基础](cs336/05-gpus.md)
    - [06 Kernel 与 Triton](cs336/06-kernels-and-triton.md)
    - [07 并行训练（一）](cs336/07-parallelism.md)
    - [08 并行训练（二）](cs336/08-parallelism-part2.md)
    - [09 Scaling Law 基础](cs336/09-scaling-laws-fundamentals.md)
    - [10 推理优化](cs336/10-inference.md)
    - [11 Scaling Law 案例](cs336/11-scaling-laws-case-studies.md)
    - [12 评测](cs336/12-evaluation.md)
    - [13 数据工程（一）](cs336/13-data.md)
    - [14 数据工程（二）](cs336/14-data-part2.md)
    - [15 对齐（一）SFT 与 RLHF](cs336/15-alignment-sft-rlhf.md)
    - [16 对齐（二）RL](cs336/16-alignment-rl.md)
    - [17 对齐（三）Policy Gradient](cs336/17-alignment-rl-part2.md)

=== "vLLM & Chitu"

    **vLLM**

    - [架构总览](vllm/01-architecture-overview.md)
    - [模型加载机制](vllm/02-model-loading.md)
    - [多模型支持与注册](vllm/03-model-registry.md)
    - [算子替换与定制](vllm/04-custom-ops-and-operator-replacement.md)
    - [推理加速技术](vllm/05-inference-acceleration.md)

    **Chitu**

    - [赤兔推理引擎面试笔记](chitu/09-chitu-inference-engine.md)

=== "C++"

    **通用知识**

    - [工具链与构建](c++/general/01-toolchain-and-build.md)
    - [对象生命周期与值语义](c++/general/02-object-lifetime-and-value-semantics.md)
    - [RAII 与智能指针](c++/general/03-raii-and-smart-pointers.md)
    - [模板与类型](c++/general/04-templates-and-types.md)
    - [STL 容器/迭代器/算法](c++/general/05-stl-containers-iterators-algorithms.md)
    - [并发基础](c++/general/06-concurrency-basics.md)
    - [运行时多态](c++/general/07-runtime-polymorphism.md)

    **版本特性**

    - [C++11](c++/versions/cpp11.md)
    - [C++14](c++/versions/cpp14.md)
    - [C++17](c++/versions/cpp17.md)
    - [C++20](c++/versions/cpp20.md)
    - [C++23](c++/versions/cpp23.md)

---

## 使用说明

!!! tip "定位"
    本站不是知识大百科，而是**面试导向的精要笔记**。每篇对应具体的工程场景或面试问题，阅读前建议先看各节的 **索引页** 了解全貌与推荐顺序。

!!! info "推荐阅读路径"
    - **快速热身（3 天）**：CS336 01→02→03 + AI Infra 算子优化 + C++ 通用知识
    - **系统深入（1 周）**：按各节索引页的"建议学习顺序"逐篇阅读
    - **面试冲刺**：各节索引页均有"面试导向"推荐顺序
