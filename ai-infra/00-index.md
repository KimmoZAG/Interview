# 索引

## 1) 基本算子（operators）

- [张量、shape 与内存布局](operators/01-tensors-shapes-layout.md)
- [Kernel 与并行基础（SIMD/SIMT）](operators/02-kernels-and-parallelism.md)
- [内存层级与性能模型（roofline）](operators/03-memory-hierarchy-and-roofline.md)
- [计算图、融合与调度](operators/04-graph-fusion-scheduling.md)
- [量化基础（INT8/INT4）与误差](operators/05-quantization-basics.md)
- [训练并行策略：DP / TP / PP / FSDP / ZeRO](operators/06-training-parallelism.md)
- [FlashAttention 与 IO-aware Attention：为什么快，快在哪里](operators/07-flashattention-io-aware.md)

## 2) 模型知识（models）

- [Transformer 推理所需的最小知识](models/01-transformer-minimum.md)
- [Attention、KV cache 与吞吐/延迟](models/02-attention-kv-cache.md)
- [常见层：Norm/激活/残差 与数值稳定性](models/03-norm-activation-stability.md)
- [Tokenizer 与采样（top-k/top-p/temperature）](models/04-tokenization-and-sampling.md)
- [评测与基准：accuracy/latency/throughput](models/05-evaluation-and-benchmarking.md)
- [训练资源核算：参数、激活、梯度、优化器状态](models/06-training-resource-accounting.md)
- [Post-training 与 Alignment：SFT、偏好优化、RL 的最小框架](models/07-post-training-and-alignment.md)
- [MoE 最小导读：路由、容量、负载均衡、系统代价](models/08-moe-minimum.md)
- [Reward 与 Verifier 设计：后训练到底在优化什么](models/09-reward-and-verifier-design.md)

## 3) 推理框架（inference）

- [推理栈全景：前端→图→kernel→执行](inference/01-inference-stack-overview.md)
- [Runtime：ONNX Runtime / TensorRT（要点清单）](inference/02-onnxruntime-tensorrt.md)
- [图编译：TVM / MLIR / XLA（概念对齐）](inference/03-graph-compiler-tvm-mlir-xla.md)
- [LLM Serving：batching、paged KV、常见方案](inference/04-llm-serving.md)
- [推理优化 Playbook（定位→动作→验证）](inference/05-optimization-playbook.md)
- [可观测性与调试（profiling、tracing、metrics）](inference/06-observability-and-debugging.md)
- [Scaling Laws 工程用法：预算、外推、实验设计](inference/07-scaling-laws-and-budgeting.md)
- [预训练数据工程：来源、过滤、去重、质量评估](inference/08-pretraining-data-engineering.md)
- [训练指标 vs 产品指标：为什么 loss 变好不代表产品更好](inference/09-training-metrics-vs-product-metrics.md)
- [数据混配与 Curriculum：模型能力是怎样被数据配方塑形的](inference/10-data-mixing-and-curriculum.md)
- [Paged KV 与 Allocator：为什么 KV cache 不是“开个数组”这么简单](inference/11-paged-kv-and-allocator.md)
- [长上下文训练与推理：瓶颈为什么会成倍放大](inference/12-long-context-training-and-serving.md)

## 4) 附录（appendix）

- [术语表](appendix/glossary.md)
- [坑位清单（持续更新）](appendix/gotchas.md)
- [CS336：Language Modeling from Scratch 资源导读](appendix/cs336-from-scratch-resource-map.md)
- [CS336 逐讲映射：从课程到 AI Infra 笔记](appendix/cs336-lecture-to-ai-infra-map.md)
- [参考资料](appendix/references.md)
