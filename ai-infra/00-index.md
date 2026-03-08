# 索引

## 1) 基本算子（operators）

- [张量、shape 与内存布局](operators/01-tensors-shapes-layout.md)
- [Kernel 与并行基础（SIMD/SIMT）](operators/02-kernels-and-parallelism.md)
- [内存层级与性能模型（roofline）](operators/03-memory-hierarchy-and-roofline.md)
- [计算图、融合与调度](operators/04-graph-fusion-scheduling.md)
- [量化基础（INT8/INT4）与误差](operators/05-quantization-basics.md)

## 2) 模型知识（models）

- [Transformer 推理所需的最小知识](models/01-transformer-minimum.md)
- [Attention、KV cache 与吞吐/延迟](models/02-attention-kv-cache.md)
- [常见层：Norm/激活/残差 与数值稳定性](models/03-norm-activation-stability.md)
- [Tokenizer 与采样（top-k/top-p/temperature）](models/04-tokenization-and-sampling.md)
- [评测与基准：accuracy/latency/throughput](models/05-evaluation-and-benchmarking.md)

## 3) 推理框架（inference）

- [推理栈全景：前端→图→kernel→执行](inference/01-inference-stack-overview.md)
- [Runtime：ONNX Runtime / TensorRT（要点清单）](inference/02-onnxruntime-tensorrt.md)
- [图编译：TVM / MLIR / XLA（概念对齐）](inference/03-graph-compiler-tvm-mlir-xla.md)
- [LLM Serving：batching、paged KV、常见方案](inference/04-llm-serving.md)
- [推理优化 Playbook（定位→动作→验证）](inference/05-optimization-playbook.md)
- [可观测性与调试（profiling、tracing、metrics）](inference/06-observability-and-debugging.md)

## 4) 附录（appendix）

- [术语表](appendix/glossary.md)
- [坑位清单（持续更新）](appendix/gotchas.md)
- [参考资料](appendix/references.md)
