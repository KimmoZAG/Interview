# 推理引擎索引

定位：回答 **模型如何从请求进入，经过 runtime / compiler / scheduler / cache / allocator，最终稳定地产生 token**。

## 建议学习顺序

1. 推理栈全景：模型前端 → IR → 编译 → Runtime
2. Runtime：ONNX Runtime / TensorRT / vLLM 的角色
3. 图编译：TVM / MLIR / XLA
4. Serving：batching / queue / continuous batching
5. KV Cache 管理：连续布局、分页布局、块管理
6. Allocator / 内存池 / 显存稳定性
7. 推理优化 Playbook：定位 → 动作 → 验证
8. 可观测性与调试：TTFT / TPOT / p95 / tracing / profiling
9. 长上下文 Serving

## 存量内容映射

- [推理栈全景：前端→图→kernel→执行](01-inference-stack-overview.md)
- [Runtime：ONNX Runtime / TensorRT](02-runtime-onnxruntime-tensorrt.md)
- [图编译：TVM / MLIR / XLA](03-graph-compiler-tvm-mlir-xla.md)
- [LLM Serving：batching、paged KV、常见方案](04-llm-serving.md)
- [推理优化 Playbook](05-optimization-playbook.md)
- [可观测性与调试](06-observability-and-debugging.md)
- [Paged KV 与 Allocator](07-paged-kv-and-allocator.md)
- [长上下文 Serving](08-long-context-serving.md)

## 后续补强建议

- 新增一篇：`02-runtime-landscape.md`（补 vLLM / TGI / SGLang 视角）
- 新增一篇：`05-kv-cache-layout.md`
- 新增一篇：`06-allocator-and-memory-pool.md`
- 把 Playbook 作为整个仓库的主线章节持续维护
