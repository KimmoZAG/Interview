# vLLM 架构总览

## 核心定义（What & Why）

> vLLM 是一个以**连续批处理 + PagedAttention** 为核心的高吞吐 LLM 推理框架，解决的问题不是"能不能运行模型"，而是"如何让 GPU 资源在动态并发请求下持续满载、延迟可控"。

核心要点：

- **LLMEngine**：唯一对外接口，协调调度、执行、缓存三层
- **Worker / ModelRunner**：运行在每个 GPU 进程，负责实际 forward pass
- **Scheduler**：决定每个 step 哪些 request 参与、分配多少 block
- **BlockSpaceManager**：KV cache 的物理块分配与回收

---

## 四层架构分解

```
┌─────────────────────────────────────────────────────┐
│                   LLMEngine / AsyncLLMEngine         │  ← 请求入队、调度、结果回收
├───────────────────────┬─────────────────────────────┤
│       Scheduler       │    BlockSpaceManager         │  ← 调度策略 + KV block 分配
├───────────────────────┴─────────────────────────────┤
│             ExecutorBase (TP / Ray / MP)              │  ← 多 GPU 执行器抽象
├─────────────────────────────────────────────────────┤
│           Worker  ×  world_size                       │  ← 每卡一个进程
│    ┌───────────────────────────────────────────┐     │
│    │  ModelRunner  →  GPUModel.forward()        │     │  ← 实际矩阵计算
│    │  CacheEngine  →  KV block on HBM           │     │  ← block 级 KV 管理
│    └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

---

## 关键数据结构

| 对象 | 所在模块 | 职责 |
|------|----------|------|
| `LLMEngine` | `vllm/engine/llm_engine.py` | 入口；持有 scheduler、executor、tokenizer |
| `EngineArgs / EngineConfig` | `vllm/config.py` | 所有配置的单一来源（模型、缓存、并行度）|
| `Scheduler` | `vllm/core/scheduler.py` | 每 step 决定 running / swapped / waiting |
| `BlockSpaceManagerV2` | `vllm/core/block_manager.py` | 物理 block 分配树（radix cache 支撑）|
| `Worker` | `vllm/worker/worker.py` | GPU 进程入口；初始化模型 + cache |
| `ModelRunner` | `vllm/worker/model_runner.py` | 构造 attention metadata，执行 forward |
| `CacheEngine` | `vllm/worker/cache_engine.py` | 分配 `kv_caches` 张量，执行 swap |
| `SequenceGroup` | `vllm/sequence.py` | 一个请求 + 其所有 beam/sample 路径 |

---

## 一次推理的完整链路

```
1. add_request()
   LLMEngine 接收 prompt，创建 SequenceGroup，入 waiting 队列

2. step()
   ├── Scheduler.schedule()
   │     ├── 从 waiting/running/swapped 中选出本 step 的 batch
   │     ├── BlockSpaceManager 分配 KV blocks
   │     └── 返回 SchedulerOutputs（prefill + decode sequences）
   │
   ├── Executor.execute_model(scheduler_outputs)
   │     └── (TP 场景) 广播到各 Worker → ModelRunner.execute_model()
   │           ├── 准备 input_ids、positions、attn_metadata
   │           ├── model.forward() → logits
   │           └── sample() → next token ids
   │
   └── process_model_outputs()
         ├── 将 token append 到 sequence
         ├── 检查 stop condition（eos / max_tokens）
         └── 返回已完成的 RequestOutput

3. 重复 step() 直到所有请求完成
```

---

## 关联知识网络

**前置**：[LLM Serving](../ai-infra/02-inference-engine/04-llm-serving.md)、[Paged KV & Allocator](../ai-infra/02-inference-engine/07-paged-kv-and-allocator.md)

**平行**：[vLLM 模型加载](02-model-loading.md)、[vLLM 算子替换](04-custom-ops-and-operator-replacement.md)

**延伸**：[vLLM 推理加速](05-inference-acceleration.md)、[训练并行](../ai-infra/04-communication/01-training-parallelism.md)

---

## 💥 实战踩坑记录

**现象**：`AsyncLLMEngine` 启动后请求长时间 pending，GPU 空载。

**根因**：`max_num_seqs` 设置过小，或 `gpu_memory_utilization` 过低导致可分配 block 数不足，Scheduler 无法推进 waiting 队列。

**排查路径**：
1. 检查 `scheduler_outputs.num_prefill_groups` 是否始终为 0
2. 查看 `BlockSpaceManager.get_num_free_gpu_blocks()`
3. 调高 `gpu_memory_utilization`（默认 0.9）或降低 `max_model_len`

---

## 🎯 面试高频 Q&A

**Q1：vLLM 和 HuggingFace generate() 最本质的区别是什么？**

> HF generate() 是**请求级**的：一次 forward 只服务一个请求，KV cache 连续分配，无法动态合并多请求。vLLM 是**系统级**的：Scheduler 在每个 step 决定哪些请求共享同一次 forward，BlockSpaceManager 以页为单位管理 KV cache，实现高并发、低碎片。

**Q2：LLMEngine 的 step() 和 AsyncLLMEngine 有什么区别？**

> `LLMEngine.step()` 是同步阻塞调用，每次返回当前 step 完成的输出，适合 Python 脚本离线推理。`AsyncLLMEngine` 封装成协程，通过 `asyncio` event loop 和内部后台线程实现非阻塞接口，适合 HTTP 服务（vLLM 的 `/generate` API）。

**Q3：Scheduler 中 waiting / running / swapped 三个队列分别什么时候用？**

> - **waiting**：新到的 request，尚未分配任何 KV block
> - **running**：已分配 block，正在参与 decode 的 sequence
> - **swapped**：GPU block 不足时被暂时 swap 到 CPU 的 sequence，等 GPU 空闲时换回

---

[← 返回 vLLM 系列索引](00-index.md) | [下一篇：模型加载机制 →](02-model-loading.md)
