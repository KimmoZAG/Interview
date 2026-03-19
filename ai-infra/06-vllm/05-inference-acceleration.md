# vLLM 推理加速技术

## 核心定义（What & Why）

> vLLM 的推理加速不依赖单一技术，而是**多层叠加**：PagedAttention 解决 KV cache 碎片化、连续批处理提升 GPU 利用率、CUDA Graphs 消除 kernel launch 开销、前缀缓存减少重复计算、投机解码和 chunked prefill 优化延迟——每层技术针对不同瓶颈，可按需叠加。

核心要点：

- **PagedAttention**：以固定大小 block 管理 KV cache，消除碎片，支撑高并发
- **连续批处理（Continuous Batching）**：per-token 调度，不浪费 decode slot
- **CUDA Graphs**：冻结 decode 阶段的计算图，从 µs 级 launch 开销降到纳秒级
- **前缀缓存（Prefix Caching / Radix Cache）**：跨请求复用相同前缀的 KV block
- **投机解码（Speculative Decoding）**：小草稿模型批量预测，大目标模型并行验证
- **Chunked Prefill**：拆分超长 prefill，与 decode 共同调度，控制 TTFT

---

## 技术 1：PagedAttention

### 问题根源

```
传统 KV cache：为每个请求预分配连续显存
  请求 A: max_len=2048 → 立即预分配 2048 个 slot（实际用了 512）
  请求 B: max_len=4096 → 立即预分配 4096 个 slot（实际用了 1024）
  
→ 内部碎片：实际利用率 < 50%
→ 外部碎片：动态到达/完成的请求撕裂显存
```

### 核心机制

```
物理 block（固定大小，默认 16 tokens）
  block_size = 16
  每个 block 存储 16 个 token 的 KV pair

逻辑 → 物理映射（block table）
  请求 A: [block_0: 0-15] [block_1: 16-31] [block_5: 32-47]
  请求 B: [block_2: 0-15] [block_3: 16-31]
  
→ 物理 block 可以非连续
→ block_table 由 BlockSpaceManager 维护
→ attention kernel 通过 block_table 聚合非连续 KV
```

### 效果量化

| 指标 | 传统连续 KV | PagedAttention |
|------|-------------|----------------|
| 内部碎片 | ~50% 浪费 | < 4%（最后一个 block 未满）|
| 外部碎片 | 高（碎片孔洞）| 接近零（block 级分配）|
| 可并发请求数 | 受最大 max_len 限制 | 按实际用量分配 |
| Block 共享 | 不支持 | 支持（Prefix Cache 基础）|

---

## 技术 2：连续批处理（Continuous Batching）

### 与静态批处理的本质区别

```
静态批处理（离线）：
  等待 N 个请求全部完成后再处理下一批
  GPU 等待最长请求 → 短请求结束后 GPU 空跑

连续批处理（per-iteration）：
  每个 step（一次 forward）重新决定 batch 成员
  某个请求完成 → 当 step 结束时立即移出 batch
  等待中的新请求 → 下个 step 立即插入
  
→ GPU 利用率从 ~40-60% 提升到 ~85-95%（高并发在线服务、请求长度混合场景）
```

### Scheduler 决策逻辑（简化）

```python
def schedule(self) -> SchedulerOutputs:
    # 优先处理 running 中的 decode 请求
    running_scheduled = self._schedule_running(budget)

    # 尝试从 swapped 恢复（如有 GPU block 空闲）
    swapped_scheduled = self._schedule_swapped(budget)

    # 从 waiting 队列添加新 prefill 请求
    prefills = self._schedule_prefills(budget)

    return SchedulerOutputs(
        scheduled_seq_groups=prefills + running_scheduled + swapped_scheduled,
        num_prefill_groups=len(prefills),
        ...
    )
```

- `budget`：每 step 最多处理的 token 数（`max_num_batched_tokens`）和最多并发请求数（`max_num_seqs`）
- **prefill 优先 decode**：默认策略；可通过 `preemption_mode` 调整

---

## 技术 3：CUDA Graphs

### 问题：decode 阶段每 step 一个 token，kernel launch 开销占主导

```
decode step：input = [batch_size, 1]（每请求 1 个 token）
  → forward 涉及 100+ kernel calls
  → 每个 kernel launch ≈ 5-20 µs CPU 开销
  → 100 层 × 几个 kernel/层 = 1-2 ms CPU overhead / step
  → 对于 H100 decode 2ms / step，CPU launch 占了近 50%!
```

### 解决方案

```python
# 首次 step：以不同 batch size 录制计算图
# vllm/worker/model_runner.py
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=self.graph_memory_pool):
        output = self.model(input_ids, positions, kv_caches, attn_metadata)
    self.cuda_graph_runner.graphs[batch_size] = graph

# 后续 decode step：直接 replay，不重新 launch kernel
def _execute_model_cuda_graph(self, batch_size, inputs):
    # 原地更新 graph 的输入 tensor（共享显存）
    self.input_ids_buffer[:batch_size].copy_(inputs.input_ids)
    self.positions_buffer[:batch_size].copy_(inputs.positions)
    # Replay：一次 CPU call 触发所有 GPU kernel
    self.cuda_graph_runner.graphs[batch_size].replay()
```

### 限制

- 只适用于 **decode** 阶段（batch size 固定、形状固定）
- **prefill** 阶段 input 长度动态变化，无法录制固定图
- 动态形状操作（如某些量化 kernel）可能与 CUDA Graph 不兼容

---

## 技术 4：前缀缓存（Prefix Caching / Radix Cache）

### 应用场景

```
系统 prompt 复用：
  10000 个请求共享同一个 4096-token system prompt
  → 若每次重算：4096 × 10000 = 4096万 token 的 prefill
  → 若缓存前缀 KV：只算一次，后续直接复用

多轮对话：
  用户历史对话（前 N 轮）KV 可以跨 turn 复用
```

### Radix Cache 实现

```
BlockSpaceManagerV2 维护一棵前缀树（Radix Tree）
  tree node = token 序列片段
  每个 node 关联若干 KV blocks

新请求到达时：
  1. 计算 prompt token 序列的 hash
  2. 在 radix tree 中做最长前缀匹配
  3. 匹配的部分：直接引用已有 block（引用计数 +1）
  4. 未匹配的部分：正常 prefill + 分配新 block + 插入树

Block 淘汰：
  LRU（最近最少使用）策略
  只有引用计数为 0 的 block 可被淘汰
```

### 命中率影响

| 场景 | 典型命中率 | 效果 |
|------|------------|------|
| 固定 system prompt + 高并发 | 95%+ | TTFT 降低 80%+ |
| 多轮对话（5 轮以上）| 60-80% | 显著降低中后期 turn 延迟 |
| 完全随机 prompt | < 5% | 基本无收益 |

---

## 技术 5：投机解码（Speculative Decoding）

### 原理

```
目标模型（大）：70B，每 step 生成 1 token，慢
草稿模型（小）：1B，每 step 生成 1 token，快

投机解码流程：
  1. 草稿模型自回归生成 γ 个 token（例如 γ=4）
  2. 目标模型并行验证这 4 个 token + 生成第 5 个
     → 一次 forward 验证多个候选
  3. 接受规则（无损）：
     - 若目标分布 P(t) ≥ 草稿分布 Q(t)：接受
     - 否则以概率 P(t)/Q(t) 接受，否则拒绝并重采样
  4. 平均接受率 α ≈ 0.7-0.8，平均每 step 产出 3 个 token

吞吐提升估算：
  加速比 = α × γ / (1 + α × γ × latency_draft / latency_target)
  # α = 草稿 token 被目标模型接受的平均概率（0~1）
  # γ = 每次草稿模型生成的候选 token 数（超参）
```

### vLLM 的投机解码实现

```python
LLM(
    model="meta-llama/Llama-3-70b",
    speculative_model="meta-llama/Llama-3-1b",  # 草稿模型
    num_speculative_tokens=5,                    # γ
    speculative_draft_tensor_parallel_size=1,    # 草稿模型 TP
)
```

也支持"无草稿模型"的变体：

- **n-gram speculation**：用 prompt 中出现的 n-gram 作为候选 token
- **MLPSpeculator**：用轻量 MLP 头预测下一 token（Eagle 思路）

---

## 技术 6：Chunked Prefill

### 问题

```
超长 prefill（如 32k tokens）垄断 GPU 一整个 step
→ 期间所有 decode 请求被阻塞
→ decode 请求 TTFT（time to first token）激增
```

### 解决方案

```python
# 开启 chunked prefill
LLM(
    model="...",
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,  # 每 step 最多处理 token 数
)

# Scheduler 将长 prefill 切成 ≤ chunk_size 的块
# 每 step 混合执行：
#   部分 prefill chunk + 全量 decode tokens
```

### 权衡

| 配置 | TTFT | TPOT | GPU 利用率 |
|------|------|------|------------|
| 不开 chunked prefill | 长 prefill 请求低，短 decode 被阻塞高 | 稳定 | 高 |
| 开 chunked prefill（chunk_size=4096）| 均衡 | 略微增加（调度开销）| 高 |
| chunk_size 太小（256）| 降低 | 显著增加 | 中（调度频繁）|

---

## 加速技术叠加效果

| 技术 | 主要改善指标 | 适用场景 |
|------|-------------|----------|
| PagedAttention | 并发请求数、内存利用率 | 所有场景，默认开启 |
| 连续批处理 | 吞吐量（tokens/s）| 高并发在线服务 |
| CUDA Graphs | decode TPOT 延迟 | decode-heavy 负载 |
| 前缀缓存 | 系统 prompt 场景 TTFT | 共享前缀、多轮对话 |
| 投机解码 | 延迟（端到端 tokens/s）| 低批量大小、latency-sensitive |
| Chunked Prefill | TTFT p99、延迟公平性 | 混合长短请求 |

---

## 关联知识网络

**前置**：[vLLM 架构总览](01-architecture-overview.md)、[Paged KV & Allocator](../02-inference-engine/07-paged-kv-and-allocator.md)、[LLM Serving](../02-inference-engine/04-llm-serving.md)

**平行**：[算子替换与定制](04-custom-ops-and-operator-replacement.md)、[FlashAttention IO-aware](../01-operator-optimization/06-flashattention-io-aware.md)

**延伸**：[长上下文 Serving](../02-inference-engine/08-long-context-serving.md)、[分布式通信](../04-communication/04-collectives.md)

---

## 💥 实战踩坑记录

**现象**：开启 CUDA Graphs 后特定 batch size 下 inference 结果不正确（而非 OOM 或 crash）。

**根因**：CUDA Graph 录制时 `attn_metadata` 的某个字段是动态计算的（如 `seq_lens_tensor`），录制后被固化为录制时的值，replay 时未更新该字段。

**修复**：确保所有需要每 step 更新的 tensor 使用 `copy_()` 原地更新（而非重新赋值），保证 graph capture 的内存地址与 replay 时一致。

---

**现象**：开启前缀缓存后 OOM，但关闭后正常。

**根因**：Radix Cache 中的 block 引用计数导致 LRU 无法及时淘汰热前缀 block（即使 block 理论上可以淘汰，但因为有 in-flight 请求引用而无法释放）。

**修复**：监控 `num_used_gpu_blocks / num_total_gpu_blocks`，若持续 > 0.9 则考虑降低并发或增大 `gpu_memory_utilization`；也可设置 `--max-num-seqs` 限制并发上限。

---

## 🎯 面试高频 Q&A

**Q1：PagedAttention 和操作系统的虚拟内存分页有哪些相似性和差异？**

> **相似**：都用固定大小 page（block）管理物理内存，逻辑地址（sequence position）→ 物理地址（GPU block）映射，支持 "copy-on-write"（beam search fork），以及 LRU 换页（swap to CPU）。**差异**：OS 有 MMU 硬件支持，Page fault 透明；vLLM 无硬件支持，block table 需软件维护且在每次 attention kernel 调用时显式传入；OS 换页是字节级，vLLM swap 是 KV block 级（每 block 可达数 MB）。

**Q2：投机解码理论上无损（exact same distribution），为什么实际效果有时比预期差？**

> 理论无损依赖于目标模型和草稿模型都是确定性采样（greedy）或接受规则精确实现。实际问题：① 草稿模型和目标模型的 tokenizer 或 precision 不完全一致，导致概率计算误差；② 当 temperature 较低时，接受率高但收益已经很大，当 temperature=0（greedy）接受率最高；③ batch 较大时，草稿模型本身的延迟不可忽略，需要精确 benchmark 确认有收益才开启。

**Q3：CUDA Graphs 和 torch.compile 在 vLLM 中的关系是什么？**

> 两者解决的层次不同。**CUDA Graphs** 是 GPU runtime 特性，将 CPU launch kernel 的序列录制成单次 replay，消除 Python/CPU overhead。**torch.compile** 是 Python 层的 JIT 编译，通过 TorchInductor 做图优化、kernel fusion、自动 codegen。vLLM 默认用 CUDA Graphs，`torch.compile` 处于实验阶段（需 `--enforce-eager=False` + `--use-v2-block-manager`），两者可叠加但兼容性需验证。

---

[← 算子替换与定制](04-custom-ops-and-operator-replacement.md) | [返回 vLLM 索引](00-index.md) | [← 返回 AI Infra 总索引](../00-index.md)
