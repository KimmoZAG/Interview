# AI Infra 重构路线图

目标：把当前以 `operators / models / inference / appendix` 为主的资料库，重构为更适合 **推理优化方向学习、复习和面试** 的五大板块：

1. 算子优化
2. 推理引擎
3. 大模型架构
4. 通信技术
5. 附录

---

## 一、重构原则

### 1. 主线优先

优先保证“推理优化主线”清晰，不追求一次性覆盖所有背景知识。

### 2. 一章只回答一个核心问题

避免一篇同时讲模型、kernel、runtime、通信，导致边界模糊。

### 3. 以面试可回答为标准

每篇最终都应支持：

- 2 分钟讲清楚核心
- 能给出最小例子
- 能回答 trade-off
- 能做基本问题定位

### 4. 统一模板

主线章节统一采用：

- 一页结论
- 通用知识
- 例子
- 常见面试问题
- 易错点
- 排查 checklist

---

## 二、优先级排序

### P0：先改入口，不急着大搬正文

- 改 `README.md`
- 改 `00-index.md`
- 建立 5 个新板块索引
- 告诉读者“新结构已经启用，旧目录为待迁移内容池”

### P1：补通信技术板块

这是当前最大短板。

建议优先新增：

1. 通信基础：latency / bandwidth / topology
2. 互联硬件：PCIe / NVLink / NVSwitch / InfiniBand / RDMA
3. Collectives：all-reduce / all-gather / reduce-scatter / all-to-all
4. 并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO

### P2：拆分边界不清的旧文档

优先拆：

- `operators/02-kernels-and-parallelism.md`
- `inference/11-paged-kv-and-allocator.md`
- `inference/12-long-context-training-and-serving.md`
- `models/02-attention-kv-cache.md`

### P3：统一章节模板

优先覆盖 10 篇核心章节：

- Transformer 最小知识
- Attention 与 KV cache
- Roofline
- FlashAttention
- LLM Serving
- KV Cache / Paged KV
- Allocator / Memory Pool
- Optimization Playbook
- Observability / Debugging
- Training Parallelism / Communication

---

## 三、旧目录到新板块映射

### 1. 算子优化

- `operators/01-tensors-shapes-layout.md`
- `operators/02-kernels-and-parallelism.md`（保留单卡执行模型部分）
- `operators/03-memory-hierarchy-and-roofline.md`
- `operators/04-graph-fusion-scheduling.md`
- `operators/05-quantization-basics.md`
- `operators/07-flashattention-io-aware.md`

### 2. 推理引擎

- `inference/01-inference-stack-overview.md`
- `inference/02-onnxruntime-tensorrt.md`
- `inference/03-graph-compiler-tvm-mlir-xla.md`
- `inference/04-llm-serving.md`
- `inference/05-optimization-playbook.md`
- `inference/06-observability-and-debugging.md`
- `inference/11-paged-kv-and-allocator.md`
- `inference/12-long-context-training-and-serving.md`（serving 部分）

### 3. 大模型架构

- `models/01-transformer-minimum.md`
- `models/02-attention-kv-cache.md`
- `models/03-norm-activation-stability.md`
- `models/04-tokenization-and-sampling.md`
- `models/05-evaluation-and-benchmarking.md`
- `models/08-moe-minimum.md`
- `models/06-training-resource-accounting.md`（选修）
- `inference/12-long-context-training-and-serving.md`（模型侧部分）

### 4. 通信技术

- `operators/06-training-parallelism.md`
- `models/06-training-resource-accounting.md`（通信 buffer / sharding 部分）
- `models/08-moe-minimum.md`（expert parallel / token dispatch 部分）
- `inference/12-long-context-training-and-serving.md`（长序列并行与扩展性问题）

### 5. 附录

- `appendix/*`
- `inference/07-scaling-laws-and-budgeting.md`
- `inference/08-pretraining-data-engineering.md`
- `inference/09-training-metrics-vs-product-metrics.md`
- `inference/10-data-mixing-and-curriculum.md`
- `models/07-post-training-and-alignment.md`
- `models/09-reward-and-verifier-design.md`

---

## 四、建议新增的章节

### 算子优化

- `04-gemm-reduction-pointwise.md`
- `08-operator-profiling-playbook.md`

### 推理引擎

- `02-runtime-landscape.md`
- `05-kv-cache-layout.md`
- `06-allocator-and-memory-pool.md`

### 大模型架构

- `07-long-context-model-design.md`

### 通信技术

- `01-communication-basics.md`
- `02-interconnects-and-topology.md`
- `03-collectives.md`
- `04-parallelism-to-communication.md`
- `06-overlap-and-bucketing.md`
- `07-multi-gpu-inference.md`
- `08-communication-debugging.md`

---

## 五、推荐执行顺序

1. 先按新目录读和维护，不再继续扩写旧索引
2. 新写章节优先放到 4 个新主板块目录中
3. 老文章逐步重写时，再决定是否物理迁移或拆分
4. 每完成一篇主线文档，就补齐“例子 + 面试题”

---

## 六、完成标准

当满足下面条件时，可以认为重构基本完成：

- 根目录入口全部切换到 5 板块
- 每个主板块都有稳定的索引页
- 通信技术不再只是分散片段
- 核心主线文章都采用统一模板
- 面试复习时可以直接按 5 板块顺序过一遍
