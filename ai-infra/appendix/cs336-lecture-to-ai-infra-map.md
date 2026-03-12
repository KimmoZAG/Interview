# CS336 逐讲映射：从课程到 AI Infra 笔记

## 要点

- 这份文档不是课程摘要，而是把 **CS336 每一讲在练什么**、**它对应你知识库里的什么位置**、**还缺什么** 直接对齐起来。
- 如果你的目标偏 AI Infra / 底层原理 / 面试，这张映射表比单纯收藏 lecture notes 更有用，因为它强迫你把课程内容落到 **张量形状、访存、kernel、并行、推理调度、数据工程、post-training** 这些工程对象上。
- 一门课真正能沉淀下来的，不是“我看过哪些视频”，而是“我能把哪几个系统问题讲清楚、画清楚、跑清楚”。

## 怎么用这张图

推荐每学一讲，都做下面 4 件事：

1. 先看官方 lecture 或 trace，建立原始语境。
2. 再看 1 份系列型笔记，验证自己是否理解到位。
3. 对照这套知识库里对应的主题笔记，把概念落到工程对象。
4. 写出 3 到 5 条“这讲真正改变了我什么工程判断”的结论。

一个简单标准：如果某一讲看完后，你仍然说不清“这讲会改变我调模型/调系统的什么动作”，那说明还停留在阅读层，没有进入吸收层。

## 总体能力地图

从 CS336 的角度看，一名真正具备大模型底层能力的工程师，至少要能跨 5 层思考：

| 层次 | 核心问题 | 课程对应 |
|---|---|---|
| 表示层 | 文本如何变成 token，token 如何变成张量 | Lecture 1-3 |
| 计算层 | 模型里真正的热点算子是什么 | Lecture 3, 5, 6 |
| 系统层 | GPU、kernel、并行、通信如何限制吞吐 | Lecture 5-8 |
| 规模层 | 在固定预算下模型和数据如何取舍 | Lecture 9, 11 |
| 生产层 | 推理、评测、数据、对齐如何形成完整 pipeline | Lecture 10, 12-17 |

你这套 AI Infra 笔记现在已经把课程主线里的模型、系统、推理、数据、对齐核心块大体补齐，剩下更多是继续向专题深挖，而不是“有没有覆盖到”。

## 逐讲映射

### Lecture 1: Overview and Tokenization

这讲的真正重点不是 BPE 细节，而是课程的思维方式：**把资源、模型、数据、推理、对齐看成同一条链路**。

你应该从这讲拿走的 3 个判断：

- tokenization 会直接改变序列长度，因此直接改变训练和推理成本
- 很多对 frontier model 有用的知识，并不是某个具体 trick，而是对资源和瓶颈的判断方式
- “小模型实验推大模型设计”是整门课的隐藏主线

对应笔记：

- [../models/04-tokenization-and-sampling.md](../models/04-tokenization-and-sampling.md)
- [../models/01-transformer-minimum.md](../models/01-transformer-minimum.md)
- [cs336-from-scratch-resource-map.md](cs336-from-scratch-resource-map.md)

建议补强：

- 增加一节“token compression ratio 与训练成本”的显式推导
- 增加 BPE 在工程上影响 batch size / context / prefill 成本的例子

### Lecture 2: PyTorch and Resource Accounting

这讲对 AI Infra 很关键，因为它把“会写模型”推进到“会算资源账”。

你应该能回答：

- 参数量、激活、梯度、优化器状态分别占多少内存
- 为什么训练时显存结构和推理时完全不同
- 为什么不同优化器会显著改变内存预算

对应笔记：

- [../models/06-training-resource-accounting.md](../models/06-training-resource-accounting.md)
- [../operators/03-memory-hierarchy-and-roofline.md](../operators/03-memory-hierarchy-and-roofline.md)
- [../inference/05-optimization-playbook.md](../inference/05-optimization-playbook.md)

建议补强：

- 把 activation checkpointing、mixed precision 与资源核算的 trade-off 再补成例子

### Lecture 3: Architectures and Hyperparameters

这讲是“最小 Transformer 知识”到“工程上怎么选型”的桥梁。

你应该能说清：

- hidden size、num heads、d_ff、num layers 分别影响什么
- 为什么某些结构决定算子热点会落在 attention，某些结构则更多落在 MLP
- pre-norm、RMSNorm、SwiGLU、RoPE 这类设计为什么成为现代 LLM 标配

对应笔记：

- [../models/01-transformer-minimum.md](../models/01-transformer-minimum.md)
- [../models/03-norm-activation-stability.md](../models/03-norm-activation-stability.md)

建议补强：

- 增加一节“超参数如何映射到 FLOPs / memory / throughput”
- 增加一张现代 decoder-only LLM block 的结构图

### Lecture 4: Mixture of Experts

这讲在面试里不一定高频，但在“计算预算不变时如何扩模型容量”上很关键。

你真正该学的是：

- 稀疏激活并不等于系统简单，反而会带来路由与负载均衡问题
- MoE 是“算法扩展”和“系统复杂度”之间的典型 trade-off

对应笔记：

- [../models/08-moe-minimum.md](../models/08-moe-minimum.md)

建议补强：

- 后续可以补 dense vs MoE 的实验对照和 expert parallel 的通信代价模型

### Lecture 5: GPUs

这讲对 AI Infra 是核心中的核心。你不需要会写所有 CUDA，但必须知道瓶颈为什么出现。

你应该能说清：

- GPU 的寄存器、shared memory、L2、HBM 分工是什么
- 为什么同样一个算子，在不同 shape 下可能从算力瓶颈切换到带宽瓶颈
- 为什么 launch overhead、occupancy、memory coalescing 会影响真实表现

对应笔记：

- [../operators/02-kernels-and-parallelism.md](../operators/02-kernels-and-parallelism.md)
- [../operators/03-memory-hierarchy-and-roofline.md](../operators/03-memory-hierarchy-and-roofline.md)

建议补强：

- 在 roofline 文档里加入一个“GPU memory hierarchy 与常见优化动作”的对照表

### Lecture 6: Kernels, Triton

这讲的重点不是“会写 Triton 语法”，而是理解为什么高层算子有时必须落到自定义 kernel。

关键吸收点：

- kernel fusion 本质是在减少中间访存和 launch overhead
- FlashAttention 之所以重要，不只是更快，而是改变了 attention 的内存访问方式
- Triton 的价值在于让你能在比 CUDA 更高的层次控制 tile 和访存模式

对应笔记：

- [../operators/04-graph-fusion-scheduling.md](../operators/04-graph-fusion-scheduling.md)
- [../operators/03-memory-hierarchy-and-roofline.md](../operators/03-memory-hierarchy-and-roofline.md)
- [../operators/07-flashattention-io-aware.md](../operators/07-flashattention-io-aware.md)

建议补强：

- 后续可以补一张 naive attention vs flash attention 的访存路径对照图

### Lecture 7-8: Parallelism

这是把“单卡训练”推进到“系统工程”的分水岭。

你应该能区分：

- data parallelism 解决的是什么问题
- tensor parallelism / pipeline parallelism / sequence parallelism 适用于什么瓶颈
- 为什么通信拓扑和梯度同步会成为训练速度上限

对应笔记：

- [../operators/02-kernels-and-parallelism.md](../operators/02-kernels-and-parallelism.md)
- [../operators/06-training-parallelism.md](../operators/06-training-parallelism.md)
- [../inference/01-inference-stack-overview.md](../inference/01-inference-stack-overview.md)

建议补强：

- 把 FSDP / ZeRO 与 TP / PP 的组合方式再补一张对照表

### Lecture 9: Scaling Laws I

这讲非常容易被读成“论文史”，但真正有价值的是：**如何用小规模实验决定大规模预算**。

你应该能说明：

- 为什么 scaling law 是预算规划工具，而不是纯理论曲线
- 在固定 FLOPs 下，模型大小和 token 数如何做取舍
- 为什么 Chinchilla optimality 不能直接等同于全生命周期最优

对应笔记：

- [../inference/07-scaling-laws-and-budgeting.md](../inference/07-scaling-laws-and-budgeting.md)

建议补强：

- 把 scaling law 与真实吞吐、训练时长、数据准备能力进一步串起来

### Lecture 10: Inference

这讲和你现有 AI Infra 目录最直接相关。

真正关键的点：

- prefill 和 decode 是两种完全不同的计算负载
- decode 往往更 memory-bound，更容易受 launch、cache、batching 影响
- 推理系统的核心不是“跑一个 forward”，而是调度、缓存和隔离

对应笔记：

- [../inference/04-llm-serving.md](../inference/04-llm-serving.md)
- [../models/02-attention-kv-cache.md](../models/02-attention-kv-cache.md)
- [../inference/06-observability-and-debugging.md](../inference/06-observability-and-debugging.md)
- [../inference/11-paged-kv-and-allocator.md](../inference/11-paged-kv-and-allocator.md)
- [../inference/12-long-context-training-and-serving.md](../inference/12-long-context-training-and-serving.md)

建议补强：

- 后续可以补 prefill / decode 的长度分桶实验模板

### Lecture 11: Scaling Laws II

这讲的价值在于把 scaling law 从“概念”推进到“怎么真正拿来做实验设计”。

你应该能回答：

- 小模型上找到的最优点，哪些能迁移，哪些不能迁移
- 学习率调度、参数化方式、数据质量变化，会如何污染外推结果

对应笔记：

- [../inference/07-scaling-laws-and-budgeting.md](../inference/07-scaling-laws-and-budgeting.md)

建议补强：

- 后续可以补一节 muP / 调度器 / 数据质量如何影响外推稳定性

### Lecture 12: Evaluation

这讲会直接决定你是否会做“靠谱 benchmark”。

你应该能区分：

- perplexity、benchmark accuracy、人工偏好、线上指标分别回答什么问题
- 为什么离线评测和线上效果经常错位
- 为什么评测必须和部署目标绑定

对应笔记：

- [../models/05-evaluation-and-benchmarking.md](../models/05-evaluation-and-benchmarking.md)
- [../inference/09-training-metrics-vs-product-metrics.md](../inference/09-training-metrics-vs-product-metrics.md)

建议补强：

- 后续可以补更细的线上实验设计、A/B 与 shadow evaluation 模板

### Lecture 13-14: Data

这两讲是很多学习资料最薄弱、但工业价值最高的部分。

真正该学会的是：

- 数据质量不是一个抽象名词，而是一系列可实现的过滤、解析、去重流水线
- 数据工程会直接影响下游 loss、泛化和对齐效果
- 数据处理不是“采样前的清洁工作”，而是模型能力塑形的一部分

对应笔记：

- [../inference/08-pretraining-data-engineering.md](../inference/08-pretraining-data-engineering.md)
- [../inference/10-data-mixing-and-curriculum.md](../inference/10-data-mixing-and-curriculum.md)

建议补强：

- 后续可以补更细的去重算法与数据质量实验模板

### Lecture 15-17: Alignment / RL

这部分的价值不在于记住所有算法名词，而在于理解：预训练模型为什么还不够，后训练究竟在改变什么。

你应该能说清：

- SFT、preference optimization、RL-based post-training 各自改什么
- 为什么 alignment 既是行为塑形，也是分布迁移
- 为什么评测和 reward 设计会反向约束整个后训练方案

对应笔记：

- [../models/07-post-training-and-alignment.md](../models/07-post-training-and-alignment.md)
- [../models/09-reward-and-verifier-design.md](../models/09-reward-and-verifier-design.md)

建议补强：

- 后续可以补 test-time compute、self-consistency 与 verifier 的组合方式

## 面向 AI Infra 的精读顺序

如果你的目标是“尽可能深，但仍然服务于底层工程能力”，推荐按下面顺序精读，而不是完全按课程发布时间线：

### 第一阶段：模型与资源直觉

优先讲次：Lecture 1, 2, 3

要拿下的问题：

- tokenization 怎么影响成本
- Transformer block 的主要算子是什么
- 训练时显存究竟花在哪

### 第二阶段：GPU 与算子

优先讲次：Lecture 5, 6

要拿下的问题：

- 为什么某些优化动作是在“救带宽”，某些是在“救算力”
- 为什么 attention 需要特化 kernel

### 第三阶段：推理与 serving

优先讲次：Lecture 10

要拿下的问题：

- prefill / decode 的系统差异
- batching / KV cache / tail latency 之间的 trade-off

### 第四阶段：并行与规模

优先讲次：Lecture 7, 8, 9, 11

要拿下的问题：

- 多卡训练为什么难
- scaling laws 如何用于预算规划，而不是只背结论

### 第五阶段：评测、数据、对齐

优先讲次：Lecture 12-17

要拿下的问题：

- 什么叫“模型好”
- 数据为什么和模型架构一样重要
- post-training 为什么是独立系统问题

## 还可以继续往哪挖

如果还要继续深化，下一批更适合做“专题深挖”而不是“补空白”：

1. prefill / decode 长度分桶实验模板
2. dense vs MoE 的实验设计与通信账本
3. 数据质量变化如何污染 scaling law 外推
4. test-time compute、verifier、采样策略联动
5. 长上下文下 packing / cache / 调度联动图

## 自检 checklist

- [ ] 我能把 CS336 17 讲按“表示层 / 计算层 / 系统层 / 规模层 / 生产层”重新分组。
- [ ] 我知道自己这套笔记目前在哪些主题上已经覆盖，哪些仍然空缺。
- [ ] 我能把 Lecture 10 的 inference 内容直接映射到 batching、KV cache、TTFT、TPOT。
- [ ] 我能解释为什么 Lecture 2、5、6、7、8 对 AI Infra 尤其重要。
- [ ] 我已经决定下一篇要补的是哪一个空缺主题，而不是继续无序收藏链接。

## 参考资料

- 官方课程主页：https://cs336.stanford.edu/
- 官方 2025 存档：https://cs336.stanford.edu/spring2025/
- 官方 lectures repo：https://github.com/stanford-cs336/spring2025-lectures
- 资源导读总表：[cs336-from-scratch-resource-map.md](cs336-from-scratch-resource-map.md)