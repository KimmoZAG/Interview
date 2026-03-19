# DeepSeek 的部署与在线推理：为什么必须拆开 Prefill 和 Decode

## 关键结论

DeepSeek 的在线部署重点，从来不是“模型能不能跑起来”，而是：**在超大 MoE 上同时守住服务延迟目标和总体吞吐。**

- `prefill / decode` 必须拆开部署，因为两阶段的算子形态、延迟目标和资源瓶颈根本不同 [DeepSeek-V3, Section 3.4]。
- `prefill` 更偏高计算密度和吞吐导向；`decode` 更偏低延迟、访存和通信敏感 [DeepSeek-V3, Sections 3.4.1-3.4.2]。
- 训练期的 balance 机制不够覆盖真实线上流量，因此 V3 还引入了 `redundant experts` 处理 serve-time 热点专家问题 [DeepSeek-V3, Sections 3.4.1-3.4.2]。
- `MLA` 解决的是状态体积问题，而 V3 的 serving 设计解决的是“超大 MoE 在线路径到底怎么组织”这个问题 [DeepSeek-V2, Section 3.2.3; DeepSeek-V3, Section 3.4]。

所以这一页最重要的结论是：**DeepSeek 的服务化关键，不是“推理更快”四个字，而是承认 prefill 和 decode 是两类完全不同的系统任务，然后分别为它们设计部署单元、并行策略和负载治理机制。**

## 背景：为什么在线推理不能再被当成训练系统的简化版

### 旧的理解为什么不够

很多模型的推理部署，常常被理解成训练系统的缩水版：

- 训练时要并行；
- 推理时也要并行；
- 只不过 batch 更小，目标从 loss 变成 latency。

但对 DeepSeek 这类 `MLA + MoE` 大模型来说，这种理解已经不够用了。因为线上推理至少同时面对三类新问题：

- `prefill` 和 `decode` 的算子形态完全不同；
- 真实请求流量会把热点专家问题重新放大；
- 即使训练时通信已经被优化过，线上系统仍然要继续处理 all-to-all、访存和尾延迟。

也就是说，**训练系统能跑通，不等于线上系统就会自然优雅。**

### 这一页真正想解决什么

这一页主要想讲清楚四件事：

1. 为什么 prefill 和 decode 必须拆开看；
2. DeepSeek-V3 分别为这两个阶段设计了什么部署形态；
3. `redundant experts` 到底在解决什么 serve-time 问题；
4. V2 到 V3 的服务化思路是怎样升级的。

## DeepSeek 具体怎么做

### 第一步：先承认 Prefill 和 Decode 不是同一种任务

在 `prefill` 阶段，系统面对的是整段 prompt：

- attention 计算更重；
- token 数量更大；
- 更适合用更高计算密度去吃满硬件；
- 通信虽然仍然贵，但更容易被大计算块掩盖。

而在 `decode` 阶段，情况几乎反过来：

- 每步只生成少量 token；
- attention 不再是 full quadratic，但需要频繁读取历史状态；
- 单个 expert 的 batch size 变小；
- 瓶颈更容易变成 memory access 与通信延迟 [DeepSeek-V3, Section 3.4.2]。

这意味着，如果把两者硬塞进同一套 deployment unit，就会出现典型的系统错配：

- 为 prefill 优化的配置，会把 decode 延迟拖高；
- 为 decode 优化的配置，又会让 prefill 吞吐浪费。

所以 DeepSeek 的第一个关键判断非常朴素，但很重要：**生成过程要分阶段治理，而不是假装它们是同一种负载。**

### 第二步：给 Prefill 配一套更偏吞吐的部署单元

DeepSeek-V3 对 prefilling 的部署思路非常明确 [DeepSeek-V3, Section 3.4.1]：

- 最小部署单元为 `4 nodes / 32 GPUs`；
- attention 采用 `TP4 + SP + DP8`；
- MoE 部分采用 `EP32`。

这套配置的直觉是：

- 适度使用 TP，把 attention 算快；
- 让 expert 侧 batch 维持在比较高效的区间；
- 把 prefill 做成更偏吞吐的高密度计算阶段。

换句话说，prefill 的目标不是最低单 token 延迟，而是**尽量高效地把整段 prompt 编成后续 decode 可用的状态。**

### 第三步：给 Decode 配一套更偏低延迟和热点治理的部署单元

到了 decode，DeepSeek-V3 直接换了一套系统组织方式 [DeepSeek-V3, Section 3.4.2]：

- 最小部署单元扩大到 `40 nodes / 320 GPUs`；
- attention 采用 `TP4 + SP + DP80`；
- MoE 采用 `EP320`；
- 每个 GPU 仅承载 `1` 个 expert；
- 另有 `64` 个 GPUs 负责 `redundant experts` 与 `shared experts`。

这个配置看起来更“夸张”，但它解决的是 decode 阶段真正的难点：

- 单步延迟更敏感；
- 热点专家更容易把尾延迟拖垮；
- all-to-all 路径需要更短、更稳；
- serve-time 负载分布比训练期更容易偏斜。

所以 DeepSeek 在 decode 阶段的重点已经不再是“每个 expert 吃大 batch”，而是：**路径更短、热点更散、单步更稳。**

### 第四步：用 redundant experts 解决 serve-time 热点问题

这是 V3 serving 里非常有代表性的一个思路：训练期 balance 不够，线上还要做一层新的负载治理。

`redundant experts` 的本质就是：

- 根据在线服务统计，检测高负载 experts；
- 周期性调整热点专家集合；
- 把最常被命中的 experts 做额外副本复制；
- 用更多资源换更稳的尾延迟和更低的热点压力 [DeepSeek-V3, Sections 3.4.1-3.4.2]。

这一步很说明 DeepSeek 的系统观：**train-time balance 和 serve-time balance 是两回事，不能假设前者天然覆盖后者。**

### 第五步：继续把通信尽量藏到计算后面

DeepSeek 没有把 overlap 只当成训练技巧。在线服务里，它同样继续做：

- 双 micro-batch overlap；
- dispatch / combine 与 attention 或 MoE 计算重叠；
- 在 decode 里用 point-to-point IB 传输和 `IBGDA` 压低延迟 [DeepSeek-V3, Sections 3.4.1-3.4.2]。

系统意义很简单：通信不会因为上线而消失，真正的目标是**让通信更少直接表现成用户可感知的等待时间。**

### 这套 serving 路线带来的直接优点

把 DeepSeek 的在线部署路线压缩一下，收益主要是：

- **阶段分治更清楚**：prefill 追吞吐，decode 追低延迟；
- **热点问题更可控**：serve-time 用 redundant experts 做额外缓冲；
- **在线路径更稳**：通信继续被尽量重叠和缩短；
- **架构收益更容易兑现**：MLA、MoE 和系统调度终于能在线上一起成立。

## 数据怎么说明这些优点

### 证据一：V2 已经证明“把 attention 状态压轻”能直接换来部署吞吐

DeepSeek-V2 已经给出很强的服务侧结果：

- 单节点 `8 x H800` 上 generation throughput 超过 `50K tokens/s`；
- prompt throughput 超过 `100K tokens/s`；
- 最大 generation throughput 达到 DeepSeek 67B 的 `5.76×` [DeepSeek-V2, Section 3.2.3]。

这说明 V2 已经解决了服务化的一层关键前提：**attention 状态不能太贵。**

### 证据二：V3 把 serving 组织方式本身写成主设计

到了 V3，论文不只是说“还能推理”，而是明确给出：

- prefill 的最小部署单元；
- decode 的最小部署单元；
- `redundant experts`；
- point-to-point IB + `IBGDA`；
- 双 micro-batch overlap [DeepSeek-V3, Section 3.4]。

这说明 V3 把在线推理正式当成了系统设计问题，而不是训练成功后的附带环节。

### 证据三：V3 还把推理加速继续往训练目标侧接回去

V3 还通过 `MTP speculative decoding` 继续提升 decoding speed，可达到约 `1.8× TPS`；端到端 generation speed 也超过 DeepSeek-V2 的 `2×` [DeepSeek-V3, Section 5.4.3; Conclusion / Limitations]。

这说明 DeepSeek 的思路不是把“训练优化”和“服务优化”割裂开来，而是继续让训练目标为服务效率留后手。

## 思考问题

- 如果你的服务资源有限，你会优先保 prefill 吞吐，还是 decode 延迟？为什么？
- `redundant experts` 更像训练期 balance 的延续，还是一种完全独立的 serve-time 机制？
- 在 MoE 在线服务里，最容易被低估的问题是什么：热点专家、all-to-all 延迟，还是 prefill/decode 共用同一部署单元？
