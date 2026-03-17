---
tags:
  - CS336
  - 课程笔记
  - LLM
  - 训练
description: Stanford CS336 《从零构建语言模型》课程中文笔记索引，覆盖训练、推理、对齐全链路。
---

# CS336 中文课程索引

这份索引页的目标不是简单列出 17 讲，而是先帮你建立一张完整脑图：

- 这门课到底在讲什么；
- 每一讲在整条链路里的位置是什么；
- 如果你的目标是面试、工程或研究，应该怎么挑着读。

配套入口：

- 系列说明： [README.md](README.md)
- 学习路线图： [study-roadmap.md](study-roadmap.md)
- 面试准备指南： [interview-prep-guide.md](interview-prep-guide.md)

## 一、CS336 的主线到底是什么

CS336 不是单独讲 Transformer，也不是单独讲 GPU。它真正想训练的是一种端到端视角：

```text
文本离散化 -> 模型结构 -> 训练资源 -> 单卡系统 -> 多卡并行 -> 规模规律 -> 推理部署 -> 数据工程 -> 对齐优化
```

所以这门课可以被拆成五条主线：

1. **基础建模**：Tokenizer、Transformer、训练循环。
2. **系统优化**：GPU、kernel、memory hierarchy、并行训练、通信。
3. **规模规律**：Scaling law、FLOPs 预算、超参外推、实验分配。
4. **数据工程**：数据来源、过滤、去重、评测、混合策略。
5. **对齐与强化学习**：SFT、RLHF、DPO、GRPO、可验证奖励。

## 二、先给自己定一个学习目标

如果你还没决定怎么读，先按目标选路径：

### 面试导向

重点是建立可复述的解释框架，优先读：

`01 -> 02 -> 03 -> 05 -> 07 -> 09 -> 10 -> 12 -> 13 -> 15 -> 16`

### 工程导向

重点是资源约束、系统瓶颈、实现 trade-off，优先读：

`02 -> 05 -> 06 -> 07 -> 08 -> 10 -> 14 -> 16 -> 17`

### 研究导向

重点是规模规律、数据、评测、post-training，优先读：

`01 -> 03 -> 04 -> 09 -> 11 -> 12 -> 13 -> 15 -> 16 -> 17`

如果你希望按节奏推进，可以直接看 [study-roadmap.md](study-roadmap.md)。

## 三、逐讲地图

| 讲次 | 主题 | 在整套课程里的作用 | 你应该抓住什么 |
|---|---|---|---|
| [01](01-overview-and-tokenization.md) | 课程总览与 Tokenization | 建立课程世界观和 tokenizer 直觉 | 为什么 tokenizer 会影响整个训练与推理链路 |
| [02](02-pytorch-and-resource-accounting.md) | PyTorch 与资源核算 | 建立 FLOPs、显存、训练时间的算账能力 | 为什么“先算能不能训得起”比先堆模型更重要 |
| [03](03-architectures-and-hyperparameters.md) | 架构与超参数 | 理解现代 Transformer 默认配置 | RMSNorm、RoPE、SwiGLU、GQA 为什么常见 |
| [04](04-mixture-of-experts.md) | MoE | 理解稀疏激活如何换参数量与系统复杂度 | 为什么 MoE 不是白拿参数，而是白拿问题 |
| [05](05-gpus.md) | GPU 基础 | 建立 roofline、memory hierarchy、tiling 直觉 | 为什么很多模型优化最后都落到数据搬运 |
| [06](06-kernels-and-triton.md) | Kernel 与 Triton | 理解 profiling、fusion、kernel 设计 | 什么叫真正的 kernel 优化，而不是改写语法 |
| [07](07-parallelism.md) | 并行训练 1 | 理解不同并行策略分别在解决什么瓶颈 | DDP、FSDP、TP、PP 怎么选 |
| [08](08-parallelism-part2.md) | 并行训练 2 | 把通信、collectives 和 pipeline 落到实现层 | 为什么扩展性问题经常卡在通信而不是算力 |
| [09](09-scaling-laws-fundamentals.md) | Scaling law 基础 | 学会在固定预算下规划模型与数据 | 参数量、token 数、FLOPs 为什么要联动看 |
| [10](10-inference.md) | 推理优化 | 建立 prefill/decode、KV cache、serving 指标直觉 | 为什么推理系统大量工作都在围绕 KV cache |
| [11](11-scaling-laws-case-studies.md) | Scaling law 案例 | 看真实论文如何做外推与资源分配 | 小实验如何支持大规模决策 |
| [12](12-evaluation.md) | 评测 | 理解 perplexity 与下游 benchmark 的边界 | 为什么好评测不只是“跑几个榜单” |
| [13](13-data.md) | 数据工程 1 | 理解语料来源、版权、质量与能力关系 | 为什么数据决定模型很多上限 |
| [14](14-data-part2.md) | 数据工程 2 | 理解过滤、去重、采样、数据清洗实现 | 为什么数据 pipeline 本身就是核心系统 |
| [15](15-alignment-sft-rlhf.md) | 对齐 1 | 建立 SFT 与 RLHF 的整体框架 | 为什么预训练模型不等于可用助手 |
| [16](16-alignment-rl.md) | 对齐 2 | 理解 DPO、PPO、GRPO 这些方法的关系 | 为什么偏好优化不等于“把奖励喂进去就行” |
| [17](17-alignment-rl-part2.md) | 对齐 3 | 把 policy gradient、baseline、KL 拉回机制层 | 为什么 RL 方法既强大又脆弱 |

## 四、如果你只想抓最核心的 8 讲

如果时间非常有限，我建议优先读下面这 8 讲：

1. [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
2. [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
3. [03-architectures-and-hyperparameters.md](03-architectures-and-hyperparameters.md)
4. [05-gpus.md](05-gpus.md)
5. [07-parallelism.md](07-parallelism.md)
6. [09-scaling-laws-fundamentals.md](09-scaling-laws-fundamentals.md)
7. [10-inference.md](10-inference.md)
8. [15-alignment-sft-rlhf.md](15-alignment-sft-rlhf.md)

这 8 讲几乎覆盖了大模型岗位最常见的基础问法。

## 五、读这套材料时建议一直带着的问题

为了避免把课程读成“概念词典”，建议每讲都追问下面四件事：

1. 这一讲在解决什么瓶颈？
2. 它改善的是 accuracy、throughput、latency 还是 cost？
3. 它的代价落在算力、显存、通信、数据还是工程复杂度上？
4. 这个结论在什么规模区间成立？

## 六、这一页之后怎么读

- 想按目标选路线： [study-roadmap.md](study-roadmap.md)
- 想按面试主题复习： [interview-prep-guide.md](interview-prep-guide.md)
- 想直接开始正文： [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
