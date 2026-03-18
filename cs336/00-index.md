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

## 五、推荐直接按“链路”复习，而不是按讲次死读

如果你已经不是第一次读 CS336，建议把这套材料当成几条可以往返跳转的系统链，而不是 17 讲顺序播放列表。

### 训练资源与系统主线

`[02 资源核算](02-pytorch-and-resource-accounting.md)`
→ `[05 GPU](05-gpus.md)`
→ `[06 Kernel/Triton](06-kernels-and-triton.md)`
→ `[07 并行策略](07-parallelism.md)`
→ `[08 Collectives 与实现](08-parallelism-part2.md)`
→ `[10 推理优化](10-inference.md)`

适合场景：

- 面 AI Infra / 推理优化 / 训练系统岗位
- 想把“FLOPs / 显存 / kernel / 通信 / serving”串成一条逻辑链

### 数据与规模规律主线

`[02 资源核算](02-pytorch-and-resource-accounting.md)`
→ `[09 Scaling Law 基础](09-scaling-laws-fundamentals.md)`
→ `[11 Scaling Law 案例](11-scaling-laws-case-studies.md)`
→ `[13 数据工程（一）](13-data.md)`
→ `[14 数据工程（二）](14-data-part2.md)`

适合场景：

- 面训练平台、数据工程、研究工程混合岗位
- 需要回答“为什么不是盲目堆参数，而是先做资源与数据预算”

### 对齐与产品化主线

`[10 推理优化](10-inference.md)`
→ `[12 评测](12-evaluation.md)`
→ `[15 SFT 与 RLHF](15-alignment-sft-rlhf.md)`
→ `[16 RL](16-alignment-rl.md)`
→ `[17 Policy Gradient](17-alignment-rl-part2.md)`

适合场景：

- 面 LLM 应用、评测、post-training 方向
- 想把“模型好不好用”从推理指标一路讲到对齐与奖励设计

## 六、和 AI Infra 模块怎么互相跳转

如果你希望把课程知识直接映射到工程面试，下面几组桥接最值得反复跳：

| CS336 | 对应 AI Infra | 为什么要连着看 |
|---|---|---|
| [02 资源核算](02-pytorch-and-resource-accounting.md) | [训练资源核算](../ai-infra/03-llm-architecture/07-training-resource-accounting.md) | 一个偏课程抽象，一个偏工程预算 |
| [01 总览与 Tokenization](01-overview-and-tokenization.md) | [Tokenizer 与采样](../ai-infra/03-llm-architecture/04-tokenization-and-sampling.md) | 一个讲课程世界观，一个讲系统成本与复现 |
| [03 架构与超参数](03-architectures-and-hyperparameters.md) | [Transformer 最小知识](../ai-infra/03-llm-architecture/01-transformer-minimum.md) / [Norm/激活/稳定性](../ai-infra/03-llm-architecture/03-norm-activation-stability.md) | 一个讲课程抽象，一个讲系统代价 |
| [04 MoE](04-mixture-of-experts.md) | [MoE 最小导读](../ai-infra/03-llm-architecture/06-moe-minimum.md) / [训练并行策略](../ai-infra/04-communication/01-training-parallelism.md) | 一个讲概念与训练，一个讲通信与落地 |
| [05 GPU](05-gpus.md) | [内存层级与 Roofline](../ai-infra/01-operator-optimization/03-memory-hierarchy-and-roofline.md) | 一个讲直觉，一个讲工程判断 |
| [06 Kernel/Triton](06-kernels-and-triton.md) | [计算图融合与调度](../ai-infra/01-operator-optimization/04-graph-fusion-scheduling.md) | 一个讲 profiling，一个讲系统落地 |
| [07/08 并行](07-parallelism.md) / [08](08-parallelism-part2.md) | [并行训练策略](../ai-infra/04-communication/01-training-parallelism.md) / [Collectives](../ai-infra/04-communication/04-collectives.md) | 一个讲课内框架，一个讲通信成本 |
| [09/11 规模规律主线](09-scaling-laws-fundamentals.md) / [11](11-scaling-laws-case-studies.md) | [训练资源核算](../ai-infra/03-llm-architecture/07-training-resource-accounting.md) / [数据混配与 Curriculum](../ai-infra/05-appendix/data-mixing-and-curriculum.md) | 一个讲外推逻辑，一个讲训练预算与配方约束 |
| [10 推理优化](10-inference.md) | [LLM Serving](../ai-infra/02-inference-engine/04-llm-serving.md) / [优化 Playbook](../ai-infra/02-inference-engine/05-optimization-playbook.md) | 一个讲原理，一个讲系统诊断 |
| [15/16/17 对齐主线](15-alignment-sft-rlhf.md) / [16](16-alignment-rl.md) / [17](17-alignment-rl-part2.md) | [Post-training 与 Alignment](../ai-infra/05-appendix/post-training-and-alignment.md) / [评测与基准](../ai-infra/03-llm-architecture/05-evaluation-and-benchmarking.md) | 一个讲课程机制，一个讲产品化与评测闭环 |

## 七、已完成“实战型重构”的核心页

- [01 总览与 Tokenization](01-overview-and-tokenization.md)
- [02 PyTorch 与资源核算](02-pytorch-and-resource-accounting.md)
- [03 架构与超参数](03-architectures-and-hyperparameters.md)
- [04 Mixture of Experts](04-mixture-of-experts.md)
- [05 GPU 基础](05-gpus.md)
- [06 Kernel 与 Triton](06-kernels-and-triton.md)
- [07 并行训练 1](07-parallelism.md)
- [08 并行训练 2](08-parallelism-part2.md)
- [09 Scaling Law 基础](09-scaling-laws-fundamentals.md)
- [10 推理优化](10-inference.md)
- [11 Scaling Law 案例](11-scaling-laws-case-studies.md)
- [12 评测](12-evaluation.md)
- [13 数据工程 1](13-data.md)
- [14 数据工程 2](14-data-part2.md)
- [15 对齐 1](15-alignment-sft-rlhf.md)
- [16 对齐 2](16-alignment-rl.md)
- [17 对齐 3](17-alignment-rl-part2.md)

## 八、读这套材料时建议一直带着的问题

为了避免把课程读成“概念词典”，建议每讲都追问下面四件事：

1. 这一讲在解决什么瓶颈？
2. 它改善的是 accuracy、throughput、latency 还是 cost？
3. 它的代价落在算力、显存、通信、数据还是工程复杂度上？
4. 这个结论在什么规模区间成立？

## 九、这一页之后怎么读

- 想按目标选路线： [study-roadmap.md](study-roadmap.md)
- 想按面试主题复习： [interview-prep-guide.md](interview-prep-guide.md)
- 想直接开始正文： [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
