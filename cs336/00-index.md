# CS336 中文课程索引

## 课程主线

CS336 可以粗分成五条主线：

1. **基础建模**：Tokenizer、Transformer、训练循环
2. **系统优化**：GPU、Kernel、并行训练、通信
3. **规模规律**：Scaling law、资源预算、超参外推
4. **数据工程**：数据来源、过滤、去重、评测
5. **对齐与强化学习**：SFT、RLHF、DPO、GRPO、可验证奖励

## 逐讲目录

| 讲次 | 主题 | 你会学到什么 |
|---|---|---|
| [01](01-overview-and-tokenization.md) | 课程总览与 Tokenization | 为什么要从零理解 LM，以及 BPE 为什么重要 |
| [02](02-pytorch-and-resource-accounting.md) | PyTorch 与资源核算 | FLOPs、显存、训练时间怎么算 |
| [03](03-architectures-and-hyperparameters.md) | 架构与超参数 | RMSNorm、RoPE、SwiGLU、GQA 等默认配置 |
| [04](04-mixture-of-experts.md) | MoE | 稀疏激活怎么换参数量、换系统复杂度 |
| [05](05-gpus.md) | GPU 基础 | roofline、memory bound、tiling、FlashAttention 直觉 |
| [06](06-kernels-and-triton.md) | Kernel 与 Triton | profiling、fusion、Triton kernel 的写法 |
| [07](07-parallelism.md) | 并行训练 1 | DDP、ZeRO、FSDP、TP、PP、SP |
| [08](08-parallelism-part2.md) | 并行训练 2 | collectives、NCCL、benchmark、pipeline 实现 |
| [09](09-scaling-laws-fundamentals.md) | Scaling law 基础 | power law、Chinchilla、tokens per parameter |
| [10](10-inference.md) | 推理优化 | TTFT、KV Cache、GQA、Speculative Decoding |
| [11](11-scaling-laws-case-studies.md) | Scaling law 案例 | MUP、WSD、MiniCPM、DeepSeek 的外推方法 |
| [12](12-evaluation.md) | 评测 | perplexity、MMLU、GPQA、Agent benchmark、安全评测 |
| [13](13-data.md) | 数据工程 1 | Common Crawl、The Pile、版权、数据混合 |
| [14](14-data-part2.md) | 数据工程 2 | 过滤、FastText、重要性采样、去重、MinHash、LSH |
| [15](15-alignment-sft-rlhf.md) | 对齐 1 | SFT、FLAN、Alpaca、RLHF 全流程 |
| [16](16-alignment-rl.md) | 对齐 2 | DPO、PPO、GRPO、可验证奖励 |
| [17](17-alignment-rl-part2.md) | 对齐 3 | policy gradient、baseline、KL、GRPO 代码思路 |

## 推荐阅读路径

### 面试导向

`01 -> 02 -> 03 -> 05 -> 07 -> 09 -> 10 -> 12 -> 13 -> 15 -> 16`

### 工程导向

`02 -> 05 -> 06 -> 07 -> 08 -> 10 -> 14 -> 16 -> 17`

### 研究导向

`01 -> 03 -> 04 -> 09 -> 11 -> 12 -> 13 -> 15 -> 16 -> 17`

## 读这套笔记时的一个总原则

CS336 的核心价值不是“背术语”，而是训练一种统一思维：

> **任何一个模型设计，都要同时问清楚：它增加了什么能力、消耗了什么算力、引入了什么系统代价。**

这也是大模型工程里最值钱的视角。朴素，但能打。  
