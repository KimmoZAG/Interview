# CS336 面试准备指南

这份指南的目标不是把 17 讲重新抄一遍，而是把它们改写成面试真正常见的知识簇。你可以把它看成 `cs336` 的二级索引：不是按讲次，而是按问题类型组织。

## 一、面试官最常考的五大主题

### 1. 模型基础

对应讲次：

- [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
- [03-architectures-and-hyperparameters.md](03-architectures-and-hyperparameters.md)
- [04-mixture-of-experts.md](04-mixture-of-experts.md)

高频问题：

1. tokenizer 为什么会影响训练和推理成本？
2. 为什么现代 Transformer 常用 RMSNorm、RoPE、SwiGLU、GQA？
3. dense model 和 MoE 的 trade-off 是什么？

### 2. 资源核算与系统瓶颈

对应讲次：

- [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
- [05-gpus.md](05-gpus.md)
- [06-kernels-and-triton.md](06-kernels-and-triton.md)

高频问题：

1. 训练 FLOPs 为什么经常近似成 $6PN$？
2. 为什么参数能装下，不代表训练一定能跑？
3. roofline 模型在大模型系统里有什么用？
4. memory-bound kernel 常见优化方向是什么？

### 3. 并行训练

对应讲次：

- [07-parallelism.md](07-parallelism.md)
- [08-parallelism-part2.md](08-parallelism-part2.md)

高频问题：

1. DDP、FSDP、ZeRO、TP、PP 分别解决什么问题？
2. 什么情况下更适合做 tensor parallel，而不是纯 data parallel？
3. 为什么并行训练的收益常被通信吃掉？

### 4. 推理与 serving

对应讲次：

- [10-inference.md](10-inference.md)
- [12-evaluation.md](12-evaluation.md)

高频问题：

1. prefill 和 decode 为什么是两个性能世界？
2. 为什么 KV cache 是推理系统核心？
3. speculative decoding 的收益来自哪里？
4. 如何同时权衡 TTFT、throughput 和并发？

### 5. 数据与对齐

对应讲次：

- [13-data.md](13-data.md)
- [14-data-part2.md](14-data-part2.md)
- [15-alignment-sft-rlhf.md](15-alignment-sft-rlhf.md)
- [16-alignment-rl.md](16-alignment-rl.md)
- [17-alignment-rl-part2.md](17-alignment-rl-part2.md)

高频问题：

1. 数据质量为什么往往比“继续加训练步数”更关键？
2. 去重、过滤、数据混合分别在解决什么问题？
3. SFT、RLHF、DPO、GRPO 有什么区别？
4. 为什么 post-training 更像行为塑形，而不是重新教知识？

## 二、面试回答时最容易暴露的短板

很多回答听起来“不像做过”，通常是因为落入了下面几类问题。

### 1. 只会背定义，不会讲代价

比如会说：

- GQA 可以减少 KV cache；
- FSDP 可以节省显存；
- tokenizer 会影响序列长度。

但如果讲不出：

- 节省的是显存、带宽还是通信；
- 代价是精度、复杂度还是实现难度；
- 在什么规模下收益才显著；

那回答通常会显得浅。

### 2. 只说模型，不说系统

大模型岗位里，面试官通常很在意你能不能把模型设计和系统代价连起来。比如：

- MoE 不是只有“参数变多”，还带来路由、负载均衡、all-to-all；
- 长上下文不是只有“效果更强”，还带来 KV cache 与带宽压力；
- 更大 batch 不是只有“训练更稳”，还带来显存与通信代价。

### 3. 没有规模感

如果你的表述像下面这样，通常会被继续追问：

- “这个 trick 一定更好。”
- “这个并行方案更高效。”
- “这个 tokenizer 更先进。”

更成熟的表达应该是：

> 在某个模型规模、上下文长度、batch、硬件和数据分布下，这个方案更有可能带来收益。

## 三、推荐的面试回答结构

绝大多数系统或算法问题，都可以用下面这个结构回答：

1. **先定义问题**：这项技术在解决什么瓶颈。
2. **再讲核心方法**：它改变了什么计算或数据流。
3. **再讲收益**：收益落在哪个指标上。
4. **最后讲代价与边界**：它引入了什么复杂度，什么场景下不一定划算。

例如回答“为什么 GQA 有用”时，可以说：

> GQA 的目标是减小推理时的 KV cache 负担。它通过让多个 query 共享更少的 K/V heads，减少了需要长期保存和读取的历史状态，因此显存占用和 decode 时的带宽压力都会下降。它的收益主要体现在长上下文和高并发推理中，但代价是注意力表达能力可能受限，实际效果还要看模型规模和训练配方。

## 四、建议优先背熟的 20 个问题

1. 为什么 tokenizer 会影响 attention 成本？
2. 训练 FLOPs 为什么近似是 $6PN$？
3. 为什么 activation 常是训练 OOM 的关键来源？
4. compute-bound 和 memory-bound 怎么区分？
5. FlashAttention 本质上优化了什么？
6. DDP 和 FSDP 的主要区别是什么？
7. tensor parallel 和 pipeline parallel 适用边界分别是什么？
8. 为什么 all-reduce 会成为扩展性瓶颈？
9. scaling law 为什么有用？
10. 为什么不能直接把小模型实验结论外推到大模型？
11. prefill 和 decode 的性能特征为什么不同？
12. KV cache 为什么是长上下文推理的核心瓶颈之一？
13. speculative decoding 为什么能加速？
14. perplexity 为什么重要但不够？
15. 数据去重为什么会影响最终模型能力？
16. 为什么高质量数据混合比盲目堆数据更重要？
17. SFT 和 RLHF 在 post-training 中分别做什么？
18. DPO 为什么不需要显式 reward model 也能工作？
19. GRPO 相比 PPO 在工程上为什么更受关注？
20. 如果预算固定，你会先优化模型、数据还是系统？为什么？

## 五、不同岗位该怎么侧重

### 算法工程 / 训练岗位

重点看：

- [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
- [03-architectures-and-hyperparameters.md](03-architectures-and-hyperparameters.md)
- [09-scaling-laws-fundamentals.md](09-scaling-laws-fundamentals.md)
- [13-data.md](13-data.md)
- [15-alignment-sft-rlhf.md](15-alignment-sft-rlhf.md)
- [16-alignment-rl.md](16-alignment-rl.md)

### 推理 / serving / infra 岗位

重点看：

- [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
- [05-gpus.md](05-gpus.md)
- [06-kernels-and-triton.md](06-kernels-and-triton.md)
- [07-parallelism.md](07-parallelism.md)
- [08-parallelism-part2.md](08-parallelism-part2.md)
- [10-inference.md](10-inference.md)

### 研究导向岗位

重点看：

- [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
- [04-mixture-of-experts.md](04-mixture-of-experts.md)
- [09-scaling-laws-fundamentals.md](09-scaling-laws-fundamentals.md)
- [11-scaling-laws-case-studies.md](11-scaling-laws-case-studies.md)
- [12-evaluation.md](12-evaluation.md)
- [13-data.md](13-data.md)

## 六、怎么把这份指南和课程配合起来

推荐用法是：

1. 先在 [00-index.md](00-index.md) 建立课程地图。
2. 再按 [study-roadmap.md](study-roadmap.md) 选择适合自己的阅读路径。
3. 然后回到这份指南，把每个主题练到能口头讲出来。

## 七、下一步去哪

- 看课程总地图： [00-index.md](00-index.md)
- 看学习路线： [study-roadmap.md](study-roadmap.md)
- 从第一讲开始： [01-overview-and-tokenization.md](01-overview-and-tokenization.md)