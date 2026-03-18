# CS336 学习路线图

这份路线图不是简单列目录，而是把 `cs336` 重新按目标拆成三种读法：

- 想尽快形成整体认知；
- 想做工程实现；
- 想准备面试。

如果你不确定怎么开始，就先读这一页，再回到 [00-index.md](00-index.md)。

## 先记住这套材料最适合怎么学

这套笔记最怕的读法是：

- 一页一页顺着看；
- 看完觉得都懂；
- 一到面试或实战，嘴里只剩术语。

更有效的读法是把每一页都当成一个小训练单元：

1. **先抓一句话结论**：它到底在解决什么瓶颈；
2. **再抓一个量化样例**：至少有一个公式或数量级能现场复算；
3. **最后抓一个工程决策**：这个结论会影响你怎么选模型、选并行、选推理方案。

如果一页读完之后，你还说不出“它到底帮我省了什么、贵了什么”，那通常说明还没有真正学会。

## 一、先建立整张脑图

如果你之前没有系统学过大模型，建议先抓住这条主线：

```text
tokenizer -> transformer -> 训练资源 -> GPU / kernel -> 并行训练 -> scaling law -> 推理 -> 数据 -> 对齐
```

这条线的意义是：

- 前半段告诉你模型在算什么；
- 中间部分告诉你系统为什么会成为瓶颈；
- 后半段告诉你为什么真实产品不只靠预训练。

## 二、两周面试冲刺路线

如果你的目标是尽快准备大模型基础设施、训练系统、推理优化、算法工程类面试，建议优先读下面这些文件：

### 第 1 阶段：建立统一语言

1. [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
2. [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
3. [03-architectures-and-hyperparameters.md](03-architectures-and-hyperparameters.md)

这一阶段要达到的目标：

- 能讲清为什么 tokenizer 影响训练和推理成本；
- 能粗算 FLOPs、显存、训练时间；
- 能说清现代 Transformer 默认结构为什么长这样。

### 这一阶段怎么练，才不会读成“概念词典”

建议每读完一页，都强迫自己回答一遍：

- 这页最值钱的一个公式是什么？
- 这个公式能支持什么工程判断？
- 如果面试官让我举数值例子，我能不能现场算一遍？

比如读完 `02`，至少要能做到：

- 会推 `6PN` 的来源；
- 会算 `16 Bytes / Param` 的 Adam 静态显存账；
- 会解释为什么 `view` 和 `contiguous()` 会带来完全不同的显存行为。

### 第 2 阶段：补系统直觉

1. [05-gpus.md](05-gpus.md)
2. [06-kernels-and-triton.md](06-kernels-and-triton.md)
3. [07-parallelism.md](07-parallelism.md)
4. [08-parallelism-part2.md](08-parallelism-part2.md)
5. [10-inference.md](10-inference.md)

这一阶段要达到的目标：

- 能解释 memory-bound 和 compute-bound；
- 能比较 DDP、FSDP、TP、PP；
- 能解释 KV cache 为什么是推理系统中心。

### 这一阶段建议按“三本账”去学

读系统页时，不要只记技术名词，建议始终按这三本账做笔记：

| 账本 | 你要追问什么 |
|---|---|
| 显存账 | 它省了参数、激活，还是 KV cache？ |
| 吞吐账 | 它是在减少计算、减少访存，还是减少等待？ |
| 通信账 | 它把同步插在了哪一段关键路径上？ |

比如：

- `05 GPU` 更偏吞吐账；
- `07 并行训练` 要同时看显存账和通信账；
- `10 推理优化` 则常常同时牵扯 KV 显存账、decode 带宽账和调度等待账。

### 第 3 阶段：补大模型方法论

1. [09-scaling-laws-fundamentals.md](09-scaling-laws-fundamentals.md)
2. [11-scaling-laws-case-studies.md](11-scaling-laws-case-studies.md)
3. [13-data.md](13-data.md)
4. [15-alignment-sft-rlhf.md](15-alignment-sft-rlhf.md)
5. [16-alignment-rl.md](16-alignment-rl.md)

这一阶段要达到的目标：

- 能讲清参数量、token 数和 FLOPs 预算怎么分配；
- 能讲清数据工程为什么会直接影响模型能力；
- 能讲清 SFT、RLHF、DPO 的区别和各自适用边界。

### 这一阶段最重要的是建立“外推边界感”

这一组内容最容易被读成“结论集合”，但真正应该练的是：

- 小模型实验什么时候能外推；
- 什么叫预算最优，不是参数越大越好；
- 为什么高质量数据和后训练行为塑形，常常比盲目多训更值钱。

## 三、四周系统学习路线

如果你想真正把课程吃透，而不是只为面试记结论，可以按下面的四周路线走。

### 第 1 周：基础建模

1. [01-overview-and-tokenization.md](01-overview-and-tokenization.md)
2. [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
3. [03-architectures-and-hyperparameters.md](03-architectures-and-hyperparameters.md)
4. [04-mixture-of-experts.md](04-mixture-of-experts.md)

### 第 2 周：系统与并行

1. [05-gpus.md](05-gpus.md)
2. [06-kernels-and-triton.md](06-kernels-and-triton.md)
3. [07-parallelism.md](07-parallelism.md)
4. [08-parallelism-part2.md](08-parallelism-part2.md)

### 第 3 周：规模规律与推理

1. [09-scaling-laws-fundamentals.md](09-scaling-laws-fundamentals.md)
2. [10-inference.md](10-inference.md)
3. [11-scaling-laws-case-studies.md](11-scaling-laws-case-studies.md)
4. [12-evaluation.md](12-evaluation.md)

### 第 4 周：数据与对齐

1. [13-data.md](13-data.md)
2. [14-data-part2.md](14-data-part2.md)
3. [15-alignment-sft-rlhf.md](15-alignment-sft-rlhf.md)
4. [16-alignment-rl.md](16-alignment-rl.md)
5. [17-alignment-rl-part2.md](17-alignment-rl-part2.md)

## 四、如果你在带学员，推荐直接用“few-shot 改写法”

如果你的目标不只是自己看懂，而是想带别人一起学，这套材料最适合的方式不是泛泛讲大道理，而是拿具体段落做 Before / After 改写。

推荐直接配合：

- [reference/note-writing-playbook.md](../reference/note-writing-playbook.md)

具体练法可以固定成下面三步：

### 第一步：找一段“只有结论”的旧笔记

例如：

- “训练 FLOPs 约等于 `6PN`”
- “Adam 会占很多显存”
- “contiguous() 可能导致显存变大”

### 第二步：强制补三样东西

1. **推导**：为什么是这样；
2. **数字**：至少给一个可复算的量级；
3. **决策**：这会怎么影响系统选型。

### 第三步：检查是不是已经从“学生视角”变成“工程视角”

最简单的判断标准：

- 有推导，不只报结论；
- 有数字，不只讲趋势；
- 有机制，不只背 API；
- 有决策，不只记定义。

如果学员能持续按这四条重写，他的笔记质量会比单纯抄 lecture 快很多。

## 五、工程实现导向怎么读

如果你更偏 infra 或系统实现，可以把重心放在下面这些讲次：

1. [02-pytorch-and-resource-accounting.md](02-pytorch-and-resource-accounting.md)
2. [05-gpus.md](05-gpus.md)
3. [06-kernels-and-triton.md](06-kernels-and-triton.md)
4. [07-parallelism.md](07-parallelism.md)
5. [08-parallelism-part2.md](08-parallelism-part2.md)
6. [10-inference.md](10-inference.md)
7. [14-data-part2.md](14-data-part2.md)

读这些时，建议始终围绕四个问题：

- 瓶颈在算力、显存、带宽还是通信？
- 这个优化改善的是吞吐、延迟还是成本？
- 收益来自更少计算，还是更少数据移动？
- 这个方法在什么规模下才值得引入？

### 一个很实用的顺序：先 02，再 05，再 06/07/10

如果你是系统或 infra 方向，建议优先吃透这条链：

`02 资源核算 -> 05 GPU -> 06 Kernel/Triton -> 07 并行 -> 10 推理`

原因很简单：

- `02` 让你学会算账；
- `05` 让你学会判断瓶颈在计算还是访存；
- `06` 让你学会用 benchmark / profile 定位热点；
- `07` 让你学会在多卡上重新分配资源；
- `10` 让你学会把模型变成服务系统。

这条链一旦吃透，很多面试问题就不再是零散术语，而会自动连起来。

## 六、面试准备时要特别补的能力

很多人读完资料后，仍然答不好面试，原因通常不是没见过术语，而是缺三种能力：

### 1. 资源核算能力

至少要能快速估：

- 参数量；
- 训练 FLOPs；
- 激活和 optimizer 显存；
- 推理时 KV cache 大小。

### 2. 瓶颈定位能力

至少要能分清：

- compute-bound vs memory-bound；
- 单卡 kernel 问题 vs 多卡通信问题；
- 训练瓶颈 vs 推理瓶颈；
- 模型能力问题 vs 数据质量问题。

### 3. 规模感

至少要知道：

- 小实验不一定能外推到大模型；
- 某些 trick 只有在特定上下文长度、batch 或并行规模下才有收益；
- 很多结论必须连同资源 regime 一起说才严谨。

## 七、读每一讲时建议自问什么

为了避免只读成“概念总结”，每一讲都可以问自己下面四个问题：

1. 这一讲的核心瓶颈是什么？
2. 这一讲最重要的 trade-off 是什么？
3. 这一讲里最容易在面试被问到的概念是什么？
4. 这一讲的结论，放到真实训练或推理系统里，落地影响是什么？

## 八、最后给自己做一次“过线检查”

如果你是为了面试或带学员，读完整套材料后，至少应该能独立讲顺下面这些题：

1. 为什么训练 FLOPs 常近似成 `6PN`？
2. 为什么 `7B` 模型训练时会很快撞上显存墙？
3. 为什么 GPU 优化常常先盯 bytes moved，而不是 FLOPs？
4. DDP / FSDP / TP / PP 分别在切什么？
5. 为什么 decode 往往比 prefill 更像带宽战？
6. 为什么高质量数据和 post-training 会直接改变产品体验？

如果这些题你能讲顺，这套路线图就算真正被你“走通”了。

## 九、接下来去哪

- 想看整套课程地图： [00-index.md](00-index.md)
- 想按面试主题复习： [interview-prep-guide.md](interview-prep-guide.md)
- 想从第一讲开始： [01-overview-and-tokenization.md](01-overview-and-tokenization.md)