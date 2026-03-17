# CS336 学习路线图

这份路线图不是简单列目录，而是把 `cs336` 重新按目标拆成三种读法：

- 想尽快形成整体认知；
- 想做工程实现；
- 想准备面试。

如果你不确定怎么开始，就先读这一页，再回到 [00-index.md](00-index.md)。

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

## 四、工程实现导向怎么读

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

## 五、面试准备时要特别补的能力

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

## 六、读每一讲时建议自问什么

为了避免只读成“概念总结”，每一讲都可以问自己下面四个问题：

1. 这一讲的核心瓶颈是什么？
2. 这一讲最重要的 trade-off 是什么？
3. 这一讲里最容易在面试被问到的概念是什么？
4. 这一讲的结论，放到真实训练或推理系统里，落地影响是什么？

## 七、接下来去哪

- 想看整套课程地图： [00-index.md](00-index.md)
- 想按面试主题复习： [interview-prep-guide.md](interview-prep-guide.md)
- 想从第一讲开始： [01-overview-and-tokenization.md](01-overview-and-tokenization.md)