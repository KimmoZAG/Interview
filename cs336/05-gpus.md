# 05｜GPU 基础与性能直觉

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec05/>

课程导航：上一讲 [04 Mixture of Experts](04-mixture-of-experts.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [06 Kernel 与 Triton](06-kernels-and-triton.md)

工程桥接：[`AI Infra / 内存层级与 Roofline`](../ai-infra/01-operator-optimization/03-memory-hierarchy-and-roofline.md)｜[`AI Infra / FlashAttention 与 IO-aware`](../ai-infra/01-operator-optimization/06-flashattention-io-aware.md)｜[`AI Infra / 计算图融合与调度`](../ai-infra/01-operator-optimization/04-graph-fusion-scheduling.md)

## 核心定义（What & Why）

> **一句话总结**：GPU 性能优化的核心不是“让计算更复杂”，而是“让数据搬运更少、复用更高”，它解决的是为什么同样的 FLOPs，在真实硬件上会跑出完全不同的速度。

## 关联知识网络

- 前置：[`02 PyTorch 与资源核算`](02-pytorch-and-resource-accounting.md)
- 延伸：[`06 Kernel 与 Triton`](06-kernels-and-triton.md)
- 延伸：[`10 推理优化`](10-inference.md)
- 平行：[`AI Infra / 内存层级与 Roofline`](../ai-infra/01-operator-optimization/03-memory-hierarchy-and-roofline.md)
- 平行：[`AI Infra / FlashAttention 与 IO-aware`](../ai-infra/01-operator-optimization/06-flashattention-io-aware.md)

## 先抓住这讲要点

- GPU 优化的第一原则不是“多算”，而是**少搬数据**。
- 现代加速器越来越是 **memory-bound**，而不是 compute-bound。
- tiling、fusion、mixed precision、recompute，本质上都在围绕一件事打转：**提高 arithmetic intensity**。
- FlashAttention 之所以经典，不是因为“注意力公式变了”，而是因为它把 IO 路径重新设计了一遍。

## 代表图

![lec05](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec05/01-05-46-1400.webp)

## 这一讲在回答什么

很多人初学 GPU 时，脑子里想的是“GPU 算得快”。  
但真正要学会的是：

- GPU 为什么有时候很快；
- 为什么有时候明明 FLOPs 很高却跑不满；
- 为什么访存模式常常比数学公式本身更重要。

一句话总结：

> GPU 优化本质上是在做数据调度，而不只是算术加速。

## 中文解读

### 1. CPU 和 GPU 的思维方式不同

- CPU：优化单线程低延迟、复杂控制流；
- GPU：优化海量线程高吞吐、规则计算。

所以写 GPU kernel 时要尽量让一大群线程做相似的事，别让 warp 内分叉得像开盲盒。

如果 CPU 像一个灵活但人少的手工作坊，那么 GPU 更像一条庞大的流水线工厂。  
流水线工厂最怕什么？最怕不是活儿难，而是：

- 原料来得不整齐；
- 工人动作不一致；
- 中间搬运太慢。

### 2. GPU 的真正瓶颈常常不是 FLOPs

现代 GPU 的 raw compute 增长速度很快，但外部 memory bandwidth 增长没那么快。  
这导致很多 kernel 的真实瓶颈不是“算不过来”，而是：

- 数据没法足够快地从 HBM 搬到 SM；
- 中间结果频繁写回再读出；
- 小 kernel launch 太多，流水线断断续续。

所以很多时候你看到“卡没跑满”，不代表算力不够，而是数据喂不进去。

### 3. Roofline 模型是最值得背的图之一

核心量：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}}
$$

- 如果 intensity 太低，性能被带宽卡住；
- 如果 intensity 足够高，才可能吃满算力峰值。

这张图的教学价值极高，因为它告诉你：

> 不是所有优化都该去“减少 FLOPs”，很多优化其实该去“减少 bytes moved”。

### 一个能在面试里现场算出来的小例子

如果一个 kernel 总共做了 `2 × 10^12 FLOPs`，同时搬运了 `5 × 10^11 Bytes`，那么它的 arithmetic intensity 就是：

$$
\frac{2 \times 10^{12}}{5 \times 10^{11}} = 4 \text{ FLOPs / Byte}
$$

这个数字本身没有绝对意义，关键是把它和硬件的 compute / bandwidth 比值对照着看。

工程上你真正该问的是：

- 这个 intensity 是否低到明显更像 bandwidth-bound；
- 如果是，那继续抠 FLOPs 意义就不大；
- 更应该做的是 fusion、tiling、layout 优化，减少 bytes moved。

也就是说，roofline 不是一张“知识点图片”，而是一张**决定下一步该改哪里**的路标。

### 4. 为什么 warp divergence 很伤

warp 内线程通常 lockstep 执行。  
如果一个 warp 里有些线程走 `if` 分支，有些走 `else` 分支，硬件往往要把两条路都执行一遍，只是分批屏蔽线程。

结果就是：

- 理论上 32 个线程并行；
- 实际上同一时间可能只有一部分在干活。

所以 GPU 特别喜欢：

- 规则的 memory access；
- 规则的控制流；
- 可批量、可向量化的运算。

### 5. 为什么 tiling 是 GPU 优化母题

假设你在做矩阵乘法。  
如果每次算一个输出元素都去 global memory 读完整行列，那代价极高。

tiling 的思路是：

1. 从 global memory 读一小块 tile 到 shared memory；
2. 在片上反复复用这块 tile；
3. 算够了再把结果写回。

收益是：

- 同一份数据被更多 FLOPs 消耗；
- global memory 访问次数显著下降；
- arithmetic intensity 提高。

这也是为什么很多高性能 kernel 看起来都在做一件事：

> 把大问题切成能放进片上缓存/共享内存的小块，再局部高复用地算完。

#### 从“听懂概念”到“把收益算出来”

如果某段数据原来每算一次输出就要从 HBM 读一次，那么复用率很低；
而 tiling 的目标是让同一块数据被多个输出重复使用。

你可以把它粗暴理解成：

- **没做 tiling**：一份数据搬进来，算一下就丢；
- **做了 tiling**：一份数据搬进来，在 shared memory 里多算几次再丢。

于是收益本质上就是：

> **同样的 bytes，喂出了更多 FLOPs。**

这也是为什么很多 GPU 优化最后都能翻译成一句很朴素的话：

- 不是你算得更“高级”了；
- 而是你终于不再反复从最贵的内存层级拿同一份数据了。

### 6. 为什么 fusion 常常很值

很多 elementwise 运算如果分成多步：

1. 读一次输入；
2. 写回中间结果；
3. 再读回来；
4. 再写一次。

这样虽然 FLOPs 不多，但 DRAM 流量会很大。  
fusion 则把多步算子合并成一个 kernel，让中间值尽量停留在寄存器或 shared memory 里。

所以 fusion 常常带来的收益是：

- 少 launch；
- 少写回；
- 少重读；
- 更高吞吐。

### 7. mixed precision 为什么本质上也是 IO 优化

大家常说混合精度是“为了更快”。  
其实更精确地说，它很多时候是为了：

- 每个数更小；
- 带宽占用更少；
- cache/片上容量更大；
- tensor core 更容易发挥。

也就是说，它并不只是数学精度策略，还是 memory traffic 策略。

### 8. recompute 为什么是合理的

如果某个中间结果：

- 存下来要占大量显存；
- 下次再读回来也很贵；
- 但重新算一遍并不贵；

那就可以选择 recompute。

这在 GPU 世界里非常自然，因为现代硬件常常是“算力相对富裕，带宽相对紧张”。  
所以：

> 多算一点，少搬一点，可能反而更快。

## 对比表：常见 GPU 优化动作到底在优化什么

| 手段 | 直接作用 | 本质收益 | 常见代价 |
|---|---|---|---|
| Tiling | 把数据块搬到片上反复复用 | 提高 arithmetic intensity，减少 HBM 往返 | tile 设计复杂，受 shared memory 限制 |
| Fusion | 合并多个小算子 | 少 launch、少写回、少重读 | kernel 更复杂，调试难 |
| Mixed precision | 降低每个元素字节数 | 降带宽压力，提升 tensor core 利用率 | 数值稳定性要额外处理 |
| Recompute | 不存中间结果，按需再算 | 省显存，少读写大激活 | 额外计算增加 |

## FlashAttention 为什么快

不是因为它“改了注意力公式”，而是因为它：

- 不把 $N \times N$ attention matrix 整块落到 HBM；
- 在片上 memory 做 tile 级计算；
- 用 online softmax 维持数值稳定；
- backward 时选择重算，减少显存写回。

它之所以经典，是因为它非常系统性地展示了一个原则：

> 在现代加速器上，重新设计 IO 路径，常常比重新设计数学公式更重要。

### 一个更工程化的回答模板

如果面试官追问你“为什么 FlashAttention 会快”，比起只说“它减少了显存”，更强的答法是分三层：

1. **它没有把整块 attention matrix 落到 HBM；**
2. **它用 tile + online softmax 把计算留在片上；**
3. **它把瓶颈从大量全局读写，转成了更高复用的片上计算。**

也就是说，它不是“公式突然变简单”，而是把最贵的数据搬运路径切短了。

## 代码拆解：在线 softmax 思想

```python
import math

def online_softmax(xs):
    m = -float('inf')
    s = 0.0
    for x in xs:
        new_m = max(m, x)
        s = s * math.exp(m - new_m) + math.exp(x - new_m)
        m = new_m
    return [math.exp(x - m) / s for x in xs]
```

这段代码说明：即使数据分块到达，也能维护一个全局正确的 softmax 归一化项。  
这就是 FlashAttention 能 tile 化 attention 的关键数学部件。

### 为什么它重要

普通 softmax 看起来像需要“一次性看到整行 logits 才能算”。  
online softmax 则告诉你：不需要。  
只要维护：

- 当前最大值；
- 当前归一化和；

就能边读 tile 边更新。这一改，直接把 attention 从“必须落大矩阵”变成了“可流式计算”。

## GPU 优化 checklist

- 访问是否 coalesced？
- 有没有不必要的 global memory round-trip？
- 能不能 fuse？
- 能不能 tile 到 shared memory？
- 能不能用更低精度？
- 能不能 checkpoint / recompute？
- warp 有没有明显 divergence？
- tile size 是否和硬件粒度对齐？

## 💥 实战踩坑记录（Troubleshooting）

> 现象：理论 FLOPs 很高，profiling 里 kernel 也不少，但 GPU utilization 还是上不去。

- **误判**：以为是“算子不够复杂”或 FLOPs 还不够高，继续在数学公式上抠细节。
- **根因**：更常见的是 memory-bound——HBM 搬运太重、小 kernel 太碎、layout 不友好，导致计算单元长期在等数据。
- **解决动作**：
    - 先用 roofline 思维判断是带宽瓶颈还是算力瓶颈；
    - 再看能不能做 tiling、fusion、改 layout、降 precision；
    - 最后才考虑是否真的需要改算法。
- **复盘**：GPU 很多时候不是“不会算”，而是“吃不到料”。

> 常见异常：FlashAttention 上了以后显存下来了，但整体收益没有想象中大。

- 这往往说明瓶颈已经不只在 attention IO，上游调度、下游 KV cache、或其他 kernel 的数据路径也在一起分摊总延迟。

## 面试里怎么讲这一讲

如果被问：**“GPU 优化最核心的原则是什么？”**

可以答：

> 现代 GPU 优化最核心的原则通常不是减少 FLOPs，而是减少数据搬运，特别是减少 global memory 和 HBM 的流量。因为很多深度学习 kernel 实际上是 memory-bound 的，所以像 tiling、fusion、mixed precision 和 recompute 这些优化，本质上都是在提高 arithmetic intensity。

如果被问：**“FlashAttention 为什么快？”**

可以答：

> 它快不是因为改变了 attention 目标，而是通过 tiling、online softmax 和重算，把原本需要写出和读取大规模 attention matrix 的 IO 路径优化掉了。所以它本质上是一个 IO-aware attention 实现。

## 本讲小结

这一讲建立的是 GPU 性能直觉：

- GPU 喜欢规则、高复用、少搬运；
- 很多模型算子不是算不动，而是喂不饱；
- 真正好的 kernel 往往先优化数据路径，再优化数学路径。

## 复习题

1. 为什么很多深度学习 kernel 是 memory-bound？
2. tiling 如何提升 arithmetic intensity？
3. FlashAttention 的本质优化点是什么？
4. 为什么说 mixed precision 也是一种 IO 优化？
5. recompute 在什么情况下会让系统更快？

## 面试常见题目

1. roofline 模型为什么能帮助你判断优化方向？
2. HBM、SRAM、register 在性能上分别意味着什么？
3. 为什么同样的 FLOPs，不同 kernel 的速度会差很多？
4. 什么时候你应该优先优化数据布局，而不是继续改数学公式？
5. 为什么很多“理论算力很高”的 GPU 程序实际跑不满？

## 面试题答题提示

### 1. 回答 GPU 问题时，先讲数据搬运

很多性能问题不是算子数学错了，而是数据没有被高效送到计算单元。这个视角很关键。

### 2. roofline 不是装饰图，是决策工具

它最有用的地方是告诉你瓶颈更接近带宽还是更接近算力，从而决定应该做 fusion、tiling 还是换算法。

### 3. FlashAttention 要讲成 IO-aware 实现

它不是在改变 attention 定义，而是在减少中间大矩阵的写回与读回成本。
