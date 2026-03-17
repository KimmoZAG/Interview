# Kernel 与并行基础（SIMD/SIMT）

## 要点

- GPU 上常见瓶颈：访存（带宽/缓存命中）与同步（block 间/warp 内）
- 很多“慢”的原因不是算子复杂，而是 **launch 次数太多** 或 **融合不足**
- CS336 在 GPU / Triton / parallelism 几讲里反复强调：真正影响性能的，往往不是公式，而是执行方式
- 想读懂 kernel 性能，至少要能分清：**算力没吃满、带宽没吃满、还是 launch / 同步把系统切碎了**。

## 通用知识

### 它是什么

kernel 是设备上一次具体执行的工作单元。对于推理优化来说，真正落到硬件上跑的，通常不是“高层算子”本身，而是一组 kernel。

理解 kernel 执行模型，就是理解：

- 工作如何被拆给线程 / warp / block
- 数据如何被读取、复用与写回
- 为什么有些算子看起来简单却跑得不快

### 它解决什么问题

它帮助你回答：

- 为什么同一个算子在不同 shape 下速度完全不同
- 为什么有些算子 FLOPs 不高却依然很慢
- 为什么 decode 阶段的小 shape kernel 特别容易“看起来都在忙，实际吞吐却不好”

### 为什么在 AI 系统里重要

因为模型最终都会变成 kernel 序列，而很多系统级性能问题都会在这里露出真身：

- kernel 太碎
- launch 太多
- 内存访问不连续
- reduction / sync 太重
- occupancy 不高

### 它的收益与代价

收益：

- 能更快识别问题更像算力瓶颈还是执行瓶颈
- 能更好理解 fusion、tiling、layout 调整为什么有效

代价：

- 需要接受一个现实：很多性能结论并不能只从公式层面推出，必须回到执行模型上看

## 你需要能解释的最小执行模型

- CPU：SIMD（向量化）、cache、分支预测
- GPU：SIMT（warp/线程块）、内存合并访问（coalescing）、共享内存/寄存器

一句够用的口语版：

- CPU 更擅长复杂控制流和低延迟逻辑
- GPU 更擅长把大量相似工作并行摊开

## Kernel 性能的常见决定因素

- 并行度：线程/warp 是否足够填满 SM
- 内存访问：是否连续、是否对齐、是否复用（cache/shared memory）
- 计算密度：FLOPs/Byte（算术强度）
- 同步与原子：是否引入大量 serialization

## CS336 里最常见的 4 类 kernel 心智模型

- GEMM：大部分线性层和投影层的核心
- Reduction：softmax、layernorm、统计量计算经常离不开它
- Pointwise：激活、bias、残差加法等，单个不重，但碎片化时会拖慢整体
- Attention kernel：往往是多种访存和计算模式的组合，不应只按“一个 op”理解

一个很有用的经验法：

- 大 GEMM 往往更吃算力利用率
- LayerNorm / Softmax 往往更容易受访存和 reduction 影响
- Decode 下的小 shape 算子更容易被 launch 与调度主导

## kernel 慢通常慢在哪

最常见的 4 类原因：

1. 线程不够饱，硬件没吃满
2. 访存模式差，Bytes 很多但效率低
3. kernel 太碎，launch 开销开始主导
4. reduction / sync / atomic 让并行被序列化

如果不先把问题归到这几类之一，后续优化动作通常会很飘。

## 最小例子

假设有两个 kernel：

- 一个是大 GEMM
- 一个是小 shape 的 pointwise / reduction 链

大 GEMM 常见特征：

- 单次时间长一些
- 但更容易吃满硬件

小 shape 链常见特征：

- 单次时间看似不长
- 但如果数量很多，launch 和同步会把整体切碎

这就是为什么有时 profiler 上“每个小 kernel 都不夸张”，系统整体却依然慢。

## “并行”至少要分两层理解

这篇的并行基础先讲单设备执行模型，但从 CS336 来看，并行至少有两层：

- kernel 内并行：warp、block、tile、SIMT
- 训练系统并行：data parallel、tensor parallel、pipeline parallel、sequence parallel

很多人会把两者混在一起，导致讨论“并行优化”时对象不清。更准确的说法应该是：

- 我现在是在解决单个 kernel 如何更快
- 还是在解决多卡之间如何拆模型/拆通信

## 工程例子

一个典型现象：

- GPU utilization 看起来不低
- 但 tokens/s 还是上不去

这时常见原因不是“GPU 不够忙”，而是：

- kernel 太多太碎
- 小 shape decode 阶段 launch 开销被放大
- reduction 和同步导致执行效率差

所以“GPU 忙”不等于“系统高效”。

## 推理优化工程师视角

对推理优化工程师来说，这篇最重要的价值是建立 3 个本能：

1. 先分 kernel 类型：GEMM、reduction、pointwise、attention
2. 再分慢因：算力、带宽、launch、同步
3. 最后再决定是否该做 fusion、换 kernel、调 layout 或调 batching

## 常见面试问题

### 初级

1. SIMD 和 SIMT 有什么直觉区别？
2. 为什么很多推理性能问题最终会落到 kernel 执行方式上？

### 中级

1. 为什么 decode 阶段的小 shape kernel 更容易被 launch 拖慢？
2. 为什么 FLOPs 不高的算子也可能很慢？

### 高级

1. 如果 GPU utilization 不低，但整体吞吐仍然不理想，你会优先怀疑哪些 kernel 级问题？
2. 如何快速区分一个热点更像 GEMM 问题、reduction 问题，还是 pointwise 碎片问题？

## 易错点

- 只看 FLOPs 忽略带宽：算子明明“计算不多”却很慢
- 小 batch / 小 shape 下 kernel launch 开销占主导
- 讨论并行时不区分单卡执行并行与多卡训练并行
- 只盯单个 kernel 耗时，不看 kernel 次数和整体碎片化

## 排查 checklist

- [ ] 单次 kernel 时间 vs kernel 次数（是不是太碎）
- [ ] 带宽是否接近峰值？若很低，访问模式是否不连续
- [ ] 是否能通过融合/合并算子减少 launch
- [ ] 当前热点属于 GEMM、reduction、pointwise，还是 attention 类 kernel？

## 参考资料

- GPU 执行模型与 profiling 资料
- 建议串读：`03-memory-hierarchy-and-roofline.md`、`04-graph-fusion-scheduling.md`

## CS336 对照

- 官方 lecture 对应：Lecture 5（GPUs）、Lecture 6（kernels, Triton）、Lecture 7-8（parallelism）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://rd.me/cs336
  - https://www.rajdeepmondal.com/blog/cs336-lecture-5
  - https://www.rajdeepmondal.com/blog/cs336-lecture-6
