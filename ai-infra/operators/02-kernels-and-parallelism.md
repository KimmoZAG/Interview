# Kernel 与并行基础（SIMD/SIMT）

## 要点

- GPU 上常见瓶颈：访存（带宽/缓存命中）与同步（block 间/warp 内）
- 很多“慢”的原因不是算子复杂，而是 **launch 次数太多** 或 **融合不足**
- CS336 在 GPU / Triton / parallelism 几讲里反复强调：真正影响性能的，往往不是公式，而是执行方式

## 你需要能解释的最小执行模型

- CPU：SIMD（向量化）、cache、分支预测
- GPU：SIMT（warp/线程块）、内存合并访问（coalescing）、共享内存/寄存器

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

## 最小例子建议（后续你可补代码/伪代码）

- GEMM（矩阵乘）是很多算子的核心
- LayerNorm / Softmax 的数值稳定写法（减 max）

## “并行”至少要分两层理解

这篇的并行基础先讲单设备执行模型，但从 CS336 来看，并行至少有两层：

- kernel 内并行：warp、block、tile、SIMT
- 训练系统并行：data parallel、tensor parallel、pipeline parallel、sequence parallel

很多人会把两者混在一起，导致讨论“并行优化”时对象不清。更准确的说法应该是：

- 我现在是在解决单个 kernel 如何更快
- 还是在解决多卡之间如何拆模型/拆通信

## 易错点

- 只看 FLOPs 忽略带宽：算子明明“计算不多”却很慢
- 小 batch / 小 shape 下 kernel launch 开销占主导
- 讨论并行时不区分单卡执行并行与多卡训练并行

## 排查 checklist

- [ ] 单次 kernel 时间 vs kernel 次数（是不是太碎）
- [ ] 带宽是否接近峰值？若很低，访问模式是否不连续
- [ ] 是否能通过融合/合并算子减少 launch
- [ ] 当前热点属于 GEMM、reduction、pointwise，还是 attention 类 kernel？

## CS336 对照

- 官方 lecture 对应：Lecture 5（GPUs）、Lecture 6（kernels, Triton）、Lecture 7-8（parallelism）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
	- https://rd.me/cs336
	- https://www.rajdeepmondal.com/blog/cs336-lecture-5
	- https://www.rajdeepmondal.com/blog/cs336-lecture-6
