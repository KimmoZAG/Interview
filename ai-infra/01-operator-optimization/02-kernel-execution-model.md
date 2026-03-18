# Kernel 与并行基础（SIMD / SIMT）

## 一句话先讲清

真正影响 kernel 性能的，往往不是公式本身，而是**它怎么被拆给线程、怎么读内存、怎么同步，以及是不是被 launch 次数切碎了**。

如果说上一页在回答“喂给 kernel 的数据长什么样”，这一页就在回答“kernel 拿到这些数据之后，硬件到底怎么跑”。

## 关联知识网络

- 输入对象：[`张量、shape 与内存布局`](01-tensors-shapes-layout.md)
- 带宽模型：[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)
- 图融合：[`计算图、融合与调度`](04-graph-fusion-scheduling.md)
- 模型侧热点：[`Transformer 推理所需的最小知识`](../03-llm-architecture/01-transformer-minimum.md)
- 推理链路：[`推理优化 Playbook`](../02-inference-engine/05-optimization-playbook.md)

## 为什么值得单独学

- 模型最终都会变成 kernel 序列，很多系统级性能问题都会在这里露真身。
- 很多“慢”的原因不是算子复杂，而是 launch 次数太多、访存模式太差、同步太重。
- 想读懂 profiling，至少要先能把问题归到：**算力、带宽、launch、同步** 这几类里。

## 最小执行模型：先别背术语，先有直觉

### CPU vs GPU

- CPU：更擅长复杂控制流、低延迟逻辑、分支丰富的任务
- GPU：更擅长把大量相似工作并行摊开

### SIMD vs SIMT

- SIMD：更像“一条指令喂一批数据”
- SIMT：更像“很多线程看起来各自执行，但硬件按 warp 等组织去调度”

面试里说到这一步就够用了，关键不是把术语讲成教科书，而是能把它和性能现象对应起来。

## Kernel 性能常见由什么决定

| 维度 | 它在问什么 |
|---|---|
| 并行度 / Occupancy | 线程、warp、block 是否足够把 SM 喂饱 |
| 内存访问 | 是否连续、是否对齐、是否有复用 |
| 算术强度 | 每搬一个 Byte 做了多少计算 |
| 同步 / 原子 | 是否引入了大量序列化 |
| Launch 开销 | kernel 是否太碎，调度成本开始主导 |

## 四类最常见的 kernel 心智模型

| 类型 | 常见例子 | 常见瓶颈倾向 |
|---|---|---|
| GEMM | 线性层、投影层 | 更偏算力利用率 |
| Reduction | softmax、layernorm、统计量 | 更偏访存与同步 |
| Pointwise | 激活、bias、残差加法 | 单个不重，但碎片化时很烦 |
| Attention kernel | attention 相关融合实现 | 往往是访存、计算和同步的组合题 |

一个很有用的经验法：

- 大 GEMM 更容易吃满硬件
- LayerNorm / Softmax 更容易受访存和 reduction 影响
- Decode 下的小 shape 算子更容易被 launch 和调度主导

## Kernel 慢通常慢在哪

最常见的四类原因：

1. 线程不够饱，硬件没吃满
2. 访存模式差，Byte 搬了很多但效率低
3. Kernel 太碎，launch 开销开始主导
4. Reduction / sync / atomic 让并行被序列化

如果不先把问题归到这几类之一，后续优化动作通常会很飘。

## 最小例子：为什么“小 kernel 都不夸张”，整体却还是慢

假设有两个热点：

- 一个是大 GEMM
- 一个是一串小 shape 的 pointwise / reduction kernel

大 GEMM 常见特征：

- 单次时间更长一些
- 但更容易吃满硬件

小 shape 链常见特征：

- 单次时间看起来不吓人
- 但如果数量很多，launch 和同步会把整体切碎

这就是为什么 profiler 上“每个小 kernel 都不夸张”，系统整体却依然慢。

## “并行”至少要分两层理解

这页讲的是**单设备执行并行**，但工程里常说的“并行”至少有两层：

- kernel 内并行：warp、block、tile、SIMT
- 系统并行：data parallel、tensor parallel、pipeline parallel、sequence parallel

很多人把这两层混在一起，讨论“并行优化”时对象就会跑偏。更准确的说法应该是：

- 我现在是在解决单个 kernel 怎么更快
- 还是在解决多卡之间怎么拆模型、拆通信

## Troubleshooting：为什么 GPU utilization 不低，tokens/s 还是上不去

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| GPU utilization 不低，但吞吐差 | kernel 太多太碎 | 看 kernel 次数与单次时长分布 |
| FLOPs 不高的算子却很慢 | 访存或 reduction 主导 | 看带宽利用与同步开销 |
| decode 阶段明显拖慢 | 小 shape + launch 开销 | 分离 prefill / decode profile |
| 热点很多但都不极端 | 整体碎片化严重 | 看 pointwise / reduction 链是否过长 |

### 一个排障顺序

1. 先分 kernel 类型：GEMM、reduction、pointwise、attention。
2. 再分慢因：算力、带宽、launch、同步。
3. 最后才决定是该做 fusion、换 kernel、调 layout，还是调 batching。

## 推理优化工程师视角

这页最重要的价值，是帮你建立三个本能：

1. 先分 kernel 类型。
2. 再分慢因。
3. 最后再决定优化动作。

如果没有这层分类，profiling 很容易变成“哪里红就点哪里”，最后忙得很努力，收益却很随机。

## 面试高频问法

### 初级

1. SIMD 和 SIMT 有什么直觉区别？
2. 为什么很多推理性能问题最终会落到 kernel 执行方式上？

### 中级

1. 为什么 decode 阶段的小 shape kernel 更容易被 launch 拖慢？
2. 为什么 FLOPs 不高的算子也可能很慢？

### 高级

1. 如果 GPU utilization 不低，但整体吞吐仍不理想，你会优先怀疑哪些 kernel 级问题？
2. 如何快速区分一个热点更像 GEMM、reduction，还是 pointwise 碎片问题？

## 易错点

- 只看 FLOPs，忽略带宽和访存模式
- 小 batch / 小 shape 下 launch 开销占主导
- 讨论并行时不区分单卡执行并行与多卡系统并行
- 只盯单个 kernel 耗时，不看 kernel 次数和整体碎片化

## 排查 checklist

- [ ] 单次 kernel 时间 vs kernel 次数，是不是太碎？
- [ ] 带宽是否接近峰值？如果很低，访问模式是否不连续？
- [ ] 是否能通过融合 / 合并算子减少 launch？
- [ ] 当前热点属于 GEMM、reduction、pointwise，还是 attention 类 kernel？

## 参考资料

- GPU 执行模型与 profiling 资料
- 建议串读：[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)、[`计算图、融合与调度`](04-graph-fusion-scheduling.md)

## CS336 对照

- Lecture 5（GPUs）
- Lecture 6（kernels, Triton）
- Lecture 7-8（parallelism）
