# 计算图、融合与调度

## 一句话先讲清

真正该问的不是“有没有融合”，而是：**融合之后，中间张量、kernel 数量和访存路径到底有没有真的改善。**

从系统视角看，融合和调度的核心目标很朴素：**少写回、少 launch、多复用。**

## 关联知识网络

- 输入与 shape：[`张量、shape 与内存布局`](01-tensors-shapes-layout.md)
- 执行模型：[`Kernel 与并行基础（SIMD/SIMT）`](02-kernel-execution-model.md)
- Roofline：[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)
- 典型例子：[`FlashAttention 与 IO-aware Attention`](06-flashattention-io-aware.md)
- 编译视角：[`图编译：TVM / MLIR / XLA`](../02-inference-engine/03-graph-compiler-tvm-mlir-xla.md)

## 为什么值得单独学

- 很多推理性能问题根本不是单个 kernel 不够快，而是图被拆得太碎。
- 图融合的收益通常来自减少中间张量写回、减少 launch 次数、提高缓存和 shared memory 复用。
- FlashAttention 之所以经典，就是因为它不是简单把 op 粘一起，而是重新设计了 IO 路径和局部调度。

## 计算图和调度，分别在关心什么

### 计算图

- 节点是算子
- 边是张量
- 关键属性是 shape / dtype / layout / 是否动态 shape

### 调度（scheduling）

- tile / blocking 如何设计
- 线程块如何映射
- buffer 如何复用
- 哪些中间结果不该频繁落回高层内存

从工程上看，一个子图如果想融合得好，通常至少要满足：

- 子图边界相对稳定
- shape / dtype / layout 没有太多阻碍重排的因素

## 图级融合 vs kernel 级融合，别混了

| 概念 | 它在做什么 |
|---|---|
| 图级融合 | 从 op graph 角度把多个节点合并，减少图上的小 op |
| kernel 级融合 | 最终执行时让更多工作在一次 launch 中完成 |

两者通常相关，但不是一回事。你可能会遇到：

- 图上看已经融合了，但底层还是多个 kernel
- 高层看似是一个 op，底层却仍然拆得很碎

所以真正的验证不能只看 graph IR，还要看 profiler 里的 kernel 数和中间张量访存。

## 最常见、最值得融合的场景

- pointwise 链：`bias + gelu (+ dropout)`
- norm + scale / bias
- attention 子图中的部分融合

一句经验法：**pointwise 链越长、kernel 越碎，就越值得优先考虑融合。**

## 最小例子：`matmul -> bias add -> gelu`

如果不做融合：

- matmul 一个 kernel
- bias add 一个 kernel
- gelu 一个 kernel
- 中间结果可能被写回两次

如果做了更好的融合和调度：

- bias add / gelu 尽量贴着主计算执行
- 中间张量写回减少
- launch 次数减少

这就是融合最朴素、也最实用的收益来源。

## FlashAttention 为什么是理解调度的经典例子

它最值得学的不是具体实现细节，而是三个原则：

- 不要急着把中间 attention matrix 全部写回高层内存
- 尽量在 tile 内完成更多局部计算与归约
- 让 softmax 的数值稳定计算和访存路径一起设计

换句话说，FlashAttention 的“快”，本质上是 **IO-aware scheduling**。

## Troubleshooting：为什么 IR 上看起来融合了，性能却没起色

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| graph / IR 看起来更紧凑 | 底层并没真正合成更少 kernel | 看 profiler 中 kernel 数量 |
| kernel 数少了但收益有限 | Bytes 没明显下降 | 看中间张量写回和带宽 |
| 某些 shape 好，某些 shape 差 | 动态 shape 让融合难稳定落地 | 看不同 shape 的路径与 cache |
| 性能变好了但输出飘了 | 融合改变了数值路径 | 对比误差与回归样例 |

### 一个排障顺序

1. 先看图上是否存在大量小 op 导致 launch 爆炸。
2. 再看融合前后 kernel 数量是否真的下降。
3. 然后看中间张量写回和 Bytes 是否一起下降。
4. 最后同时验证误差变化，别只盯性能数字。

## 推理优化工程师视角

这页最核心的价值是：

1. 不把融合当成“编译器 magic”。
2. 要把融合收益翻译成：更少 kernel、更少 Bytes、更少写回。
3. 任何“融合已生效”的结论，最好都由 profiler 和误差对比一起支持。

如果只能在图层面说“已经融合”，却没法在 kernel 层和访存层证明收益，那这个结论通常还不够硬。

## 面试高频问法

### 初级

1. 图级融合和 kernel 级融合有什么区别？
2. 为什么融合通常能带来性能提升？

### 中级

1. 为什么 pointwise 链特别适合融合？
2. 为什么动态 shape 会让融合更复杂？

### 高级

1. 如果 IR 上看起来已经融合，但性能并没有改善，你会先检查什么？
2. 为什么 FlashAttention 经常被当成“融合 + 调度”思维的代表例子？

## 易错点

- 融合后数值误差变化，尤其在低精度下
- 动态 shape 导致某些优化失效或需要多个编译 cache
- 只在图层面宣称“已经融合”，却没有在 kernel 层验证 launch 是否真的减少
- 只盯 kernel 数，不看 Bytes 和数值行为是否一起改善

## 排查 checklist

- [ ] 图中是否存在大量小 op（尤其 pointwise）导致 launch 爆炸？
- [ ] 是否能接受对部分子图做 ahead-of-time 编译缓存？
- [ ] 融合前后是否对比了中间张量误差？
- [ ] profiler 里 kernel 数量、时间分布、Bytes 真的下降了吗？

## 参考资料

- 图编译 / kernel fusion / scheduling 相关资料
- 建议串读：[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)、[`FlashAttention 与 IO-aware Attention`](06-flashattention-io-aware.md)
