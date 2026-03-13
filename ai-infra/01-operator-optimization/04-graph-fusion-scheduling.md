# 计算图、融合与调度

## 要点

- 推理加速常见路径：**图级优化（融合/常量折叠）+ kernel 级优化（高效实现）**
- 融合的收益主要来自：减少中间张量写回、减少 kernel launch、提高缓存复用
- 从 CS336 的角度看，FlashAttention 是理解“融合为什么重要”的最好例子之一，因为它不是单纯把多个 op 拼一起，而是重新设计 IO 路径
- 真正该问的不是“有没有融合”，而是：**融合之后，中间张量、kernel 数量和访存路径到底有没有真正改善**。

## 通用知识

### 它是什么

计算图可以理解为：

- 节点是算子
- 边是张量

融合和调度则是在问：

- 这些算子能否合并得更紧凑
- 数据能否少写回、少读回
- 工作能否在更合适的 tile / block / memory reuse 方式下执行

### 它解决什么问题

它主要解决：

- 图上有很多小 op，导致 kernel 太碎
- 中间张量写回太多，内存流量过大
- 同一段子图明明逻辑简单，却被拆得支离破碎

### 为什么在 AI 系统里重要

因为很多推理性能问题根本不是“单个 kernel 不够快”，而是：

- 图被拆得太散
- 中间结果落地太多
- launch 次数太多
- tile / blocking / buffer reuse 没设计好

### 它的收益与代价

收益：

- 减少中间张量访存
- 减少 launch 次数
- 提高 cache / shared memory 复用

代价：

- 数值误差可能变化
- 动态 shape 会让融合和缓存策略更复杂
- 调试难度通常会上升

## 计算图（抽象）

- Op graph：节点是算子，边是张量
- 关键属性：shape/dtype/layout、是否动态 shape、是否可重排

从工程上看，一个图如果想融合得好，通常要先满足两件事：

- 子图边界相对稳定
- shape / dtype / layout 没有太多阻碍重排的因素

## 融合的常见形态

- pointwise 链：`bias + gelu + dropout`（推理里 dropout 通常关）
- Norm + scale/bias
- attention 子图（QKV 投影、softmax、matmul）中的部分融合

## 图级融合 vs kernel 级融合

这是很容易混淆的一组概念：

- 图级融合：从 op graph 角度把多个节点合并，减少图上的小 op
- kernel 级融合：最终执行时让更多工作在一次 launch 中完成

两者通常相关，但不是一回事。你可能：

- 在图上做了融合，但底层仍然落成多个 kernel
- 或者看似是一个高层 op，但底层其实拆成很多 kernel

因此，真正的验证不能只看 graph IR，还要看 profiler 里的 kernel 数和中间张量访存。

## 调度（scheduling）关注点

- tile/blocking（提高数据复用）
- 并行策略（线程块映射）
- 内存分配与重用（buffer reuse）

调度真正关心的是：

- 哪些数据应该尽量留在快存储里
- 哪些工作应该被一起做完
- 哪些中间结果不该频繁落回高层内存

## 最小例子

考虑一段简单子图：

- `matmul -> bias add -> gelu`

如果不做融合：

- matmul 一个 kernel
- bias add 一个 kernel
- gelu 一个 kernel
- 中间结果可能被写回两次

如果做了更好的融合和调度：

- bias add / gelu 可以尽量贴着主计算执行
- 中间张量写回减少
- launch 次数减少

这就是融合最朴素也最实用的收益来源。

## FlashAttention 为什么是理解调度的经典例子

它最值得学的不是具体实现细节，而是这几个原则：

- 不要急着把中间 attention matrix 全部写回高层内存
- 尽量在 tile 内完成更多局部计算与归约
- 让 softmax 的数值稳定计算与访存路径一起设计

换句话说，FlashAttention 的“快”不是来自一个数学近似，而是来自 **IO-aware scheduling**。

## 工程例子

一个典型现象：

- 图上看已经“融合”了
- 但 profiler 里 kernel 数量并没有明显减少

这通常意味着：

- 图级表示层面看起来更紧凑了
- 但底层执行并没有真正合成更少的 kernel
- 或者动态 shape / layout 约束让部分融合无法稳定落地

这也是为什么工程里一定要同时看：

- graph/IR
- kernel 数
- 中间张量访存

只看其中一个，很容易误判。

## 推理优化工程师视角

对推理优化工程师来说，这篇最核心的价值是：

1. 不把融合当成“编译器 magic”
2. 要把融合收益翻译成：更少 kernel、更少 Bytes、更少写回
3. 任何“融合已生效”的结论，都最好由 profiler 和误差对比一起支持

## 常见面试问题

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

- 融合后数值误差变化（尤其低精度）
- 动态 shape 导致某些优化失效或需要多个编译 cache
- 只在图层面宣称“已经融合”，却没有在 kernel 层验证 launch 是否真的减少
- 只盯 kernel 数，不看 Bytes 和数值行为是否也一起改善

## 排查 checklist

- [ ] 图中是否存在大量小 op（pointwise）导致 launch 爆炸？
- [ ] 是否能接受对部分子图做 ahead-of-time 编译缓存？
- [ ] 融合前后是否对比了中间张量的最大误差？
- [ ] profiler 里 kernel 数量、时间分布、Bytes 真的下降了吗？

## 参考资料

- 图编译 / kernel fusion / scheduling 相关资料
- 建议串读：`03-memory-hierarchy-and-roofline.md`、`06-flashattention-io-aware.md`

## CS336 对照

- 官方 lecture 对应：Lecture 6（kernels, Triton）、Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://github.com/anenbergb/LLM-from-scratch
  - https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments
  - https://www.rajdeepmondal.com/blog/cs336-lecture-6
