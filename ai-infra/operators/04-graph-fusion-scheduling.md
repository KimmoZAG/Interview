# 计算图、融合与调度

## 要点

- 推理加速常见路径：**图级优化（融合/常量折叠）+ kernel 级优化（高效实现）**
- 融合的收益主要来自：减少中间张量写回、减少 kernel launch、提高缓存复用
- 从 CS336 的角度看，FlashAttention 是理解“融合为什么重要”的最好例子之一，因为它不是单纯把多个 op 拼一起，而是重新设计 IO 路径

## 计算图（抽象）

- Op graph：节点是算子，边是张量
- 关键属性：shape/dtype/layout、是否动态 shape、是否可重排

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

## FlashAttention 为什么是理解调度的经典例子

它最值得学的不是具体实现细节，而是这几个原则：

- 不要急着把中间 attention matrix 全部写回高层内存
- 尽量在 tile 内完成更多局部计算与归约
- 让 softmax 的数值稳定计算与访存路径一起设计

换句话说，FlashAttention 的“快”不是来自一个数学近似，而是来自 **IO-aware scheduling**。

## 易错点

- 融合后数值误差变化（尤其低精度）
- 动态 shape 导致某些优化失效或需要多个编译 cache
- 只在图层面宣称“已经融合”，却没有在 kernel 层验证 launch 是否真的减少

## 排查 checklist

- [ ] 图中是否存在大量小 op（pointwise）导致 launch 爆炸？
- [ ] 是否能接受对部分子图做 ahead-of-time 编译缓存？
- [ ] 融合前后是否对比了中间张量的最大误差？
- [ ] profiler 里 kernel 数量、时间分布、Bytes 真的下降了吗？

## CS336 对照

- 官方 lecture 对应：Lecture 6（kernels, Triton）、Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
	- https://github.com/anenbergb/LLM-from-scratch
	- https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments
	- https://www.rajdeepmondal.com/blog/cs336-lecture-6
