# 内存层级与性能模型（Roofline）

## 要点

- 用 roofline 把问题先粗分：**带宽受限** vs **算力受限**
- 性能优化要“有证据”：先测，再改，再验证
- CS336 的 GPU / inference 相关内容都可以放回 roofline 视角重新理解：很多优化动作本质上是在改变 Bytes，而不是改变 FLOPs
- roofline 最有价值的地方，不是公式本身，而是它能把“为什么慢”先粗分成一两类正确方向。

## 通用知识

### 它是什么

roofline 是一种性能分析视角，用来把一个算子的实际性能放到两类上限之间看：

- 算力上限
- 带宽上限

它不是为了算一个漂亮图，而是为了帮助你判断：

- 当前更像算力受限
- 还是更像内存带宽受限

### 它解决什么问题

它主要解决：

- 看到“慢”时，不知道该优先优化计算还是访存
- 不知道融合、tiling、量化、batching 这些动作本质在改变什么
- 不知道一个优化应该期待提升哪个方向

### 为什么在 AI 系统里重要

因为 LLM 里的热点非常不一样：

- GEMM 往往更像算力问题
- LayerNorm / Softmax 往往更像带宽问题
- decode attention 常常是典型 memory-bound

如果没有 roofline 这种语言，很多优化讨论就会退化成：“听说这个技巧很快，我们也试试。”

### 它的收益与代价

收益：

- 能快速缩小问题空间
- 能把优化动作翻译成更清晰的物理意义

代价：

- 它是粗粒度判断工具，不是替代 profiler 的万能钥匙
- 只能先告诉你方向，不会自动告诉你具体实现细节

## 内存层级（概念级）

- CPU：L1/L2/L3、NUMA、主存
- GPU：寄存器、shared memory、L2、HBM（以及可能的 L1/texture cache）

一句很实用的话是：

- 离计算单元越近的存储越快，也越小；离得越远，越大但越慢。

很多优化的本质，就是尽量让数据少走慢路，多待在快层里。

## Roofline 的最小用法

- 算术强度：$AI = \frac{FLOPs}{Bytes}$
- 经验法：
  - AI 低：多半带宽瓶颈 → 减少访存、提高复用、融合
  - AI 高：可能算力瓶颈 → 用更高效 kernel、使用更低精度、提高并行度

可以把它记成一句话：

- AI 低，多半先救 Bytes；AI 高，多半先救 FLOPs 利用率。

## 几个在 LLM 里常见的直觉例子

- 大 GEMM：通常 AI 较高，更可能接近算力瓶颈
- LayerNorm / Softmax：AI 往往更低，更容易受带宽影响
- Decode attention：虽然数学量不一定夸张，但反复读历史 KV，常常更偏 memory-bound
- Prefill attention：shape 大时更可能回到 compute-heavy 模式

这也是为什么同一个 attention，在 prefill 和 decode 的优化策略可能完全不同。

## 最小例子

假设两个热点：

1. 一个大 GEMM
2. 一个反复读 KV 的 decode attention

大 GEMM 常常更像：

- 数据复用较好
- 单次计算量大
- 更有机会接近算力上限

decode attention 常常更像：

- 每步都要读很多历史 KV
- shape 不大但 Bytes 很多
- 更容易先撞上带宽或访存问题

这就是为什么“同样都叫 attention / matmul”，优化动作却可能完全不同。

## 推理里常见现象

- Decode 阶段常见：小矩阵/小 batch → launch 与访存占主导
- Prefill 阶段常见：大 GEMM → 更接近算力瓶颈

## 把“优化动作”翻译成 roofline 语言

- 融合：减少中间张量写回，降低 Bytes
- 更好的 tile/blocking：提升数据复用，提高有效 AI
- FlashAttention：重排访存路径，减少不必要中间结果落到高层内存
- KV cache 量化：降低每次读取 Bytes
- 动态 batching：让小 shape 更接近能吃满硬件的区域

这是 roofline 真正好用的地方：

- 你不只是知道某个 trick 快，而是知道它快在“减少了什么”或者“提升了什么”。

## 工程例子

一个常见误判：

- 看到 decode 慢，就直接想换更强 GEMM kernel

但如果问题本质更像：

- KV 读取 Bytes 太多
- launch 太碎
- attention 更偏 memory-bound

那你换 GEMM kernel 可能几乎没有体感收益。

相反，如果是 prefill 大矩阵乘慢，图融合和 tile/blocking 优化往往更像正解。

## 推理优化工程师视角

对推理优化工程师来说，这篇最关键的价值是：

1. 先用 roofline 把慢因粗分方向
2. 再决定该看 kernel、layout、fusion、量化还是 batching
3. 不把所有问题都归结成“算力不够”

会这样做之后，你的优化动作通常会更有命中率。

## 常见面试问题

### 初级

1. 什么是算术强度？
2. roofline 为什么能帮助判断是带宽瓶颈还是算力瓶颈？

### 中级

1. 为什么 LayerNorm / Softmax 常比大 GEMM 更像 memory-bound？
2. 为什么 decode attention 常和 prefill attention 的优化方向不同？

### 高级

1. 如果一个优化声称“更快”，你如何用 roofline 语言解释它的收益？
2. 如果 profiler 显示带宽打不高、FLOPs 也不高，你会怀疑哪些执行问题？

## 排查 checklist

- [ ] 你能给出这个算子的 AI 粗估吗？
- [ ] profiler 显示的 achieved bandwidth / achieved FLOPs 大概多少？
- [ ] 优化后是否同时验证“正确性 + 性能回归曲线（不同 shape）”？
- [ ] 你提出的优化动作，到底是在减少 Bytes，还是在提高有效 FLOPs 利用？

## 参考资料

- roofline / GPU memory hierarchy 相关资料
- 建议串读：`02-kernel-execution-model.md`、`04-graph-fusion-scheduling.md`

## CS336 对照

- 官方 lecture 对应：Lecture 5（GPUs）、Lecture 6（kernels, Triton）、Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://bearbearyu1223.github.io/posts/cs336-training-a-transformer-lm-part-1/
  - https://realwujing.github.io/page/3/
  - https://www.rajdeepmondal.com/blog/cs336-lecture-5
