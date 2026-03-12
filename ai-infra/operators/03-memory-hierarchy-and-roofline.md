# 内存层级与性能模型（Roofline）

## 要点

- 用 roofline 把问题先粗分：**带宽受限** vs **算力受限**
- 性能优化要“有证据”：先测，再改，再验证
- CS336 的 GPU / inference 相关内容都可以放回 roofline 视角重新理解：很多优化动作本质上是在改变 Bytes，而不是改变 FLOPs

## 内存层级（概念级）

- CPU：L1/L2/L3、NUMA、主存
- GPU：寄存器、shared memory、L2、HBM（以及可能的 L1/texture cache）

## Roofline 的最小用法

- 算术强度：$AI = \frac{FLOPs}{Bytes}$
- 经验法：
  - AI 低：多半带宽瓶颈 → 减少访存、提高复用、融合
  - AI 高：可能算力瓶颈 → 用更高效 kernel、使用更低精度、提高并行度

## 几个在 LLM 里常见的直觉例子

- 大 GEMM：通常 AI 较高，更可能接近算力瓶颈
- LayerNorm / Softmax：AI 往往更低，更容易受带宽影响
- Decode attention：虽然数学量不一定夸张，但反复读历史 KV，常常更偏 memory-bound
- Prefill attention：shape 大时更可能回到 compute-heavy 模式

这也是为什么同一个 attention，在 prefill 和 decode 的优化策略可能完全不同。

## 推理里常见现象

- Decode 阶段常见：小矩阵/小 batch → launch 与访存占主导
- Prefill 阶段常见：大 GEMM → 更接近算力瓶颈

## 把“优化动作”翻译成 roofline 语言

- 融合：减少中间张量写回，降低 Bytes
- 更好的 tile/blocking：提升数据复用，提高有效 AI
- FlashAttention：重排访存路径，减少不必要中间结果落到高层内存
- KV cache 量化：降低每次读取 Bytes
- 动态 batching：让小 shape 更接近能吃满硬件的区域

## 排查 checklist

- [ ] 你能给出这个算子的 AI 粗估吗？
- [ ] profiler 显示的 achieved bandwidth / achieved FLOPs 大概多少？
- [ ] 优化后是否同时验证“正确性 + 性能回归曲线（不同 shape）”？
- [ ] 你提出的优化动作，到底是在减少 Bytes，还是在提高有效 FLOPs 利用？

## CS336 对照

- 官方 lecture 对应：Lecture 5（GPUs）、Lecture 6（kernels, Triton）、Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://bearbearyu1223.github.io/posts/cs336-training-a-transformer-lm-part-1/
  - https://realwujing.github.io/page/3/
  - https://www.rajdeepmondal.com/blog/cs336-lecture-5
