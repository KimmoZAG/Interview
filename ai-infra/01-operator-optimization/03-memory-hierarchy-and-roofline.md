# 内存层级与性能模型（Roofline）

## 一句话先讲清

Roofline 最有价值的地方，不是画出一张很酷的图，而是先把“为什么慢”粗分成两个方向：**更像算力受限，还是更像带宽受限。**

很多优化动作本质上不是在“让公式更高级”，而是在改变 **Bytes**、改变数据复用、或者让硬件更接近上限。

## 关联知识网络

- 输入与 layout：[`张量、shape 与内存布局`](01-tensors-shapes-layout.md)
- 执行模型：[`Kernel 与并行基础（SIMD/SIMT）`](02-kernel-execution-model.md)
- 图融合：[`计算图、融合与调度`](04-graph-fusion-scheduling.md)
- 量化：[`量化基础（INT8/INT4）与误差`](05-quantization-basics.md)
- 典型实例：[`FlashAttention 与 IO-aware Attention`](06-flashattention-io-aware.md)

## 为什么值得单独学

- 看到“慢”时，很多人第一反应是换更快 kernel，但问题可能根本不在算力侧。
- Roofline 能帮你先判断该优先优化计算还是访存。
- 它还能把融合、tiling、量化、batching 这些动作翻译成更清晰的物理意义。

## 它到底在回答什么

Roofline 是一种性能分析视角，用来把一个算子的实际表现放到两类上限之间看：

- 算力上限
- 带宽上限

它不是 profiler 的替代品，而是一个**先定方向**的工具。

## 内存层级，先有这个最小直觉

- CPU：L1 / L2 / L3 / NUMA / 主存
- GPU：寄存器 / shared memory / L2 / HBM（以及可能的 L1/texture cache）

一句很实用的话：**离计算单元越近的存储越快，也越小；离得越远，越大但越慢。**

很多优化的本质，就是尽量让数据少走慢路，多待在快层里。

## Roofline 的最小用法

算术强度：

$$
AI = \frac{FLOPs}{Bytes}
$$

一个够用的经验法：

- $AI$ 低：多半更像带宽瓶颈 → 先减少访存、提高复用、做融合
- $AI$ 高：更可能接近算力瓶颈 → 先看更高效 kernel、低精度、并行度与 tile 设计

把它记成一句话就行：**AI 低，多半先救 Bytes；AI 高，多半先救 FLOPs 利用率。**

## 在 LLM 里常见的几个直觉例子

| 热点 | 更常见的倾向 |
|---|---|
| 大 GEMM | 通常 AI 较高，更可能接近算力瓶颈 |
| LayerNorm / Softmax | AI 更低，更容易受带宽影响 |
| Decode attention | 反复读历史 KV，常更偏 memory-bound |
| Prefill attention | shape 大时更可能回到 compute-heavy 模式 |

这也是为什么同一个 attention，在 prefill 和 decode 的优化策略常常完全不同。

## 最小例子：为什么换 GEMM kernel 不一定救得了 decode

假设有两个热点：

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

## 把优化动作翻译成 Roofline 语言

| 优化动作 | 它更像在改善什么 |
|---|---|
| 融合 | 减少中间张量写回，降低 Bytes |
| 更好的 tile / blocking | 提高数据复用，提高有效 AI |
| FlashAttention | 重排访存路径，减少高层内存往返 |
| KV cache 量化 | 降低每次读取 Bytes |
| 动态 batching | 让小 shape 更接近能吃满硬件的区域 |

这就是 Roofline 真正好用的地方：你不只是知道某个 trick 快，而是知道它快在“减少了什么”或“提高了什么”。

## Troubleshooting：profiler 里 FLOPs 不高，带宽也不高，那到底慢在哪

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| FLOPs 不高，带宽也不高 | launch 太碎或同步太多 | 看 kernel 次数、单次时长、trace |
| decode 慢 | memory-bound + 小 shape | 分离 prefill / decode profile |
| 大 GEMM 不理想 | tile / kernel / 并行度没吃满 | 看 achieved FLOPs 与 occupancy |
| 优化后收益不稳 | 只对部分 shape 有效 | 看不同 shape 的回归曲线 |

### 一个排障顺序

1. 先粗估这个热点的 $AI$。
2. 再看 profiler 里的 achieved bandwidth / achieved FLOPs。
3. 然后判断你提出的优化到底是在减少 Bytes，还是在提高 FLOPs 利用。
4. 最后再去做具体 kernel、fusion、layout 或 batching 的动作选择。

## 推理优化工程师视角

这页最关键的价值是：

1. 先用 Roofline 把慢因粗分方向。
2. 再决定该看 kernel、layout、fusion、量化还是 batching。
3. 不把所有问题都归结成“算力不够”。

如果能做到这一步，你的优化动作通常会更有命中率，也更容易向别人解释清楚“为什么这么改”。

## 面试高频问法

### 初级

1. 什么是算术强度？
2. Roofline 为什么能帮助判断是带宽瓶颈还是算力瓶颈？

### 中级

1. 为什么 LayerNorm / Softmax 常比大 GEMM 更像 memory-bound？
2. 为什么 decode attention 和 prefill attention 的优化方向经常不同？

### 高级

1. 如果一个优化声称“更快”，你如何用 Roofline 语言解释它的收益？
2. 如果 profiler 显示带宽打不高、FLOPs 也不高，你会优先怀疑哪些执行问题？

## 易错点

- 把 Roofline 当成万能诊断器，而不是方向判断工具
- 看到 decode 慢就先换 GEMM kernel
- 不区分“减少 Bytes”和“提升 FLOPs 利用率”这两类优化

## 排查 checklist

- [ ] 你能给出这个算子的 $AI$ 粗估吗？
- [ ] profiler 显示的 achieved bandwidth / achieved FLOPs 大概多少？
- [ ] 优化后是否同时验证了正确性与不同 shape 的性能曲线？
- [ ] 你的优化动作，到底是在减少 Bytes，还是在提高有效 FLOPs 利用？

## 参考资料

- Roofline / GPU memory hierarchy 相关资料
- 建议串读：[`Kernel 与并行基础（SIMD/SIMT）`](02-kernel-execution-model.md)、[`计算图、融合与调度`](04-graph-fusion-scheduling.md)
