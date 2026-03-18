# 常见层：Norm / 激活 / 残差 与数值稳定性

## 一句话先讲清

这页回答的是：**为什么 Norm、激活和残差这些“看起来不大”的层，既会影响深层模型的数值稳定性，也会在推理实现里变成一串讨厌但必须处理好的 pointwise 热点。**

它们不像大 GEMM 那样显眼，但常常决定“模型稳不稳”和“尾延迟丑不丑”。

## 关联知识网络

- 结构背景：[`Transformer 推理所需的最小知识`](01-transformer-minimum.md)
- 注意力成本：[`Attention、KV cache 与吞吐/延迟`](02-attention-kv-cache.md)
- 推理侧证据：[`可观测性与调试`](../02-inference-engine/06-observability-and-debugging.md)
- 算子融合：[`计算图融合与调度`](../01-operator-optimization/04-graph-fusion-scheduling.md)
- 量化背景：[`量化基础`](../01-operator-optimization/05-quantization-basics.md)

## 为什么值得单独学

- 低精度下的很多数值问题，不是出在 attention 大公式，而是出在 norm、softmax、激活近似和 accumulate 精度。
- 这些层通常是 memory-bound 的 pointwise 链，单个不大，串起来却能贡献不少 launch 和访存开销。
- 真正的工程难点常常是：**你想通过融合把它们加速，但又不能把数值稳定性搞飞。**

## 这一类层分别在干什么

### Norm

- 常见：LayerNorm、RMSNorm
- 作用：控制数值尺度，帮助深层网络稳定
- 系统关注点：reduction、同步、accumulate 精度、融合顺序

### 激活与门控

- 常见：GELU、SiLU、SwiGLU
- 作用：引入非线性，提高表达能力
- 系统关注点：近似实现、pointwise 开销、门控额外张量操作

### 残差

- 典型形式：`y = x + f(x)`
- 作用：让深层结构更稳定、更容易优化
- 系统关注点：能否和前后层融合，减少中间读写

## 为什么它们在系统里麻烦

| 层类型 | 模型侧价值 | 系统侧常见问题 |
|---|---|---|
| Norm | 控制尺度、提升稳定性 | reduction 带同步，低精度下易数值漂移 |
| 激活 | 提供非线性 | 近似实现差异、pointwise 链过长 |
| 残差 | 保证深层信息传递 | 与融合顺序、读写次数相关 |

这类层一个很典型的特征是：**算得不算重，搬得倒挺勤。**

## 最小工程直觉

一个常见场景是：

- 权重和激活都使用低精度
- norm 的 reduction 也跟着低精度累计

这时就更容易出现：

- 均值或方差估计不稳
- 输出轻微漂移
- 极端情况下出现 `inf` / `nan`

所以很多实现会在关键 reduction 上保留 FP32 accumulate。这个动作看起来“不够极致”，但往往是稳定性的保险丝。

## LayerNorm 和 RMSNorm 应该怎么记

| 方案 | 直觉差异 | 常见工程感受 |
|---|---|---|
| LayerNorm | 同时关心均值与方差 | 计算更完整，reduction 更重 |
| RMSNorm | 更关注尺度归一，不显式减均值 | 实现更简，常被认为更适合部分 LLM 结构 |

面试里不一定要把公式背完整，但最好能说出：**不同 norm 设计不仅影响训练稳定性，也会影响推理实现的成本与融合方式。**

## Troubleshooting：为什么融合后更快了，结果却轻微漂了

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 融合后吞吐提升，但回归样例漂移 | 浮点计算顺序改变 | 固定输入比较 max / mean error |
| 低精度下偶发 `inf/nan` | norm / softmax accumulate 精度不够 | 检查 reduction 是否保留 FP32 |
| 平均值更快，但 p99 没明显改善 | pointwise 链仍然碎，launch 还很多 | 看 kernel 数量与时间分布 |
| 某些模型更敏感 | 激活近似或门控实现不同 | 对照不同实现输出误差 |

### 一个排障顺序

1. 先用固定输入对比融合前后误差范围。
2. 再看有没有 `inf` / `nan` 或异常值扩散。
3. 检查 norm / softmax 等 reduction 是否仍保留更高精度 accumulate。
4. 最后再判断这次融合收益是否真的值得引入额外回归成本。

## 推理优化工程师视角

这页最核心的提醒是：**不是所有热点都长得像大 GEMM。**

- Norm / activation / residual 往往是 memory-bound。
- 单个 kernel 看起来不大，但串起来会带来很多 launch 与读写往返。
- 一旦做融合，就必须同时承担数值漂移验证成本。

因此这类层的优化永远是两条线一起抓：

1. **性能线**：减少 pointwise 碎片、减少中间读写、提升融合度。
2. **正确性线**：控制误差、检查 accumulate 精度、维护固定回归样例。

如果一个优化只换来一点吞吐提升，却让误差边界不清楚、回归成本暴涨，那它通常不是好优化。

## 面试高频问法

### 初级

1. LayerNorm 和 RMSNorm 在直觉上有什么差别？
2. 为什么激活和 norm 往往更像 memory-bound？

### 中级

1. 为什么低精度下 norm 常需要更高精度 accumulate？
2. 为什么 pointwise 融合虽然单次收益不大，却经常值得做？

### 高级

1. 如果融合后性能更好但输出漂移，你会先排查哪些数值因素？
2. 为什么一些大模型更偏爱 RMSNorm 这类设计？

## 易错点

- 低精度下 softmax 或 norm 数值不稳，导致输出漂移
- 融合后改变计算顺序，引入浮点非结合律带来的细微差异
- 只盯大 GEMM，不关注 pointwise 链对尾延迟和 kernel 数的影响

## 排查 checklist

- [ ] 使用固定输入比较融合前后的最大 / 均方误差
- [ ] 是否出现 `inf` / `nan`（尤其在低精度下）？
- [ ] norm / softmax 的 reduction 是否保留 FP32 accumulate？
- [ ] pointwise 链是否真的减少了 launch 和中间读写？

## 参考资料

- LayerNorm / RMSNorm 相关资料
- 激活函数与数值稳定性实践资料
- kernel 融合与低精度实现相关文档
