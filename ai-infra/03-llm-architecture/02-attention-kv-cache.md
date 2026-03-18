# Attention、KV cache 与吞吐/延迟

## 核心定义（What & Why）

> **一句话总结**：Attention 与 KV cache 共同决定了 LLM 推理阶段“算什么、存什么、读什么”，它们解决的是自回归生成中的重复计算问题，同时也把显存、带宽和尾延迟问题一并带了进来。

## 关联知识网络

- 前置：[`Transformer 最小知识`](01-transformer-minimum.md)
- 平行：[`FlashAttention 与 IO-aware`](../01-operator-optimization/06-flashattention-io-aware.md)
- 延伸：[`LLM Serving`](../02-inference-engine/04-llm-serving.md)
- 工程落地：[`Paged KV 与 Allocator`](../02-inference-engine/07-paged-kv-and-allocator.md)
- 多卡关联：[`并行训练策略`](../04-communication/01-training-parallelism.md)

## 要点

- Attention 在 decode 阶段常见瓶颈：读取历史 KV（带宽）+ 小 shape 计算（launch/调度）
- KV cache 的组织方式（连续/分页/分块）会显著影响显存占用与吞吐
- 从 CS336 的视角看，attention 不是一个公式，而是一类随 workload 改变瓶颈形态的系统问题

## 通用知识

### 它是什么

在自回归 LLM 中，attention 负责让当前 token 读取历史上下文信息。为了避免每步都重复计算历史 token 的 K/V，系统会把每层的 Key 和 Value 缓存在 KV cache 中。

### 它解决什么问题

如果没有 KV cache：

- 每生成一个新 token，都要重新计算整段上下文的 K/V

有了 KV cache：

- 历史 token 的 K/V 可以复用
- 新 token 只需计算自己的 Q/K/V，再与历史 cache 交互

### 为什么在 AI 系统里重要

因为 decode 阶段的瓶颈经常不再是“算 attention 公式”，而是：

- 读历史 KV 要读多少字节
- 这些 KV 如何布局
- 小 shape attention kernel 是否被 launch 和访存拖慢

### 它的收益与代价

收益：

- 避免重复计算历史上下文
- 让自回归生成在工程上可行

代价：

- KV cache 会迅速吞掉显存
- layout / allocator / batching 会直接影响吞吐与尾延迟

## Attention 的推理关注点

- Prefill：典型是“全量 attention”，计算更重
- Decode：单 token query 与历史 KV 做 attention，主要成本在读 KV 与 softmax/加权求和

补一个常用直觉：

- Prefill 更像一次大矩阵计算问题
- Decode 更像一次高频、小 shape、带状态的访存问题

## KV cache

- 存什么：每层的 K/V
- 为什么需要：自回归生成避免重复计算历史 token 的 K/V
- 关注项：
  - dtype（FP16/BF16/INT8 等）
  - layout（便于连续读取）
  - 分配策略（避免频繁扩容/拷贝）

一个最常用的显存估算：

- 单层 KV cache 元素数约为：$2 \times B \times S \times H$
- 全模型 KV cache 元素数约为：$2 \times B \times S \times H \times L$

## 最小例子

假设：

- batch = 8
- seq = 4096
- hidden = 4096
- layers = 32
- KV 用 FP16/BF16 存储（2 字节）

则全模型 KV cache 字节数粗略为：

$$
2 \times 8 \times 4096 \times 4096 \times 32 \times 2
$$

这里只看数量级就足够说明一件事：

- 上下文长度、并发数、层数一上去，KV cache 很快就会成为线上显存大户

## 连续布局 vs 分页布局

- 连续布局：实现简单，顺序访问友好，但扩容和碎片问题更明显
- 分页/分块布局：更容易做动态请求管理、减少大块拷贝，但索引和调度更复杂

## 对比表

| KV 布局 | 优点 | 代价 | 更适合的场景 |
|---|---|---|---|
| 连续布局 | 访问简单、实现直接 | 扩容和大块拷贝成本高，碎片更明显 | 固定 batch、长度分布较稳定 |
| 分页 / 分块布局 | 更适合动态请求与复用，减少大块搬运 | 索引、调度和 allocator 复杂度更高 | 在线 serving、长短请求混跑 |

## 吞吐/延迟的常见矛盾

- 吞吐：更依赖 batching（合并请求）
- 尾延迟：更敏感于排队、同步点、显存抖动、cache miss

## 工程例子

一个典型现象是：

- 模型参数不算太夸张
- 但一旦并发提高，系统突然显存吃紧、TPOT 变差

常见原因不是权重本身，而是：

- 并发数增大导致 KV cache 成倍增长
- decode 每步都要更频繁地读大块历史 KV
- allocator / layout 不稳时，尾延迟会进一步被放大

## 💥 实战踩坑记录（Troubleshooting）

> RuntimeError: CUDA out of memory while allocating KV cache

- **现象**：模型权重可以正常加载，但并发一高就出现 OOM，且 TPOT 明显变差。
- **误判**：最开始以为是 batch 太大或者权重量超了，想先压缩权重精度。
- **根因**：真正失控的是 KV cache，而不是模型参数本身；并发数和上下文长度一起上来后，KV 访存与显存占用快速膨胀。
- **解决动作**：
  - 先估算 KV cache 的理论显存占用；
  - 再分开看 prefill 与 decode 的资源模式；
  - 最后检查是连续扩容导致拷贝抖动，还是 allocator 碎片放大了问题。
- **复盘**：看到“高并发 + 长上下文 + OOM / TPOT 上升”这个组合，第一反应就该想到 KV cache，而不是只盯权重。

## 推理优化工程师视角

如果你做的是推理优化，而不是只看模型公式，这一章最重要的不是“attention 会算什么”，而是先判断当前瓶颈属于哪一类：

- prefill 还是 decode
- 算子本身慢，还是 KV 读取慢
- kernel 设计问题，还是 cache layout / allocator / batching 问题

一个很实用的工作顺序是：

1. 先分开统计 prefill 和 decode 的时间占比
2. 再估算 KV cache 显存与带宽压力是否已经逼近硬件上限
3. 最后才决定优先换 attention kernel、改 page/block 设计，还是先改调度与 batching

很多“attention 优化收益不明显”的根因，并不是 kernel 没写好，而是系统已经被 KV cache 访存、碎片、请求混跑和尾延迟放大效应卡住了。

## 常见面试问题

### 初级

1. KV cache 存的是什么？为什么 LLM 推理需要它？
2. 为什么 decode 阶段常比 prefill 更像 memory-bound？

### 中级

1. 为什么上下文长度和并发数会直接影响显存？
2. 连续布局和分页布局的主要 trade-off 是什么？

### 高级

1. 如果 TPOT 很高，你如何判断问题更像 attention kernel 本身，还是 KV cache 读取？
2. 为什么有时换更快的 attention kernel，系统整体收益仍然有限？

## 易错点

- 只看平均延迟不看 p95/p99
- 不同长度请求混在一起导致 padding 浪费或调度不稳定
- 只知道 KV cache 会占显存，但不会估算它到底为什么会爆
- 把 KV cache 完全当成模型问题，忽略它同时也是 serving 系统状态问题

## 排查 checklist

- [ ] 统计 prefill 与 decode 的时间占比
- [ ] KV cache 的显存占用随并发增长曲线
- [ ] 是否存在频繁的 KV reallocation 或 copy
- [ ] 当前 workload 下更像 compute-bound 还是 memory-bound？

## 参考资料

- Attention / FlashAttention 相关资料
- vLLM / TensorRT-LLM 的 KV cache 设计资料
- 推理系统和 allocator 相关工程总结
