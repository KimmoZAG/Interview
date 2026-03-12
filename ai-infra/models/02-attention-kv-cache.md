# Attention、KV cache 与吞吐/延迟

## 要点

- Attention 在 decode 阶段常见瓶颈：读取历史 KV（带宽）+ 小 shape 计算（launch/调度）
- KV cache 的组织方式（连续/分页/分块）会显著影响显存占用与吞吐
- 从 CS336 的视角看，attention 不是一个公式，而是一类随 workload 改变瓶颈形态的系统问题

## Attention 的推理关注点

- Prefill：典型是“全量 attention”，计算更重
- Decode：单 token query 与历史 KV 做 attention，主要成本在读 KV 与 softmax/加权求和

如果用复杂度直觉来抓重点：

- Prefill：更像批量矩阵乘，compute 更重
- Decode：每步只生成一个 token，但需要跨层读整段历史 KV，更容易 memory-bound

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

如果每个元素用 BF16/FP16 存储，则字节数再乘以 2；如果是 FP32，再乘以 4。

这也是为什么上下文长度、并发数和层数一上去，KV cache 会迅速成为线上显存主力。

## 连续布局 vs 分页布局

- 连续布局：实现简单，顺序访问友好，但扩容和碎片问题更明显
- 分页/分块布局：更容易做动态请求管理、减少大块拷贝，但索引和调度更复杂

工程上真正要问的不是“哪种更高级”，而是：

- 请求长度分布是否变化剧烈
- 是否存在大量中途结束/中途插入的新请求
- 你更怕的是 allocator 抖动，还是索引开销

## 吞吐/延迟的常见矛盾

- 吞吐：更依赖 batching（合并请求）
- 尾延迟：更敏感于排队、同步点、显存抖动、cache miss

一个常见误区是只看平均 tokens/s。在线服务里，真正难的是同时控制：

- TTFT：首 token 能否尽快出来
- TPOT：后续 token 是否稳定
- p95/p99：长请求和短请求混跑时是否抖动严重

## CS336 里 attention 最值得吸收的系统直觉

- FlashAttention 的核心不是“换一个公式”，而是减少中间结果写回、提升访存效率
- decode 阶段的瓶颈常常不在数学复杂度，而在 KV 读取与调度开销
- attention 的系统优化必须联动考虑 batching、cache layout、kernel 与 allocator

## 易错点

- 只看平均延迟不看 p95/p99
- 不同长度请求混在一起导致 padding 浪费或调度不稳定
- 只知道 KV cache 会占显存，但不会估算它到底为什么会爆

## 排查 checklist

- [ ] 统计 prefill 与 decode 的时间占比
- [ ] KV cache 的显存占用随并发增长曲线
- [ ] 是否存在频繁的 KV reallocation 或 copy
- [ ] 当前 workload 下更像 compute-bound 还是 memory-bound？

## CS336 对照

- 官方 lecture 对应：Lecture 6（kernels, Triton）、Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-10
  - https://realwujing.github.io/page/3/
  - https://github.com/anenbergb/LLM-from-scratch
