# 评测与基准：accuracy/latency/throughput

## 要点

- 评测需要同时覆盖：正确性（或质量）与性能（吞吐、延迟、尾延迟）
- 基准要可复现：固定硬件/软件版本、固定输入集、明确 warmup 与统计方法
- CS336 的一个核心提醒是：不同阶段的评测对象不一样，不能把 pretraining loss、产品质量、alignment 表现混成一个指标

## 指标

- 质量：任务集得分、人工评审、回归样例
- 性能：
  - throughput：tokens/s、req/s
  - latency：TTFT（首 token 时间）、TPOT（每 token 时间）、p50/p95/p99
  - 资源：显存占用、GPU 利用率、CPU 利用率

## 把评测拆成 4 层

从 CS336 后半程的视角，评测至少分 4 层：

1. 预训练评测：loss、perplexity、held-out validation
2. 能力评测：benchmark accuracy、task success、reasoning / coding / instruction following
3. 系统评测：TTFT、TPOT、吞吐、尾延迟、显存
4. 行为评测：helpfulness、harmlessness、format adherence、reward / preference 指标

很多团队的问题不是没做评测，而是把这 4 层混在一起汇报，最后无法定位退化到底来自模型、系统、还是 post-training。

## 基准设计

- 固定输入长度分桶：短/中/长
- 区分 prefill 与 decode
- 并发水平：1、N（接近线上）

再补几个 CS336 风格的要求：

- 明确基准是在训练阶段、离线推理阶段，还是在线 serving 阶段
- 明确 benchmark 与真实任务之间的映射关系
- 对 reasoning / alignment 场景，区分“会不会”与“是否稳定遵循格式/约束”

## 训练指标 vs 产品指标

一个非常常见的错配：

- loss/perplexity 下降，不代表聊天体验一定更好
- benchmark accuracy 提升，不代表在线延迟和成本可接受
- alignment 分数提升，不代表基础能力没有被破坏

因此真正有用的评测表，应该至少同时列出：

- 一个训练质量指标
- 一个目标任务指标
- 一个系统性能指标
- 一个资源成本指标

## 易错点

- 只做单条请求 benchmark，和线上差距巨大
- 忽略 warmup/缓存（CUDA graph、kernel cache、编译 cache）
- 只追单一 benchmark 分数，忽略了真实 workload 和行为约束

## 排查 checklist

- [ ] benchmark 脚本是否固定了版本与参数？
- [ ] 是否按长度分桶报告 p95/p99？
- [ ] 是否记录了运行时的 GPU/CPU/显存曲线？
- [ ] 你汇报的指标属于训练、能力、系统、行为中的哪一层？

## CS336 对照

- 官方 lecture 对应：Lecture 12（evaluation）、Lecture 15-17（alignment / RL）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-12
  - https://realwujing.github.io/page/3/
  - https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments
