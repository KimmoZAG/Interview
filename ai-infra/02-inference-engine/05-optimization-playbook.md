# 推理优化 Playbook（定位→动作→验证）

## 核心定义（What & Why）

> **一句话总结**：推理优化 Playbook 是一套把“慢了”拆成可定位、可选择动作、可验证收益的系统诊断方法，它解决的是线上推理问题复杂且易误判的现实痛点。

## 关联知识网络

- 前置：[`LLM Serving`](04-llm-serving.md)
- 前置：[`可观测性与调试`](06-observability-and-debugging.md)
- 平行：[`Paged KV 与 Allocator`](07-paged-kv-and-allocator.md)
- 平行：[`Attention 与 KV Cache`](../03-llm-architecture/02-attention-kv-cache.md)
- 课程桥接：[`CS336 / 10 推理优化`](../../cs336/10-inference.md)
- 课程桥接：[`CS336 / 06 Kernel、Profiling 与 Triton`](../../cs336/06-kernels-and-triton.md)

## 要点

- 优化要闭环：**定位瓶颈 → 选择动作 → 验证正确性 → 验证性能曲线**
- 任何优化都要在“代表性 workload”上验证（不同长度/不同并发）
- CS336 把 inference 讲得很系统，一个重要启发是：优化动作应该按 prefill / decode / serving 调度分层，而不是混成一个“提速清单”
- 最危险的优化方式，是在没定位清楚之前就开始“上手段”；那通常只会把问题从一个角落推到另一个角落。

## 通用知识

### 它是什么

Playbook 不是一份“万能优化招式表”，而是一套工作方法：

- 先拆问题
- 再提假设
- 再用证据验证
- 最后确认收益能在真实 workload 上成立

### 它解决什么问题

它解决的是：

- 推理性能问题往往很复杂，不能靠直觉乱改
- 同一个“慢”，可能分别来自 queue、prefill、decode、allocator、kernel 或 CPU 前处理
- 单次 benchmark 的收益，不一定能迁移到线上并发环境

### 为什么在 AI 系统里重要

因为 LLM 推理系统不是单个算子，而是一整条链路：

- 请求排队
- tokenize
- prefill
- decode
- cache 管理
- 后处理

所以“优化”如果不先拆阶段，很容易出现：

- 把 tail latency 问题误当成算子效率问题
- 把 queue 问题误当成 attention kernel 问题
- 把 cold start 问题误当成 steady-state 性能问题

### 它的收益与代价

收益：

- 减少无效优化
- 更容易快速收敛到根因
- 更容易和团队同步同一套定位语言

代价：

- 前期需要花时间做观测与分解
- 不能一上来就追求“先改再说”的快感

## 1) 定位（最小闭环）

- 先拆阶段：预处理 / prefill / decode / 后处理
- 再拆设备：CPU / GPU / I/O
- 工具：profiler（GPU/CPU）、trace、指标（metrics）

最重要的一句是：

- 不要先问“哪个优化最强”，先问“问题究竟发生在哪一段链路”。

## 2) 动作（从高收益到低收益）

- 减少工作量：更短上下文、KV 复用、裁剪无用输出
- 减少 launch：融合、CUDA Graph（如果适用）、批处理
- 提升算子效率：更好 GEMM/attention kernel、layout 调整
- 降低精度：FP16/BF16/INT8/INT4（配合回归验证）
- 内存优化：内存池、预分配、减少中间张量

这些动作本身没有对错，错的是：

- 没有先判断问题更像 compute-bound、bandwidth-bound、launch-bound，还是 queue-bound

## 按阶段拆优化动作

### 如果 TTFT 高

优先看：

- tokenize / 输入处理是否过慢
- prefill shape 是否异常大
- 动态 batching 是否把短请求拖进长请求队列
- 首轮 graph compile / kernel cache 是否污染了结果

### 如果 TPOT 高

优先看：

- decode attention 是否被 KV 读取拖慢
- 小 shape kernel 是否太碎
- continuous batching 是否有效
- KV cache layout / allocator 是否抖动

### 如果 p99 高

优先看：

- 请求长度分布是否严重混杂
- 队列策略是否让长请求拖慢短请求
- 显存碎片与临时 reallocation 是否造成尾部抖动

### 如果吞吐低但平均延迟不高

优先看：

- batch 是否太小
- 请求合并策略是否保守
- GPU 是否被 CPU / I/O / host scheduling 卡住

## 对比表：常见线上症状应该先怀疑什么

| 现象 | 更像哪类问题 | 第一批该看的证据 | 常见误判 |
|---|---|---|---|
| TTFT 高，TPOT 正常 | queue / tokenize / prefill 问题 | 分阶段时延、长度分桶、trace | 一上来就换 decode attention kernel |
| TPOT 高，TTFT 正常 | decode / KV / allocator / 小 kernel 问题 | decode profile、KV 占用、kernel 次数 | 误以为整条链都慢 |
| p99 差，平均值正常 | 请求混跑 / 尾部抖动 / 碎片问题 | p95/p99、长度桶、显存波动 | 只看平均 tokens/s |
| 吞吐低但平均延迟不高 | batch 合并 / CPU 前处理 / host 调度问题 | QPS、batch size、CPU 利用率 | 盲目上更激进量化或大 kernel |

## 最小例子

假设线上现象是：

- 用户抱怨“首 token 慢”
- 但一旦开始出 token，整体生成速度还行

这时一个合格的 Playbook 不会直接说“去换 attention kernel”，而是会先问：

1. TTFT 和 TPOT 是否分开统计了？
2. queue / tokenize / prefill 哪一段最重？
3. 是否有长 prompt 请求把短请求拖住了？
4. 是否存在首次 graph compile / engine build 污染结果？

如果最后发现问题主要在 queue 和 prefill，那动作可能是：

- 调整 batching 窗口
- 分桶请求长度
- 预热 compile cache

而不是直接去优化 decode attention。

这就是 Playbook 的价值：

- 它帮助你避免“优化了一个看起来很高级，但根本不是瓶颈的东西”。

## 3) 验证

- 正确性：回归样例、误差阈值、线上灰度
- 性能：吞吐 + p95/p99 + 显存

再补一条非常重要的经验：

- 如果一个优化只在单一 shape、单并发、单请求下成立，那它很可能只是 benchmark 优化，不是系统优化

## 一个更像 CS336 的优化闭环

1. 先问 workload 属于 prefill-heavy 还是 decode-heavy
2. 再问问题主要落在 compute、bandwidth、launch、queue 哪一类
3. 决定动作后，不只看平均值，还要看长度分桶和并发分桶
4. 最后确认提升是否建立在可持续的实现上，而不是只对某个 shape 有效

## 工程例子

一个典型场景：

- 量化后平均 tokens/s 提升了
- 但 p99 和 TTFT 没有改善，甚至更差

一个成熟的 Playbook 会继续追问：

- 这次收益是不是主要来自 steady-state decode？
- 是否把 prefill、queue 或 cache 抖动掩盖掉了？
- 长短请求分桶后，提升是否仍然存在？

如果这些问题答不上来，那“性能提升”这句话通常还不完整。

## 💥 实战踩坑记录（Troubleshooting）

> 现象：量化后单请求 benchmark 漂亮了，但线上 TTFT 和 p99 反而变差。

- **误判**：只看 steady-state tokens/s，就宣布“优化成功”。
- **根因**：量化收益主要落在某些 decode kernel 上，但 queue、prefill、长度混跑和 allocator 抖动仍在伤用户体验。
- **解决动作**：
  - 先把 TTFT、TPOT、吞吐、p95/p99 分开；
  - 再按输入长度 / 输出长度 / 并发分桶；
  - 最后确认收益是端到端成立，而不是只在某一个微基准 shape 下成立。
- **复盘**：真正的系统优化，要防止“一个局部指标更好，用户体感却更差”。

## 推理优化工程师视角

对推理优化工程师来说，这篇最想建立的不是技巧，而是习惯：

1. 所有优化都先分阶段
2. 所有瓶颈都先要证据
3. 所有收益都要分 workload 验证
4. 所有改动都要考虑是否只是把问题转移了

会这样工作之后，你做的就不再是“性能调参”，而更像是系统诊断。

## 常见面试问题

### 初级

1. 为什么推理优化不能只看一个总 latency 指标？
2. 为什么优化前要先区分 prefill 和 decode？

### 中级

1. 如果 TTFT 高但 TPOT 正常，你的定位顺序是什么？
2. 为什么单一 shape benchmark 的提升不一定能迁移到线上？

### 高级

1. 如果一个优化提升了平均吞吐，却伤了 p99，你怎么判断是否值得保留？
2. 如何判断问题更像 queue-bound、launch-bound，还是 bandwidth-bound？

## 易错点

- 只在单一 shape 上优化，线上 shape 变了就回退
- 没有把编译/冷启动纳入总成本
- 看到 attention 慢就一股脑换 kernel，而不先分清是 prefill 还是 decode 在慢
- 把“有收益”理解成“在所有 workload 上都有收益”

## 排查 checklist

- [ ] 你的瓶颈假设能被 profiler 直接证伪/证实吗？
- [ ] 性能报告是否包含 p95/p99 与显存？
- [ ] 是否记录了软件栈版本（驱动/CUDA/库/框架）？
- [ ] 你当前的优化动作，针对的是 TTFT、TPOT、吞吐，还是尾延迟？

## 参考资料

- profiling / tracing / serving 调优资料
- 各类 runtime 与量化优化实践
- 建议串读：`06-observability-and-debugging.md`、`04-llm-serving.md`、`07-paged-kv-and-allocator.md`

## CS336 对照

- 官方 lecture 对应：Lecture 10（inference）、Lecture 12（evaluation）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-10
  - https://bearbearyu1223.github.io/posts/cs336-building-a-complete-training-loop/
  - https://realwujing.github.io/page/3/
