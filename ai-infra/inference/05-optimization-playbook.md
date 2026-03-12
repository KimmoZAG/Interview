# 推理优化 Playbook（定位→动作→验证）

## 要点

- 优化要闭环：**定位瓶颈 → 选择动作 → 验证正确性 → 验证性能曲线**
- 任何优化都要在“代表性 workload”上验证（不同长度/不同并发）
- CS336 把 inference 讲得很系统，一个重要启发是：优化动作应该按 prefill / decode / serving 调度分层，而不是混成一个“提速清单”

## 1) 定位（最小闭环）

- 先拆阶段：预处理 / prefill / decode / 后处理
- 再拆设备：CPU / GPU / I/O
- 工具：profiler（GPU/CPU）、trace、指标（metrics）

## 2) 动作（从高收益到低收益）

- 减少工作量：更短上下文、KV 复用、裁剪无用输出
- 减少 launch：融合、CUDA Graph（如果适用）、批处理
- 提升算子效率：更好 GEMM/attention kernel、layout 调整
- 降低精度：FP16/BF16/INT8/INT4（配合回归验证）
- 内存优化：内存池、预分配、减少中间张量

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

## 3) 验证

- 正确性：回归样例、误差阈值、线上灰度
- 性能：吞吐 + p95/p99 + 显存

## 一个更像 CS336 的优化闭环

1. 先问 workload 属于 prefill-heavy 还是 decode-heavy
2. 再问问题主要落在 compute、bandwidth、launch、queue 哪一类
3. 决定动作后，不只看平均值，还要看长度分桶和并发分桶
4. 最后确认提升是否建立在可持续的实现上，而不是只对某个 shape 有效

## 易错点

- 只在单一 shape 上优化，线上 shape 变了就回退
- 没有把编译/冷启动纳入总成本
- 看到 attention 慢就一股脑换 kernel，而不先分清是 prefill 还是 decode 在慢

## 排查 checklist

- [ ] 你的瓶颈假设能被 profiler 直接证伪/证实吗？
- [ ] 性能报告是否包含 p95/p99 与显存？
- [ ] 是否记录了软件栈版本（驱动/CUDA/库/框架）？
- [ ] 你当前的优化动作，针对的是 TTFT、TPOT、吞吐，还是尾延迟？

## CS336 对照

- 官方 lecture 对应：Lecture 10（inference）、Lecture 12（evaluation）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-10
  - https://bearbearyu1223.github.io/posts/cs336-building-a-complete-training-loop/
  - https://realwujing.github.io/page/3/
