# 推理栈全景：前端→图→kernel→执行

## 要点

- 把推理系统拆成 4 层更容易定位问题：**模型前端 → 中间表示(IR/图) → 编译/优化 → 执行时(runtime)**
- 性能与稳定性问题，往往发生在“层与层交界处”（shape/dtype/layout/动态批处理）
- 从 CS336 Lecture 10 的视角看，推理系统还必须显式分成 **prefill** 与 **decode** 两种负载，否则你会把两个完全不同的问题混在一起

## 典型数据流

1. 前端（PyTorch/TF/JAX/自研）：定义模型与权重
2. 导出/表示：ONNX / TorchScript / StableHLO 等
3. 优化与编译：图优化、算子选择、融合、量化、代码生成
4. Runtime：内存管理、kernel 调度、stream 同步、并发与 batching

## 再细一层：LLM 推理的最小请求生命周期

对于大语言模型，更实用的链路通常是：

1. 请求进入
2. tokenize / 输入校验
3. prefill
4. decode 循环
5. detokenize / 后处理
6. 日志、指标、缓存回收

这条链路里，prefill 与 decode 往往需要被分开观测、分开优化、分开汇报。

## 你需要能画出来的“最小架构图”（建议你后续补）

- 请求进入 → 预处理(tokenize) → prefill → decode 循环 → 后处理
- 动态 batching 在哪里做？cache 在哪里存？

## 为什么 CS336 会把 inference 单独拎出来讲

因为在线推理不是“训练时 forward 的一个子集”，而是一套独立系统：

- 训练更关心吞吐和大 batch
- 推理更关心 TTFT、TPOT、队列、cache 和尾延迟
- 训练的热点不一定等于推理的热点

一个很有用的心智模型是：

- prefill 更像 dense compute 系统问题
- decode 更像缓存、调度和访存系统问题

## 易错点

- 只在模型层看问题，忽略 runtime 的同步与内存抖动
- 只看单次推理，忽略并发与排队造成的尾延迟
- 不区分 prefill 和 decode，导致定位和优化动作都偏了

## 排查 checklist

- [ ] 把耗时拆解到：预处理 / prefill / decode / 后处理
- [ ] 确认“图优化是否生效”（kernel 数量是否下降）
- [ ] 确认“shape/dtype/layout 是否符合预期”
- [ ] 指标是否分开统计了 TTFT、TPOT、吞吐、p95/p99？

## CS336 对照

- 官方 lecture 对应：Lecture 10（inference）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
	- https://www.rajdeepmondal.com/blog/cs336-lecture-10
	- https://rd.me/cs336
	- https://realwujing.github.io/page/3/
