# Collectives：all-reduce / all-gather / reduce-scatter / all-to-all

## 核心定义（What & Why）

> **一句话总结**：Collectives 是多卡系统里最常见的一组协同通信原语，它们解决的是“多张卡如何交换与汇总数据”，也是并行训练和多卡推理性能最容易暴露瓶颈的地方。

## 关联知识网络

- 前置：[`通信基础`](02-communication-foundations.md)
- 平行：[`并行训练策略`](01-training-parallelism.md)
- 延伸：[`并行到通信映射`](05-parallelism-to-communication.md)
- 模型相关：[`MoE 最小导读`](../03-llm-architecture/06-moe-minimum.md)
- 系统落地：[`LLM Serving`](../02-inference-engine/04-llm-serving.md)

## 要点

- 大模型系统里“通信很慢”通常并不抽象，最后几乎都会落到几个具体 collective 上。
- 不同 collective 的瓶颈不一样：有的更怕带宽，有的更怕延迟，有的更怕负载不均和消息重排。
- 真正常用的能力不是死记定义，而是看到某种并行策略时，能立刻想到它会触发什么 collective。

## 通用知识

### 它是什么

collective 是多参与方同时参与的一类通信操作。常见的包括：

- all-reduce：各卡先归约再让所有参与方都拿到结果
- all-gather：各卡把各自分片收集成完整结果
- reduce-scatter：先归约，再把结果按分片散回各卡
- broadcast：从一方广播给所有方
- all-to-all：每个参与方都向其他参与方发送不同数据块

### 它解决什么问题

- all-reduce：常用于梯度同步
- all-gather：常用于收集参数分片或激活分片
- reduce-scatter：常用于和 all-gather 配对的高效梯度/状态切分
- all-to-all：常用于 MoE token dispatch / gather

### 为什么在 AI 系统里重要

因为并行策略的成本，经常等于“这一轮训练/推理需要触发多少次 collective、每次多大、是否能 overlap”。

### 它的收益与代价

- 合适的 collective 组合能减少总通信量或减少内存峰值
- 不合适的 collective 会把每层计算前后都插进同步点，GPU 算力再强也会被拖住

## 最小例子

以 4 卡 data parallel 为例：

- 每张卡算出一份梯度分片
- 通过 all-reduce，把 4 份梯度求和并同步到每卡

以 4 卡 tensor parallel 为例：

- 每张卡只算某层矩阵的一部分输出
- 某些阶段需要 all-gather 才能拼回完整激活

这就是为什么 DP 更像“阶段末同步一次大梯度”，而 TP 更像“层间更频繁同步激活或局部结果”。

## 对比表

| Collective | 核心作用 | 常见场景 | 典型风险 |
|---|---|---|---|
| all-reduce | 归约后所有参与方都拿到结果 | DDP 梯度同步 | 小消息过多、同步点密集 |
| all-gather | 收集所有分片拼成完整结果 | TP 激活或参数分片拼接 | 带宽压力大，峰值内存上升 |
| reduce-scatter | 先归约再把结果切分回各卡 | ZeRO / FSDP 等切分优化 | 次数多时容易被延迟主导 |
| all-to-all | 各卡彼此交换不同分片 | MoE token dispatch | 负载不均、长尾 rank 抖动 |

## 工程例子

如果一个训练任务表现为：

- GPU utilization 不低
- 但 step time 总是被大量小通信切碎

很可能问题不是某个单次 all-reduce 太慢，而是：

- bucket 太小
- collective 次数太多
- 通信与计算没有 overlap 起来

这种情况下，减少 collective 频次有时比优化单次 collective 更有效。

## 💥 实战踩坑记录（Troubleshooting）

> Watchdog caught collective operation timeout

- **现象**：并行训练并没有完全挂死，但 step time 周期性拉长，最终触发 NCCL 超时。
- **误判**：一开始容易怪某张 GPU 慢，或者怀疑单次 all-reduce 的实现有 bug。
- **根因**：更常见的情况是 collective 次数过多、小消息太碎，或者某个 rank 因为数据分布 / MoE dispatch 不均而形成长尾。
- **解决动作**：
	- 先看是哪一种 collective 在超时；
	- 再分清是单次太大，还是次数太多；
	- 然后检查 bucket、overlap 和负载均衡，而不是只盯带宽峰值。
- **复盘**：通信问题经常不是“总量大”，而是“同步点多、消息碎、尾巴长”。

## 推理优化工程师视角

从推理优化视角看，collective 的意义在于：它把“通信很慢”这种模糊说法，变成可定位的具体对象。

- 是 all-gather 太频繁
- 是 reduce-scatter 插得太密
- 是 all-to-all 在长尾 rank 上放大抖动

这会直接决定你后续是该改并行策略、调 batch、改 bucket，还是该回头审视拓扑。

一个很实用的工程习惯是：看到某个并行方案时，先在脑中把对应的 collective 顺序过一遍。只要能说清“哪一步会触发哪种 collective、频率大概多高”，你对性能风险的判断就已经比只背概念强很多了。

## 常见面试问题

### 初级

1. all-reduce 和 all-gather 的区别是什么？
2. reduce-scatter 常和什么场景一起出现？

### 中级

1. 为什么 TP 常比 DP 需要更多 all-gather / reduce-scatter？
2. 为什么 all-to-all 往往比 all-reduce 更难优化？

### 高级

1. 如果你看到 step 内有大量小 all-reduce，第一反应是什么？
2. 为什么“总通信量一样”不代表性能一样？

## 易错点

- 只背定义，不知道它们在训练图里的具体位置
- 只看通信总字节数，不看消息个数与同步点
- 把 all-to-all 当成普通 gather/scatter 的简单组合

## 排查 checklist

- [ ] 当前瓶颈 collective 是哪一种？
- [ ] 是单次太大，还是次数太多？
- [ ] 是否已经和计算 overlap？
- [ ] 是否存在小消息过多导致延迟主导？

## 参考资料

- NCCL collectives 文档
- MPI / distributed systems 基础资料
- Megatron / DeepSpeed 并行实现文档