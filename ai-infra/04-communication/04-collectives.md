# Collectives：all-reduce / all-gather / reduce-scatter / all-to-all

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

## 工程例子

如果一个训练任务表现为：

- GPU utilization 不低
- 但 step time 总是被大量小通信切碎

很可能问题不是某个单次 all-reduce 太慢，而是：

- bucket 太小
- collective 次数太多
- 通信与计算没有 overlap 起来

这种情况下，减少 collective 频次有时比优化单次 collective 更有效。

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