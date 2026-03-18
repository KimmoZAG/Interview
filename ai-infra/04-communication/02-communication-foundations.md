# 通信基础：latency / bandwidth / topology

## 一句话先讲清

通信问题最容易被说得很玄，但本质上先回答三个问题就够了：**这次慢，是因为启动通信太频繁、单条链路搬不动，还是数据根本走错了路。**

换成标准语言，就是：**latency、bandwidth、topology**。

## 关联知识网络

- 硬件路径：[`硬件互联：PCIe / NVLink / NVSwitch / InfiniBand / RDMA`](03-interconnects-and-topology.md)
- 集合通信：[`Collectives：all-reduce / all-gather / reduce-scatter / all-to-all`](04-collectives.md)
- 并行映射：[`并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO`](05-parallelism-to-communication.md)
- 训练切分：[`训练并行策略：DP / TP / PP / FSDP / ZeRO`](01-training-parallelism.md)
- 模型侧高通信场景：[`MoE 最小导读`](../03-llm-architecture/06-moe-minimum.md)

## 为什么值得先学这一页

- 多卡扩展效率下降，很多时候不是单点 bug，而是通信与同步链路的累积结果。
- 同样一个 collective，落在不同链路和拓扑上，表现可能完全两样。
- 如果一开始就把问题判断错，比如把 latency 问题当 bandwidth 问题，后面的优化动作基本都会偏航。

## 三个概念，分别在回答什么

| 概念 | 它描述什么 | 典型症状 |
|---|---|---|
| Latency | 一次通信启动到开始有效传输的固定开销 | 小消息很多时特别痛 |
| Bandwidth | 链路单位时间能搬多少数据 | 大消息、持续传输时更明显 |
| Topology | 设备如何连接、数据走哪条路 | 同卡数不同机器表现差很多 |

一句最实用的翻译：

- **latency** 看“每次发消息贵不贵”
- **bandwidth** 看“持续搬数据快不快”
- **topology** 看“这条数据到底怎么走、会不会绕路或拥塞”

## 为什么通信问题不能只看总字节数

考虑两种情况：

1. 传 `1KB`，但传很多次
2. 传 `100MB`，但传很少次

第一种往往更像 latency-bound，因为每次启动通信的固定成本占比高。

第二种更容易 bandwidth-bound，因为关键变成链路能否持续搬大块数据。

所以“总共传了多少数据”并不能单独决定快慢，**消息大小分布** 和 **同步频率** 同样重要。

## 把它放回 AI 系统里会发生什么

| 场景 | 更常见的通信压力 |
|---|---|
| DP 梯度同步 | 更偏 bandwidth 问题 |
| 小 bucket、频繁同步 | 更偏 latency 问题 |
| TP / MoE / 跨机训练 | topology 与延迟都更敏感 |
| FSDP 参数 gather / shard | 既可能带宽重，也可能因碎片化而 latency 重 |

这也是为什么讨论通信时不能只盯“总通信量”，还要看：

- 消息大小分布
- 通信频次
- 落在机内还是跨机
- 是否成功和计算 overlap

## 最小工程例子

假设你在做分布式训练，发现：

- 单步总通信量不算夸张
- 但 step time 里通信占比仍然很高

常见根因不是“网太慢”，而是：

- bucket 太碎
- 每层或每几个算子就同步一次
- 消息非常多，通信被 latency 切碎

这时最有效的优化往往不是换网络，而是：

- 合并 bucket
- 减少同步频次
- 提高 overlap 比例

## Troubleshooting：扩展效率差时，先怎么判方向

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 总通信量不大，但通信占比高 | latency 问题 | 看消息数量、bucket 粒度、collective 频次 |
| 单次 collective 很重 | bandwidth 问题 | 看消息大小与链路利用率 |
| 同样卡数，不同机器差很多 | topology 问题 | 看链路路径、机内/跨机比例、慢 rank 分布 |
| 平均值还行，但 p99 很差 | 局部拥塞或慢 rank | 看 rank 级分布而不是全局平均 |
| 理论通信时间不差，实测很慢 | overlap 不足或同步点过密 | 看 trace 中通信与计算重叠情况 |

### 一个实用排障顺序

1. 先判断是**少量大消息**还是**大量小消息**。
2. 再判断主要落在 latency、bandwidth，还是 topology。
3. 然后看慢 rank 是否固定出现在某些设备、节点或链路上。
4. 最后再去怀疑库、驱动或具体实现细节。

这个顺序非常土，但非常有效。很多“系统吞吐上不去”的问题，最后真不是算子慢，而是通信模式不对。

## 推理优化工程师视角

对推理优化工程师来说，这一页最大的价值是建立一个很硬的判断顺序：

- 不要一看到通信慢，就先怀疑库或驱动。
- 先判断是小消息太多，还是大消息太重。
- 再判断问题主要落在 latency、bandwidth，还是 topology。

很多推理系统问题，最终也会回到这三类：

- 调度把通信切碎了
- batching 方式导致同步频率太高
- 原本机内可接受的模式，一跨机就被 latency 放大

## 面试高频问法

### 初级

1. latency 和 bandwidth 的区别是什么？
2. 为什么通信问题不能只看总字节数？

### 中级

1. 什么情况下通信更像 latency-bound？什么情况下更像 bandwidth-bound？
2. 为什么 topology 会影响同一个 collective 的性能？

### 高级

1. 当 32 卡扩展效率下降时，你如何判断主要是 latency、bandwidth 还是 topology 问题？
2. 为什么某些 workload 在单机扩展正常，但一跨机就明显退化？

## 易错点

- 把“通信量大”误当成唯一问题，忽略消息数量和同步频率
- 只看平均带宽，不看局部拥塞和慢 rank
- 不分机内和跨机路径，导致排查对象错位

## 排查 checklist

- [ ] 当前慢的是少量大消息，还是大量小消息？
- [ ] 当前瓶颈更像 latency、bandwidth，还是拓扑绕路？
- [ ] 慢 rank 是否固定出现在某些设备或节点上？
- [ ] 通信是否成功和计算 overlap 了？
- [ ] 当前 collective 落在机内还是跨机路径上？

## 参考资料

- NCCL 文档
- NVIDIA 多 GPU 通信资料
- 分布式训练系统相关工程总结
