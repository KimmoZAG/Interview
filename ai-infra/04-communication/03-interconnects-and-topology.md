# 硬件互联：PCIe / NVLink / NVSwitch / InfiniBand / RDMA

## 一句话先讲清

通信优化的第一原则不是“把代码写得更花”，而是先确认：**数据到底走哪条链路，这条链路的上限和抖动特征是什么。**

同样是 GPU 间通信，机内 NVLink / NVSwitch 和跨机 InfiniBand / RDMA，工程体验几乎不是同一种东西。

## 关联知识网络

- 基础语言：[`通信基础：latency / bandwidth / topology`](02-communication-foundations.md)
- 集合通信：[`Collectives：all-reduce / all-gather / reduce-scatter / all-to-all`](04-collectives.md)
- 并行切分：[`训练并行策略：DP / TP / PP / FSDP / ZeRO`](01-training-parallelism.md)
- 通信映射：[`并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO`](05-parallelism-to-communication.md)
- 高敏感场景：[`MoE 最小导读`](../03-llm-architecture/06-moe-minimum.md)

## 为什么值得单独学

- 很多训练/推理方案理论上只差一个 collective，但工程上差的是 collective 落在了哪种链路上。
- 拓扑决定的不只是峰值带宽，还决定了是否会出现热点链路、绕路、oversubscription 和尾延迟放大。
- “同样 8 张卡”为何差很多，往往不是卡的问题，而是卡之间到底怎么连。

## 常见链路，先建立最小直觉

| 链路 | 主要角色 | 工程直觉 |
|---|---|---|
| PCIe | 通用总线，GPU / CPU / NIC 都可能挂在上面 | 通用但不算豪横，容易受 root complex / NUMA 影响 |
| NVLink | GPU-GPU 高速点对点 | 更适合频繁 GPU 间交换 |
| NVSwitch | 多 GPU 通过交换芯片形成高带宽互联 | 更有利于多卡高频通信和更均匀路径 |
| InfiniBand | 跨机器高性能网络 | 多机训练常见主战场 |
| RDMA | 绕过更多 CPU 参与的远程访问能力 | 降低跨机数据搬运的额外开销 |

不要死背参数，先记一句：**链路类型决定你的 collective 是“勉强能跑”，还是“可以放心大胆地设计”。**

## 为什么拓扑会直接决定并行策略能不能成立

| 并行/场景 | 更依赖什么链路特性 |
|---|---|
| TP | 低延迟、高频 GPU-GPU 通信 |
| FSDP / ZeRO | 参数 gather / shard 的稳定带宽与重叠能力 |
| MoE / EP | all-to-all 与复杂 token 路由，对拓扑更敏感 |
| 跨机扩展 | NIC、交换网络、节点间路径一致性 |

所以问“8 卡 TP 能不能跑好”，答案从来不只取决于模型大小，还取决于这 8 张卡到底是怎么连的。

## 最小工程例子

假设有 8 张 GPU：

- 如果它们通过 NVSwitch 高速互联，做 TP 时每层的 all-gather / reduce-scatter 代价可能还能接受。
- 如果它们只是通过 PCIe，且跨多个 root complex，同样的 TP 可能很快就把每层通信变成瓶颈。

这就是为什么“卡数相同”不等于“通信能力相同”。

## 从单机扩到多机，为什么经常突然变脸

一个经典现象：

- 单机 8 卡吞吐不错
- 扩到 2 机 16 卡后，扩展效率突然明显掉下去

常见根因不是代码突然变差，而是：

- 原本大量通信都走机内 NVLink / NVSwitch
- 扩机后，一部分 collective 被迫跨机走 InfiniBand
- 如果 bucket 太小、同步太频繁，延迟开销就会被放大

所以“从单机扩到多机”为何经常翻脸，本质是热点通信路径发生了变化。

## Troubleshooting：同样模型、同样卡数，为什么表现差很多

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 同样 8 卡，吞吐差很多 | 拓扑不同、路径不对称 | 查 GPU 拓扑、root complex、NVLink/NVSwitch 可达性 |
| 扩机后效率骤降 | 机内通信变成跨机通信 | 看 collective 是否跨 NIC / 交换网络 |
| 少数 rank 长期更慢 | NUMA / NIC / root complex 绑定不合理 | 看 rank 到 GPU/NIC 映射 |
| 理论带宽很高，实测上不去 | 链路拥塞或绕路 | 对比实测带宽与理论上限、看是否热点集中 |

### 一个排障顺序

1. 先把 rank → GPU → NIC → 交换网络 的映射画清楚。
2. 再确认热点 collective 主要走机内还是跨机链路。
3. 然后看是否存在部分 rank 明显更慢，暗示拓扑不对称或绑定有问题。
4. 最后再判断是链路上限问题，还是软件没有把链路吃满。

## 推理优化工程师视角

推理优化里，很多“同样模型、同样卡数，表现却差很多”的现象，本质上都要回到这一章：

- 单机多卡和多机多卡不是同一种问题。
- 拓扑不只是背景信息，而是决定方案能否成立的前提条件。
- 你选择 TP、EP、paged KV 跨卡共享，还是干脆分片部署，都会被链路约束。

一个非常重要的工程习惯是：**在讨论优化方案之前，先把设备连接关系画出来。**

至少要明确：

1. 热点通信走机内还是跨机
2. 哪些 rank 之间的路径最贵
3. 是否存在 NIC / NUMA / root complex 绑定不合理

## 面试高频问法

### 初级

1. PCIe、NVLink、InfiniBand 分别大致解决什么问题？
2. 为什么多卡通信不能只看 GPU 数量，还要看拓扑？

### 中级

1. 为什么 TP 通常比 DP 更依赖高速 GPU-GPU 互联？
2. 为什么同样 8 卡，不同机箱拓扑下性能差距可能很大？

### 高级

1. 如果 16 卡扩展效率从 80% 掉到 45%，你会怎么判断是链路问题还是软件实现问题？
2. 为什么 MoE 的 all-to-all 往往比普通 all-reduce 更敏感于拓扑？

## 易错点

- 把“有 8 张卡”误当成“有相同的通信能力”
- 只看理论峰值带宽，不看真实路径和拥塞
- 忽略 CPU root complex、NUMA 和 NIC 绑定关系

## 排查 checklist

- [ ] rank 到 GPU、GPU 到 NIC、NIC 到交换网络的映射是否清楚？
- [ ] 当前热点 collective 主要走机内还是跨机链路？
- [ ] 是否存在部分 rank 明显更慢，暗示拓扑不对称？
- [ ] 实测带宽与链路理论上限差多少？
- [ ] 当前并行策略是否默认了机器并不具备的拓扑条件？

## 参考资料

- NVIDIA NVLink / NVSwitch 文档
- NVIDIA NCCL 文档
- InfiniBand / RDMA 官方资料