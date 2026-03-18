# 通信技术索引

定位：回答 **多卡系统为什么慢、慢在什么通信模式、为什么通信会比计算更早成为上限**。

## 建议学习顺序

1. 通信基础：latency / bandwidth / topology
2. 硬件互联：PCIe / NVLink / NVSwitch / InfiniBand / RDMA
3. Collectives：all-reduce / all-gather / reduce-scatter / broadcast / all-to-all
4. 并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO
5. 训练并行实战：什么时候该切参数，什么时候该切状态
6. MoE / Expert Parallel / all-to-all
7. 通信与计算重叠：overlap / bucket / stream / pipeline
8. 通信问题排查：NCCL / 拓扑 / 带宽不达标 / 扩展性崩塌

## 当前章节

- [通信基础：latency / bandwidth / topology](02-communication-foundations.md)
- [硬件互联：PCIe / NVLink / NVSwitch / InfiniBand / RDMA](03-interconnects-and-topology.md)
- [Collectives：all-reduce / all-gather / reduce-scatter / all-to-all](04-collectives.md)
- [并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO](05-parallelism-to-communication.md)
- [训练并行策略：DP / TP / PP / FSDP / ZeRO](01-training-parallelism.md)

## 已完成“实战型重构”的核心页

- [通信基础：latency / bandwidth / topology](02-communication-foundations.md)
- [硬件互联：PCIe / NVLink / NVSwitch / InfiniBand / RDMA](03-interconnects-and-topology.md)
- [训练并行策略：DP / TP / PP / FSDP / ZeRO](01-training-parallelism.md)
- [Collectives：all-reduce / all-gather / reduce-scatter / all-to-all](04-collectives.md)
- [并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO](05-parallelism-to-communication.md)

## 推荐阅读顺序

1. 先读 `02-communication-foundations.md`，建立 latency / bandwidth / topology 的语言体系
2. 再读 `03-interconnects-and-topology.md`，理解硬件上限来自哪里
3. 然后读 `04-collectives.md`，把常见 collective 和通信代价对上号
4. 接着读 `05-parallelism-to-communication.md`，把通信映射回模型并行策略
5. 最后再看 `01-training-parallelism.md`，把定义和工程动作对应起来

## 本板块最应该掌握的 3 个判断

- 当扩展性差时，先问是 **latency 限制**、**bandwidth 限制**，还是 **拓扑/同步限制**。
- 当某个并行策略看起来“理论上省显存”，要继续问：它增加了哪些 collective，以及这些 collective 是否正好落在慢链路上。
- 当多卡吞吐没有线性增长时，不要先怀疑模型，先把通信体积、通信频次、rank 间负载均衡和 overlap 情况量化出来。
