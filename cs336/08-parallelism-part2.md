# 08｜并行训练（二）：Collectives 与实现细节

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec08/>

课程导航：上一讲 [07 并行训练 1](07-parallelism.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [09 Scaling law 基础](09-scaling-laws-fundamentals.md)

工程桥接：[`AI Infra / Collectives`](../ai-infra/04-communication/04-collectives.md)｜[`AI Infra / 互联与拓扑`](../ai-infra/04-communication/03-interconnects-and-topology.md)｜[`AI Infra / 并行到通信映射`](../ai-infra/04-communication/05-parallelism-to-communication.md)

## 核心定义（What & Why）

> **一句话总结**：这一讲讲的是高层并行策略落地后的“底层交通规则”——collective 与通信实现细节，它解决的是为什么同样的并行方案，在不同拓扑和实现下性能会差得非常离谱。

## 关联知识网络

- 前置：[`07 并行训练（一）`](07-parallelism.md)
- 延伸：[`09 Scaling law 基础`](09-scaling-laws-fundamentals.md)
- 平行：[`AI Infra / Collectives`](../ai-infra/04-communication/04-collectives.md)
- 平行：[`AI Infra / 互联与拓扑`](../ai-infra/04-communication/03-interconnects-and-topology.md)
- 系统映射：[`AI Infra / 并行到通信映射`](../ai-infra/04-communication/05-parallelism-to-communication.md)
- 排障：[`AI Infra / 可观测性与调试`](../ai-infra/02-inference-engine/06-observability-and-debugging.md)

## 先抓住这讲要点

- 分布式训练里很多“高层并行策略”，最后都会落到少数几种 collective 上：`all-reduce`、`reduce-scatter`、`all-gather`、`broadcast`。
- 真正决定性能的不只是算法名字，还有**拓扑、消息大小、同步方式、实现细节**。
- 很多线上训练 hang、吞吐异常、扩卡不线性，根源都不在模型，而在 collective 行为和通信组织方式。

## 这一讲在整门课里的位置

第 7 讲解决的是“我要用哪种并行策略”；这一讲解决的是：

> 当我真的把训练拆到多卡上时，底层通信到底在发生什么？

如果不理解 collectives，你会很容易停留在概念层：

- 知道 FSDP 会 `all-gather`，但不知道为什么慢；
- 知道 DDP 要 `all-reduce`，但不知道为什么某些机型扩卡突然掉速；
- 知道 TP 依赖通信，但不知道究竟是哪个 collective 卡住了关键路径。

所以这讲是“并行策略”和“系统实现”之间的桥。

## 这讲想训练你什么能力

学完这一讲，你应该具备三种直觉：

1. **算子直觉**：知道不同 collective 的语义和用途；
2. **拓扑直觉**：知道同一种 collective 在不同硬件拓扑上表现可能完全不同；
3. **测量直觉**：知道 benchmark 和 profile 通信时，哪些结果能信，哪些结果是幻觉。

## 代表图

![lec08](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec08/00-19-06-1400.webp)

## 为什么说 collective 是分布式训练的“交通规则”

你可以把多 GPU 系统想成一个城市：

- GPU 是工厂；
- 显存是仓库；
- 计算是工厂内部加工；
- collective 则是**不同工厂之间运货和汇总的交通规则**。

如果交通组织得好，更多工厂就真的能提高总产出；  
如果交通组织得差，大家都堵在路上，再多 GPU 也只是更大的堵车现场。

所以分布式训练的核心从来不是“有没有通信”，而是：

> 什么时候通信、通信谁和谁、一次传多少、能不能和计算 overlap。

## 四个最关键的 collective

### 1. all-reduce：先归约，再让每个人都拿到结果

这是 DDP 最经典的动作。  
每个 rank 先有一份本地结果，例如本地梯度；然后把大家的结果做求和/平均，最后每个 rank 都得到同样的完整归约结果。

它的常见用途是：

- 梯度同步；
- 标量统计汇总；
- 某些 TP 中间结果规约。

### 2. reduce-scatter：先归约，但每个人只拿自己那一片

如果你不需要“人人都拿完整结果”，那 all-reduce 往往就有点浪费。  
这时更高效的做法是：

1. 先把所有 rank 的对应片段做归约；
2. 再把归约结果按片分给不同 rank。

这非常适合：

- ZeRO / FSDP 的梯度分片；
- 某些 TP 的输出分片；
- 通信后天然继续本地持有 shard 的场景。

### 3. all-gather：每个人贡献一片，再让所有人拼成完整结果

它可以理解成 reduce-scatter 的“反向心智模型”：

- 本地只有一部分；
- 但接下来计算需要完整结果；
- 那就把所有分片收集回来，拼成一个全量张量。

这在以下场景很常见：

- FSDP 聚合参数；
- TP 聚合分片激活；
- checkpoint 或状态恢复时重建完整对象。

### 4. broadcast：从一个源头发给所有人

它在概念上最简单：

- 某个 rank 有“权威版本”；
- 其他 rank 都需要一致副本；
- 那就广播。

常用于：

- 初始化参数；
- 发布配置；
- 某些恢复流程中的状态同步。

## 对比表：常见 collective 的语义与系统代价

| Collective | 语义 | 常见场景 | 更敏感的成本 |
|---|---|---|---|
| all-reduce | 归约后所有 rank 都拿完整结果 | DDP 梯度同步 | 总字节量、同步点、尾部 bucket |
| reduce-scatter | 归约后每个 rank 只拿一片 | ZeRO/FSDP 梯度分片 | 分片均衡、小消息延迟 |
| all-gather | 所有分片拼回完整结果 | FSDP 参数聚合、TP 激活拼接 | 带宽压力、峰值内存 |
| broadcast | 一份数据发给所有 rank | 初始化、配置同步 | 单源瓶颈、同步时机 |

## all-reduce 为什么是 DDP 的心脏

DDP 的世界观很明确：

- 每张卡有完整模型；
- 每张卡独立算本地样本；
- 最后只需要把梯度合并。

于是 all-reduce 成为 DDP 的核心原语。  
它厉害的地方在于：语义简单，适配性很强；  
它昂贵的地方在于：**模型越大，需要同步的梯度字节数越大**。

所以很多 DDP 优化，不是改变算法本质，而是在做这些事情：

- 梯度桶化；
- 分层触发同步；
- 尽量让后半段反向和前半段通信 overlap；
- 避免尾部大桶拖慢 step time。

## 为什么 reduce-scatter + all-gather 组合如此常见

从数学上看，`all-reduce ≈ reduce-scatter + all-gather`。  
但从系统实现上看，这两个分开的原语在很多场景更自然，因为它们更贴近“分片状态”的数据流。

举个直觉例子：

- 对于 ZeRO/FSDP，梯度最终就想留成 shard，那做完 `reduce-scatter` 后，不一定非要立刻变回完整；
- 对于 FSDP 参数，只有某层执行前才需要完整参数，所以只在那一刻 `all-gather` 就好。

也就是说：

> 分开使用 collectives，往往不是为了炫技巧，而是为了让通信模式更贴合数据的存储形态。

## 代码拆解：初始化分布式环境

```python
import torch.distributed as dist

def init(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()
```

这里 `barrier()` 的意义不是“仪式感”，而是：

- 确保所有 rank 都进入一致状态；
- 避免有人已经开始 collective，另一些人还没准备好；
- 减少最经典的那类分布式 hang：**参与者不齐，大家一起等到天荒地老**。

## 代码拆解：all-gather 的语义

```python
def conceptual_all_gather(local_tensor, world_tensors):
    # world_tensors: every rank contributes one shard
    return torch.cat(world_tensors, dim=0)
```

这段代码看起来像“简单拼接”，但在系统语义里它代表的是：

> 我本地只有一块拼图，接下来这个算子需要整张图，所以我要把所有拼图块都拿回来。

在 tensor parallel 里，这通常意味着：

- 各卡算出部分激活；
- 为了继续下一步，需要拼出完整激活；
- 所以进入 `all-gather`。

## 拓扑为什么会决定 collective 表现

很多人第一次做多机训练时会疑惑：

> 为什么同样的代码，在 A 机器上扩 8 卡几乎线性，在 B 机器上却像踩了刹车？

答案往往不在代码，而在拓扑。

### 节点内和节点外不是一个世界

- 节点内可能有 NVLink / NVSwitch；
- 节点外可能要走 NIC、交换机、IB / RoCE 网络；
- 两者带宽和延迟差异经常是数量级级别。

因此：

- 高频细粒度通信更适合留在节点内；
- 跨节点通信更怕 latency 和拥塞；
- 同一个 collective 算法在 ring / tree / hierarchical 组织方式下表现也可能不同。

### TP 为什么对拓扑特别敏感

因为 TP 的通信常常在层内关键路径上，每一层都可能来一次。  
这类通信不能太慢，也没太多地方躲。  
所以 TP 往往优先放在单节点内，而把更粗粒度的并行维度扩到多节点。

## benchmark collectives 时，最容易犯的错

通信 benchmark 是特别容易“测得很认真，结论很离谱”的领域。  
最常见的坑有：

### 1. 没有 warmup

第一次调用常常会包含：

- kernel/load 初始化；
- 通信器建立；
- 缓存未命中；
- lazy init。

如果把这些也算进 steady-state 性能，结果会失真。

### 2. timing 前后没同步

GPU 调用是异步的。  
如果你不在 timing 边界同步，测到的可能只是“把任务扔出去”的时间，而不是通信真实完成时间。

### 3. 各 rank 的 tensor shape 不一致

这不是“稍微有点误差”，而是很可能直接：

- 报错；
- hang；
- 或得到完全不可比较的结果。

### 4. 不统计真实字节量

不同 collective 的“有效通信量”不是一个公式。  
只看 wall-clock time 不够，还要结合传输字节量、拓扑和 world size 理解结果。

## 一个更靠谱的 benchmark 骨架

```python
import time
import torch
import torch.distributed as dist

def benchmark_collective(fn, tensor, warmup=10, iters=50):
    for _ in range(warmup):
        fn(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    return (t1 - t0) / iters
```

这个骨架虽然简化，但它至少体现了三件必要的事：

- warmup；
- timing 边界同步；
- 多 rank 共同进入同一测量窗口。

## 工程上你真正该看的不是“某个数字”，而是行为模式

一个成熟工程师做通信分析时，不会只问“这个 all-reduce 是 1.8ms 还是 2.1ms”。  
更重要的是看：

- world size 翻倍后是否接近线性变差；
- 小消息和大消息表现是否截然不同；
- 节点内外是否出现拐点；
- 是否能和计算 overlap；
- 是否某个 collective 在 profile 中形成长尾。

因为优化方向往往来自“模式”，不是来自单个孤立数字。

## 💥 实战踩坑记录（Troubleshooting）

> 现象：collective benchmark 看着很正常，但真实训练里某一轮经常卡住甚至 hang。

- **误判**：先怪网络设备、框架 bug，或者觉得“这波只是偶发抖动”。
- **根因**：更常见的是参与 collective 的 rank 没走到一致状态，或者某些 rank 的张量 shape / 顺序不一致，导致大家互相等到天荒地老。
- **解决动作**：
    - 先确认所有 rank 的调用顺序一致；
    - 再检查 shape、dtype、world size 配置是否完全一致；
    - 必要时在关键阶段加 barrier 和日志，把“谁先掉队”抓出来。
- **复盘**：分布式 hang 很少是玄学，通常只是“有人没按交通规则开车”。

> 常见异常：扩卡后 benchmark 不差，但真实训练吞吐却不线性。

- 这往往说明问题不在单次 collective，而在它和计算的 overlap、消息粒度、拓扑层级或尾部长消息行为。

## 面试里可以怎么讲

如果面试官问：**“为什么说 collective 是分布式训练的基础？”**

你可以答：

> 因为高层并行策略最终都要落到底层数据交换上。DDP 依赖 all-reduce 做梯度同步，FSDP/ZeRO 依赖 reduce-scatter 和 all-gather 处理分片状态，TP 依赖 all-gather 或 all-reduce 拼接或规约层内结果。理解 collective，才能理解并行训练真实的性能边界。

如果面试官问：**“benchmark collective 时最常见的失误是什么？”**

可以答：

> 最常见的是没有 warmup、计时边界不做同步、各 rank shape 不一致、只看时间不看有效通信字节和拓扑背景。这些问题会让 benchmark 看起来很精确，但结论完全不可靠。

## 复习题

1. `all-reduce`、`reduce-scatter`、`all-gather` 分别适合什么语义场景？
2. 为什么 `reduce-scatter + all-gather` 在分片训练里常比直接 `all-reduce` 更自然？
3. 节点内互联和节点外网络差异，会怎样影响并行策略设计？
4. 为什么 TP 比较怕高延迟网络？
5. 如果 collective benchmark 数字异常好看，你会先怀疑哪些测量错误？

## 面试常见题目

1. collective 选择为什么会直接影响扩展效率？
2. NCCL benchmark 结果应该和哪些背景信息一起解读？
3. pipeline parallel 的 bubble 为什么很难完全消掉？
4. 为什么真实系统常把 TP 限在节点内？
5. 网络拓扑为什么会反过来约束并行设计？

## 面试题答题提示

### 1. 回答通信问题时，一定带上拓扑

离开 NVLink、NVSwitch、InfiniBand、跨机延迟这些背景，很多并行结论都不成立。

### 2. benchmark 要讲测量边界

warmup、同步、消息大小、rank 数、实际字节量，这些都是 benchmark 是否可信的前提。

### 3. collective 不是抽象 API，而是系统成本

更成熟的回答方式是：某个 collective 改变了什么数据流，因此改变了通信频率、带宽需求和关键路径延迟。
