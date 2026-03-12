# 训练并行策略：DP / TP / PP / FSDP / ZeRO

## 要点

- 从 CS336 Lecture 7-8 的视角看，训练并行不是“把程序开多卡”这么简单，而是要决定：**参数、激活、梯度、优化器状态、计算图** 分别怎么切。
- 最常见的并行维度包括：Data Parallelism、Tensor Parallelism、Pipeline Parallelism、Sequence Parallelism，以及基于状态切分的 FSDP / ZeRO。
- 多卡训练的真正上限往往不是 GPU 算力，而是 **通信量、同步点、负载均衡、内存切分方式**。
- 讨论并行时，先明确你要解决的是哪类问题：
  - 单卡装不下
  - 单卡太慢
  - 通信太贵
  - pipeline 气泡太大

## 先区分两类“并行”

很多讨论把它们混在一起：

- kernel 并行：warp、block、tile，属于单设备执行模型
- 训练并行：多卡如何切模型与状态，属于分布式训练系统

这篇讲的是第二类。

## 1. Data Parallelism（DP）

最直观的做法：

- 每张卡放完整模型
- 不同卡处理不同 mini-batch 分片
- 反向后对梯度做 all-reduce / reduce-scatter 同步

优点：

- 实现最直观
- 对大多数模型结构侵入最小

缺点：

- 每张卡都要放完整参数
- 通信成本会随着参数规模增大而明显上升

适用场景：

- 模型能装进单卡
- 主要想扩 batch / 提高总吞吐

## 2. Tensor Parallelism（TP）

思路：

- 把单层里的大矩阵沿某个维度切到多张卡上
- 每张卡只算局部块，之后通过 all-gather / all-reduce 拼结果

优点：

- 可以把单层超大矩阵拆开，缓解单卡内存与算力压力

缺点：

- 通信更频繁，尤其在每层前后都可能需要同步
- 对模型结构与实现细节侵入更强

适用场景：

- 单层矩阵太大，单卡放不下或算不动
- 高速互联较好，通信可接受

## 3. Pipeline Parallelism（PP）

思路：

- 按层把模型切成多个 stage
- 每张卡或每组卡负责一段连续层
- 用 micro-batch 把流水线填满

优点：

- 按层切分，逻辑直观
- 对超深模型尤其有用

缺点：

- 会有 pipeline bubble
- stage 间负载不均时利用率差
- 调试和调度复杂度显著提升

适用场景：

- 层数多、整体模型太深
- 愿意接受更复杂的调度

## 4. Sequence Parallelism（SP）

思路：

- 在序列维度切分部分激活或计算
- 常和 TP 一起出现，用来缓解长上下文下激活与通信压力

它不是最先接触的并行方式，但在长上下文训练时非常有价值。

## 5. FSDP / ZeRO：切的是“状态”

这类方法的核心不是切算子，而是切训练状态：

- 参数
- 梯度
- 优化器状态

以 ZeRO 为例，可粗略理解为：

- Stage 1：切优化器状态
- Stage 2：再切梯度
- Stage 3：再切参数

FSDP 的核心直觉也类似：

- 让每张卡只在需要的时候临时 gather 某一层参数
- 用更复杂的通信换更小的常驻显存

优点：

- 对“装不下”问题特别有效

缺点：

- 参数 gather / sharding 带来额外通信和调度复杂度

## 该怎么选

一个够用的工程判断顺序：

1. 如果模型能装进单卡，但想提吞吐：先看 DP
2. 如果模型单层太大：看 TP
3. 如果模型整体太深：看 PP
4. 如果显存主要卡在参数/梯度/优化器状态：看 FSDP / ZeRO
5. 如果是长上下文激活压力：再考虑 SP / checkpointing 等组合

真实大模型训练里，通常不是单独使用一种，而是混合：

- DP + TP
- TP + PP
- FSDP + activation checkpointing
- DP + ZeRO + sequence parallelism

## 通信对象要分清

多卡训练里常见通信对象有：

- 梯度
- 参数分片
- 激活边界张量
- 优化器状态分片

常见 collective：

- all-reduce
- all-gather
- reduce-scatter
- broadcast

一旦对象不清，很多“为什么变慢”就解释不通。

## 你应该能回答的最小问题

- 为什么 DP 通常要求每卡保留完整模型副本
- 为什么 TP 更依赖高速互联
- 为什么 PP 会出现 bubble
- 为什么 FSDP / ZeRO 更像在改显存结构，而不是直接提高算力

## 最常见的三个瓶颈

### 1. 通信比计算更贵

表现：卡很多，但扩展效率很差。

### 2. Pipeline stage 不均衡

表现：有些卡忙，有些卡在等。

### 3. 参数切分省了显存，但 gather 太重

表现：能跑，但吞吐差得离谱。

## 易错点

- 把 DP、TP、PP 只当定义背，不去问“它们到底切了什么”
- 只关心单卡显存，不关心多卡通信曲线
- 认为卡数翻倍吞吐就应当接近翻倍
- 没有分清自己是在解决容量问题还是速度问题

## 排查 checklist

- [ ] 当前瓶颈是单卡装不下，还是多卡同步太慢？
- [ ] 你切分的是参数、激活、梯度，还是优化器状态？
- [ ] profiler / trace 能看出通信时间占比吗？
- [ ] stage 之间或 rank 之间是否明显负载不均？

## CS336 对照

- 官方 lecture 对应：Lecture 7-8（parallelism）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://rd.me/cs336
  - https://www.rajdeepmondal.com/blog/cs336-lecture-7
  - https://www.rajdeepmondal.com/blog/cs336-lecture-8