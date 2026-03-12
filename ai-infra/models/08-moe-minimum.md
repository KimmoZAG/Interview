# MoE 最小导读：路由、容量、负载均衡、系统代价

## 要点

- 从 CS336 Lecture 4 的视角看，Mixture of Experts 不是“白送参数量”，而是在固定计算预算下，用更复杂的系统机制换更大的模型容量。
- MoE 的核心思想是：每个 token 只激活少数 expert，而不是让所有 token 都走同一条 dense MLP 路径。
- MoE 真正的难点不在“概念上把 MLP 换成多个 expert”，而在 **路由、负载均衡、跨设备通信、capacity 控制**。
- 一个现实判断：MoE 往往让“参数规模”与“每 token 计算量”开始解耦，但代价是系统复杂度显著上升。

## 1. 为什么需要 MoE

在 dense Transformer 里，如果你想扩大模型容量，最直接的方法是：

- 加深层数
- 增大 hidden size
- 增大 MLP 宽度

但这些动作通常会同时增加：

- 参数量
- 每 token 计算量
- 显存与通信压力

MoE 的目标是：

- 让总参数量变大
- 但每个 token 只计算其中一小部分参数

所以它解决的不是“模型不够大”本身，而是“模型容量继续增大时，dense 计算太贵”的问题。

## 2. 最小结构图

把 dense block 里的 MLP 替换成 MoE 后，可以粗略理解成：

1. 一个 router 看当前 token 表示
2. router 为每个 token 选择 top-k experts
3. token 被发送到对应 experts 做前向
4. 把 expert 输出聚合回主干

所以一个 MoE block 至少包含两类对象：

- router
- experts

## 3. Top-k Routing 在做什么

最常见的路由方式是：

- 对每个 token 计算它应该更偏向哪些 expert
- 选 top-k 个 expert 处理它

如果 $k = 1$，每个 token 只进一个 expert；如果 $k = 2$，则每个 token 会在两个 expert 上计算并聚合。

直觉上：

- $k$ 越小，计算越省，但路由风险更集中
- $k$ 越大，计算更稳，但 sparse 的收益下降

## 4. Capacity Factor 是什么

MoE 最大的工程问题之一是：

- 某些 expert 可能特别受欢迎
- 某些 expert 可能几乎没人用

如果热门 expert 接收的 token 太多，就会出现 overflow。

capacity factor 的作用是：

- 给每个 expert 一个可接收 token 数上限
- 超过上限的 token 需要被丢弃、回退或重分配

这说明 MoE 不是单纯的“更智能路由”，而是受物理容量约束的调度问题。

## 5. 为什么必须做负载均衡

如果没有负载均衡，训练很容易出现：

- 某些 expert 过载
- 某些 expert 几乎不训练
- 总吞吐下降
- 模型容量实际利用率很差

因此 MoE 里常见的辅助目标之一，就是让 router 更平均地使用 experts。

一个足够实用的理解：

- 负载均衡项不是“锦上添花”，而是 sparse 模型能否训练稳定的关键组件

## 6. 为什么 MoE 是系统问题

MoE 会引入 dense 模型没有的额外成本：

- token dispatch：把 token 发到不同 experts
- token gather：把结果收回来
- expert 间负载不均
- 跨设备 expert parallel 通信

这意味着你节省了某些 FLOPs，但增加了：

- 通信
- 路由开销
- 调度复杂度

因此 MoE 的收益不能只看理论 FLOPs，还必须看真实系统吞吐。

## 7. Expert Parallelism 是什么

当 expert 数量很多时，常见做法是：

- 不把所有 expert 放在同一设备上
- 而是把不同 experts 分散到不同 GPU

这样做可以扩总容量，但也带来：

- token 在设备间搬运
- 通信拓扑更复杂
- 热门 expert 所在设备更容易成为瓶颈

## 8. Dense vs MoE 应该怎么比较

比较 MoE 和 dense 模型时，至少要同时看：

- 总参数量
- 每 token 激活参数量
- 实际吞吐
- 通信开销
- 最终验证 loss / 下游能力

如果只比较“总参数量更大”，结论通常没意义。

## 9. 你至少要会回答的三个问题

### 例 1：为什么 MoE 能在相近计算量下拥有更大容量

因为每个 token 只激活少量 expert，而不是计算全部参数。

### 例 2：为什么负载均衡很重要

因为如果 token 分配极不均匀，某些 expert 会过载，某些 expert 会闲置，系统吞吐和训练质量都会受损。

### 例 3：为什么 MoE 在系统上更难

因为它需要额外处理 token 路由、容量限制、dispatch/gather 和跨设备通信。

## 易错点

- 把 MoE 只理解成“更大但更省”
- 只看参数量，不看 active parameters per token
- 忽略 overflow、capacity factor 和负载均衡
- 在没有高速互联和成熟通信实现时，低估 expert parallel 成本

## 排查 checklist

- [ ] 当前问题是 dense 模型容量不够，还是系统吞吐已经到瓶颈？
- [ ] 你比较的是总参数量，还是每 token 激活计算量？
- [ ] router 是否存在明显负载不均？
- [ ] 通信开销是否吃掉了 sparse 带来的理论收益？

## CS336 对照

- 官方 lecture 对应：Lecture 4（Mixture of Experts）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-4
  - https://rd.me/cs336
  - https://github.com/YYZhang2025/Stanford-CS336