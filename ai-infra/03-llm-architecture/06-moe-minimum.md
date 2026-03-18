# MoE 最小导读：路由、容量、负载均衡、系统代价

## 一句话先讲清

MoE 真正的难点，不在于“把一个 dense MLP 换成多个 expert”，而在于：**你把一部分 dense 计算换成了路由、容量控制、负载均衡和跨设备通信问题。**

所以它最容易面试答空的点，不是定义，而是：**为什么 paper 上看起来更省，系统里却可能更慢。**

## 关联知识网络

- 基础结构：[`Transformer 推理所需的最小知识`](01-transformer-minimum.md)
- 通信映射：[`并行策略与通信映射：DP / TP / PP / SP / FSDP / ZeRO`](../04-communication/05-parallelism-to-communication.md)
- 训练并行：[`训练并行策略：DP / TP / PP / FSDP / ZeRO`](../04-communication/01-training-parallelism.md)
- Collectives：[`Collectives：all-reduce / all-gather / reduce-scatter / all-to-all`](../04-communication/04-collectives.md)
- 系统评估：[`评测与基准：accuracy/latency/throughput`](05-evaluation-and-benchmarking.md)

## 为什么值得单独学

- MoE 让“总参数量”和“每 token 激活计算量”开始解耦，这很诱人，但系统代价会同步上升。
- 一旦涉及 expert parallel，通信模式会明显变化，尤其是 all-to-all 与 token permutation。
- 你不只是在学一个模型结构，而是在学一个**稀疏计算 + 分布式调度**问题。

## MoE 到底解决什么问题

在 dense Transformer 里，如果你想扩大模型容量，通常会同时增加：

- 参数量
- 每 token 计算量
- 显存与通信压力

MoE 的思路是：

- 让总参数量继续扩大
- 但每个 token 只激活少量 expert

这就是它最核心的吸引力：**总容量可以变大，但单 token 的激活计算不必按总参数线性上涨。**

## 最小结构：把 MoE block 口述成 4 步

1. token 表示先进入 router
2. router 为每个 token 选择 top-k 个 experts
3. token 被 dispatch 到对应 experts 做前向
4. expert 输出再 gather / combine 回主干

最值得强调的一句是：**MoE 不只是“多个 MLP 并列摆着”，而是多了一层 routing + dispatch + combine 的调度系统。**

## Dense vs MoE：根本区别在哪

| 维度 | Dense Transformer | MoE |
|---|---|---|
| MLP 路径 | 每个 token 走同一套参数 | 每个 token 只走少量 experts |
| 计算路径 | 更稳定、更规整 | 更稀疏、更动态 |
| 通信模式 | 更容易预测 | 更依赖 dispatch / gather / all-to-all |
| 系统风险 | 主要是计算与显存 | 负载均衡、overflow、通信更突出 |

这也是为什么 dense 更像“规整的大矩阵问题”，而 MoE 更像“带路由的分布式系统问题”。

## Top-k / Capacity / 负载均衡，怎么一口气讲清楚

| 概念 | 它决定什么 | 工程翻译 |
|---|---|---|
| Top-k | 每个 token 激活多少 expert | 每个 token 最终要吃多少份 expert 计算 |
| Capacity factor | 单个 expert 的上限 | 一个 expert 一次最多能装多少 token |
| Load balance | token 是否均匀分散 | 会不会所有 token 一窝蜂挤到少数 expert |

如果只记一句话：**top-k 决定算多少，capacity 决定装多少，负载均衡决定是否跑得稳。**

## Capacity overflow 到底意味着什么

如果某个 expert 太热门，就会出现：

- 收到的 token 数超过当前 step 的处理上限

这时系统通常只能：

- 丢弃一部分 token
- 回退到其他 expert
- 或者让这一轮局部拥塞更严重

不管哪种，都会带来代价：

- 训练稳定性变差
- expert 利用率不均
- 吞吐下降

所以 capacity factor 不是“一个普通超参”，而是 MoE 能否稳定运行的重要护栏。

## 为什么 MoE 很容易从算力问题变成通信问题

MoE 会引入 dense 模型没有的额外成本：

- token dispatch / gather
- expert 间负载不均
- 跨设备 expert parallel 通信

如果 experts 分散在不同 GPU 上，那么 token 往返 experts 的过程，本质上就是一次高度动态的 all-to-all 类通信问题。

也正因如此，MoE 经常和下面这些词绑在一起：

- expert parallel
- all-to-all
- token permutation
- overlap

一句话总结：**MoE 常常把单机算子问题升级成了分布式调度问题。**

## 最小工程例子

假设一个 block 有 8 个 experts，`top-k = 2`：

- 每个 token 只激活其中 2 个 experts
- 总参数量看起来很大
- 但单 token 真正参与计算的参数只是其中一部分

把它和 dense 对比会更好记：

- dense 如果把 MLP 宽度扩大 8 倍，单 token 计算通常也会明显上升
- MoE 如果换成 8 个 experts、`top-k = 2`，总容量可以上去很多，但单 token 只真正访问其中 2 个 experts

这就是 MoE 的诱惑与代价同时所在：容量上去了，系统复杂度也一起上去了。

## Troubleshooting：为什么理论 FLOPs 很漂亮，真实吞吐却不升反降

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 理论上更省计算，step time 却更差 | all-to-all 通信吃掉收益 | 看通信时间占比与 overlap 情况 |
| 少数 rank 长期更慢 | router 倾斜或 expert 热点不均 | 统计 rank 级 step time 与 expert 热度 |
| GPU 平均利用率看着还行 | 平均值掩盖了局部拥塞 | 看 rank 级别而不是全局平均 |
| overflow 频繁 | capacity factor 太紧或负载均衡差 | 统计 dropped tokens / overflow 比例 |

### 一个实用排障顺序

1. 先看比较的是**总参数量**还是**每 token 激活计算量**。
2. 再看 router 是否存在明显倾斜，expert 是否长期冷热不均。
3. 然后确认 all-to-all 是否吃掉了理论 FLOPs 节省。
4. 最后再结合互联拓扑判断：问题是模型路由，还是链路和并行布局本身不合适。

## 推理优化工程师视角

即便你主要做推理优化，也至少应该会判断 3 件事：

1. 模型是否包含 MoE，是否会引入 token 路由和 expert dispatch。
2. 它的瓶颈更可能落在算力还是通信。
3. 如果未来要做多卡推理或专家并行，当前互联拓扑是否扛得住。

因为一旦模型是 MoE，很多对 dense 模型成立的直觉都会变弱：

- 总参数量不再等于每 token 计算量
- FLOPs 好看，不等于吞吐一定高
- 平均负载正常，不代表所有 rank 都正常

## 面试高频问法

### 初级

1. 为什么 MoE 能在相近激活计算量下扩大模型容量？
2. top-k routing 和 capacity factor 分别做什么？

### 中级

1. 为什么负载均衡对 MoE 很关键？
2. 为什么 MoE 经常把问题从计算转到通信？

### 高级

1. 如果 MoE 理论上更省计算，但真实吞吐更差，你会先看哪些系统因素？
2. Dense 模型和 MoE 模型应该怎么做更公平的比较？
3. 如果某些 rank 持续偏慢，你如何区分是 router 倾斜、expert 布局还是互联拓扑问题？

## 易错点

- 只看总参数量，不看 active parameters per token
- 忽略 overflow、capacity factor 和负载均衡
- 低估 expert parallel 的通信成本
- 把 paper 里的 FLOPs 节省直接当成真实吞吐收益

## 排查 checklist

- [ ] 当前问题是容量不够，还是系统吞吐已到瓶颈？
- [ ] 比较的是总参数量，还是每 token 激活计算量？
- [ ] router 是否存在明显负载不均？
- [ ] 通信开销是否吃掉 sparse 的理论收益？
- [ ] 是否观察了 rank 级 step time、all-to-all 时间和 expert 热度分布？

## 参考资料

- MoE / expert parallel 相关论文与工程实现
- Megatron / DeepSpeed MoE 文档
- all-to-all 与分布式通信优化资料
