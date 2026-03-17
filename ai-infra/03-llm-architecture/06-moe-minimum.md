# MoE 最小导读：路由、容量、负载均衡、系统代价

## 要点

- MoE 真正的难点不在“概念上把 MLP 换成多个 expert”，而在 **路由、负载均衡、跨设备通信、capacity 控制**。
- 一个现实判断：MoE 往往让“参数规模”与“每 token 计算量”开始解耦，但代价是系统复杂度显著上升。
- 面试里最容易答空的地方，不是定义，而是：**为什么 paper 上看起来更省，系统里却可能更慢**。

## 通用知识

### 它是什么

MoE（Mixture of Experts）可以理解为：

- 不再让所有 token 都走同一个 dense MLP
- 而是让 router 为每个 token 选择少数 expert 参与计算

### 它解决什么问题

它要解决的是：

- 如果继续堆 dense 模型，参数量和每 token 计算量会一起上涨

MoE 的思路是：

- 总参数量继续扩大
- 但单个 token 只激活少量专家

### 为什么在 AI 系统里重要

因为 MoE 看起来像“省计算”，但工程上常常把问题从 FLOPs 转移成：

- token routing
- expert 负载均衡
- all-to-all 通信
- capacity overflow

### 它的收益与代价

收益：

- 可以在相近激活计算量下扩大总模型容量

代价：

- 路由复杂
- 通信更重
- 负载均衡困难
- 实际吞吐不一定比 dense 更稳

### 它与 dense Transformer 的根本区别

dense block 的直觉是：

- 每个 token 都走同一套 MLP 参数

MoE block 的直觉是：

- 每个 token 先做一次“选路”
- 再只走少量 expert

所以 dense 更像：

- 计算路径稳定、结构规整、通信更容易预测

而 MoE 更像：

- 计算量被稀疏化了
- 但 token 去哪、每个 expert 收多少 token、这些 token 是否需要跨卡搬运，都变成了系统变量

## 为什么需要 MoE

在 dense Transformer 里，如果你想扩大模型容量，通常会同时增加：

- 参数量
- 每 token 计算量
- 显存与通信压力

MoE 的目标是：

- 让总参数量变大
- 但每个 token 只计算其中一小部分参数

一个常见判断是：

- 如果你的问题是“dense 模型容量不够，但每 token 计算已经很贵”，MoE 才开始有吸引力
- 如果你的问题本来就是通信、吞吐或部署复杂度已经吃紧，MoE 往往会把系统压力继续放大

## 最小结构图

把一个 MoE block 口述成 4 步，通常就够面试用了：

1. 当前 token 表示先进入 router
2. router 为它选择 top-k 个 experts
3. token 被 dispatch 到对应 experts 做前向
4. expert 输出再被 gather / combine 回主干

这里最值得强调的一句是：

- MoE 不只是“多个 MLP 并列摆着”，而是多了一层 **routing + dispatch + combine** 的调度系统

## 最小例子

假设一个 block 有 8 个 expert，top-k=2：

- 每个 token 只激活其中 2 个 expert
- 所以总参数量看起来很大
- 但单 token 实际参与计算的参数只是其中一部分

这就是 MoE 常说的“容量变大，但每 token 计算量不按总参数量同比增长”。

再补一个更工程化的说法：

- dense 模型如果把 MLP 宽度翻 8 倍，通常每个 token 的计算也会明显跟着涨
- MoE 如果换成 8 个 expert、top-k=2，则总容量可以上去很多，但单 token 只真正访问其中 2 个 expert

这就是 MoE 最核心的诱惑：

- **总参数更大**
- **单 token 激活计算不按总参数线性上涨**

## Top-k Routing / Capacity / 负载均衡

- top-k 决定每个 token 激活多少 expert
- capacity factor 给每个 expert 设置上限
- 负载均衡决定 expert 是否被均匀利用

把这三个概念翻成工程语言会更好记：

- top-k：每个 token 最终要吃多少份 expert 计算
- capacity：单个 expert 一次最多接多少 token
- load balance：这些 token 会不会一窝蜂挤到少数几个 expert

如果只记一句话，可以记：

- top-k 决定算多少，capacity 决定装多少，负载均衡决定是否跑得稳

## Capacity Overflow 到底意味着什么

如果某个 expert 太热门，就会出现：

- 它收到的 token 数超过当前 step 能处理的上限

这时系统通常只能做几种事：

- 丢弃一部分 token
- 回退到其他 expert
- 或者让这一步局部拥塞变得更严重

无论哪种，都会带来代价：

- 训练稳定性变差
- expert 利用率不均
- 吞吐下降

所以 capacity factor 从来不只是“一个超参”，而是稀疏模型是否可运行的重要护栏

## 为什么 MoE 是系统问题

MoE 会引入 dense 模型没有的额外成本：

- token dispatch / gather
- expert 间负载不均
- 跨设备 expert parallel 通信

再补一层非常重要的现实：

- 如果 expert 分散在不同 GPU 上，那么 token 往返 expert 的过程本质上就是一次高度动态的 all-to-all 类通信问题

这也是为什么 MoE 经常和下面这些词绑在一起出现：

- expert parallel
- all-to-all
- token permutation
- overlap

也正因如此，MoE 往往不是“把 dense MLP 换个模块”这么简单，而是把单机算子问题升级成了分布式调度问题

## Dense vs MoE 应该怎么比

比较两者时，至少要分清 4 件事：

1. 总参数量是不是同一个量级
2. 每 token 激活参数量是不是同一个量级
3. 实际吞吐和 step time 怎么样
4. 通信开销是否把理论 FLOPs 节省吃掉了

如果只说：

- “MoE 参数更多，所以更强”

这个比较通常是不成立的。

更合理的说法是：

- MoE 用更复杂的路由和通信，把一部分 dense 计算换成了更大的总容量
- 是否值得，取决于真实系统里 **算力、带宽、互联和负载均衡** 的共同结果

## 工程例子

一个典型现象是：

- 理论 FLOPs 看起来很漂亮
- 但真实训练吞吐不升反降

原因常常是：

- 热门 expert 过载
- all-to-all 通信吃掉理论收益
- token 分布不均导致部分设备更忙

所以 MoE 是否“更省”，不能只看 paper 上的稀疏激活描述，还要看实际通信和负载均衡表现。

再给一个更贴近排障的例子：

- 训练日志里总 loss 正常下降
- 但少数 rank 总是明显更慢

这往往提示：

- 不是模型本身算不动
- 而是某些 experts 或某些设备长期更热，导致 rank 间负载不均

这类问题如果只盯平均 GPU utilization，很容易看不出来。

## 推理优化工程师视角

即便你主要做推理优化，也应该至少会判断 3 件事：

1. 这个模型是否包含 MoE，是否会引入 token 路由和 expert dispatch
2. 它的瓶颈更可能落在算力还是通信
3. 如果未来要做多卡推理或专家并行，当前互联拓扑是否支撑得住

因为一旦模型是 MoE，很多原本对 dense 模型成立的直觉都会变弱：

- 总参数量不再等于每 token 计算量
- FLOPs 看起来好看，不等于吞吐一定高
- 平均负载正常，不代表所有 rank 都正常

## 常见面试问题

### 初级

1. 为什么 MoE 能在相近激活计算量下扩大模型容量？
2. top-k routing 和 capacity factor 分别做什么？

### 中级

1. 为什么负载均衡对 MoE 很关键？
2. 为什么 MoE 常常把问题从计算转到通信？

### 高级

1. 如果 MoE 理论上更省计算，但真实吞吐更差，你会先看哪些系统因素？
2. dense 模型和 MoE 模型应该怎么做公平比较？
3. 如果某些 rank 持续慢，你如何判断是 router 倾斜、expert 分布，还是互联拓扑问题？

## 易错点

- 只看参数量，不看 active parameters per token
- 忽略 overflow、capacity factor 和负载均衡
- 低估 expert parallel 的通信成本
- 把 paper 里的 FLOPs 节省直接当成线上吞吐收益

## 排查 checklist

- [ ] 当前问题是容量不够，还是系统吞吐已到瓶颈？
- [ ] 比较的是总参数量，还是每 token 激活计算量？
- [ ] router 是否存在明显负载不均？
- [ ] 通信开销是否吃掉 sparse 的理论收益？
- [ ] 是否观察了 rank 级别的 step time、all-to-all 时间和 expert 热度分布？

## 参考资料

- MoE / expert parallel 相关论文与工程实现
- Megatron / DeepSpeed MoE 文档
- all-to-all 与分布式通信优化资料
