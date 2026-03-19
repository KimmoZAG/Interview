# 路由与负载均衡：为什么 MoE 的收益常常死在系统里

## 关键结论

MoE 真正难的地方，不是把专家数写得很大，而是让这些专家在训练和推理时**既能形成分工，又不会把通信、热点和延迟打爆**。

- DeepSeekMoE 很早就意识到，只有专家专门化还不够，还必须处理 expert-level 与 device-level 的负载均衡 [DeepSeekMoE, Section 3.3]。
- 到 DeepSeek-V2，路由已经开始接受系统约束，不再是“从所有专家里自由 top-k”这么简单，而是引入 `device-limited routing` [DeepSeek-V2, Sections 2.2.1-2.2.3]。
- 到 DeepSeek-V3，DeepSeek 进一步把主机制从“用 auxiliary loss 逼平衡”推进到“把平衡写进路由控制逻辑”，也就是 `auxiliary-loss-free load balancing` [DeepSeek-V3, Section 2.1.2]。

所以这一页最重要的结论是：**DeepSeek 的 MoE 之所以能继续做大，不只是因为专家设计对了，还因为路由越来越像系统设计，而不只是训练技巧。**

## 背景：为什么 MoE 最大的坑常常不是模型本身

### 旧做法为什么不够

MoE 最容易让人误会的地方，是大家常把它理解成一个“更省计算的 FFN 替代品”。但一旦模型做大，真正先爆炸的往往不是 FFN 本身，而是：

- token 会不会疯狂挤向少数热点专家；
- all-to-all 通信会不会成为主瓶颈；
- 某些 device 会不会长期过载，另一些 device 却在闲着；
- 为了追求平衡而加入的辅助损失，会不会反过来伤害模型质量。

也就是说，**MoE 的上限常常不是由专家数量决定，而是由路由和负载均衡的系统友好程度决定。**

### 这一页真正想解决什么

这一页主要想讲清楚四件事：

1. 为什么 DeepSeek 把路由问题一路从论文配角推成主角；
2. DeepSeekMoE、V2、V3 在路由策略上各自解决了什么瓶颈；
3. 为什么 auxiliary loss 到后面不再够用；
4. 什么叫“把平衡写进路由控制逻辑”。

## DeepSeek 具体怎么做

### 第一步：先承认 MoE 需要显式平衡机制

DeepSeekMoE 已经明确处理了两个层面的平衡问题 [DeepSeekMoE, Section 3.3]：

- `expert-level balance`：不要让少数专家长期被选中；
- `device-level balance`：不要让某些设备承受明显更多 token。

这一步听起来朴素，但非常关键，因为它意味着 DeepSeek 很早就接受了一个现实：**MoE 不会天然均匀，平衡必须被设计出来。**

### 第二步：V2 开始给路由加系统约束

到了 V2，DeepSeek 的问题已经不再只是“选哪些专家更准”，而是“选这些专家的代价能不能承受” [DeepSeek-V2, Sections 2.2.1-2.2.3]。

于是它引入了 `device-limited routing`。直觉上，这等于给路由器加了一条硬约束：

- 不是所有高分专家都能随便选；
- 一个 token 的目标 experts 最多分布在有限数量的设备上；
- 路由质量要和通信成本一起优化。

这种设计的意义很直接：**MoE 路由从“只看模型分数”变成了“分数 + 系统代价”的联合问题。**

### 第三步：V3 从“用 loss 逼平衡”转向“把平衡写进选择逻辑”

V2 仍然主要依赖辅助损失来维持平衡；但规模继续上去后，这种方法会越来越别扭：

- loss 太弱，热点压不住；
- loss 太强，又可能伤害主任务质量；
- 你是在优化一个统计趋势，而不是直接控制运行时行为。

因此 V3 进一步走到 `auxiliary-loss-free load balancing` [DeepSeek-V3, Section 2.1.2]。

它的直觉形式可以写成：

$$
g'_{i,t}=
\begin{cases}
 s_{i,t}, & s_{i,t}+b_i \in \operatorname{TopK}(\{s_{j,t}+b_j\}, K_r) \\
 0, & \text{otherwise}
\end{cases}
$$

这里：

- $s_{i,t}$ 是 token $t$ 对 expert $i$ 的原始 affinity score；
- $b_i$ 是动态调整的 expert bias；
- 是否入选由 $s_{i,t}+b_i$ 决定；
- 真正参与加权的值仍保留原始分数 $s_{i,t}$。

这套机制厉害的地方在于：**你可以调节路由分布，却不必强行把主优化目标扭向“人人平均分工”。**

### 第四步：把路由做成更适合大规模系统的样子

V3 的路由设计不是单独存在的，它和 node-limited routing、no token-dropping、跨节点 all-to-all 优化、冗余专家部署是一起工作的 [DeepSeek-V3, Sections 2.1.2, 3.2, 3.4]。

也就是说，DeepSeek 后期的思路已经很清楚：

- 路由不是训练时的小技巧；
- 它直接决定通信扇出、热点分布和线上延迟；
- 设计得好，MoE 才能把“总参数大、激活计算小”的优势兑现成真实系统收益。

### 这套设计带来的直接优点

把这条路由主线压缩一下，收益主要是：

- **热点更少**：不容易出现少数专家被挤爆；
- **通信更可控**：token 不会被随意撒到过多设备上；
- **模型质量更稳**：不必过度依赖会伤主任务的强辅助损失；
- **更适合大规模训练和部署**：MoE 的理论收益更有机会落到真实系统里。

## 数据怎么说明这些优点

### 证据一：DeepSeekMoE 已经把 balance 问题写进核心设计

DeepSeekMoE 并没有把负载均衡藏到附录，而是明确讨论了 expert-level 与 device-level balance [DeepSeekMoE, Section 3.3]。

这说明 DeepSeek 很早就认识到：**专家是否专门化，和专家是否能被稳定调度，是同一个问题的两面。**

### 证据二：V2 把通信约束直接并入路由设计

V2 的 `device-limited routing` 与 communication balance loss 表明，DeepSeek 已经不满足于“路由平均一点”，而是开始给通信开销设显式上界 [DeepSeek-V2, Sections 2.2.2-2.2.3]。

这一步很关键，因为它意味着 MoE 的系统成本第一次被主设计正面处理，而不是留给工程同学赛后抢救。

### 证据三：V3 之所以能把 671B / 37B 激活模型训起来，路由控制是前提之一

V3 的主线是超大规模系统协同：aux-loss-free balancing、DualPipe、cross-node all-to-all kernels、FP8、冗余专家部署共同支撑了训练与服务闭环 [DeepSeek-V3, Sections 2.1.2, 3.2-3.4]。

虽然这不是单靠路由一项完成的，但恰恰说明了一件事：**没有系统友好的路由，后面的那些工程优化很难全部接得上。**

## 思考问题

- 你觉得 MoE 更容易先死在哪一步：专家学不专，还是专家分得太散导致通信爆炸？为什么？
- `auxiliary-loss-free` 的核心价值，更像“保模型质量”，还是“保系统可控性”？
- 如果你只能在路由里保留一个约束，你会优先保 `expert balance`、`device balance`，还是 `node-limited routing`？
