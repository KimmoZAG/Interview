# 04｜Mixture of Experts（MoE）

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec04/>

课程导航：上一讲 [03 架构与超参数](03-architectures-and-hyperparameters.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [05 GPU 基础](05-gpus.md)

工程桥接：[`AI Infra / MoE 最小导读`](../ai-infra/03-llm-architecture/06-moe-minimum.md) ｜ [`AI Infra / 训练并行策略`](../ai-infra/04-communication/01-training-parallelism.md) ｜ [`AI Infra / Collectives`](../ai-infra/04-communication/04-collectives.md)

## 先抓住这讲要点

- MoE 的本质：**让参数量增长得比 FLOPs 更快**。
- 它通常把 dense FFN 换成多个 expert，再用 router 只激活其中少数几个。
- MoE 的主要难点不在“公式”，而在**路由、负载均衡、通信与训练稳定性**。
- 真正让 MoE 难落地的，不是 `top-k` 三个字符，而是 all-to-all、dead expert、capacity overflow、fine-tuning fragility 这些工程问题。

## 为什么这页很关键

MoE 是课程里一个非常典型的“模型问题瞬间变系统问题”的例子：

- 从论文视角看，它像是在讨论稀疏激活；
- 从工程视角看，它在讨论路由、负载均衡、通信和训练动态。

这也是它特别适合作为模型岗和系统岗交界面试题的原因。

## 代表图

![lec04](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec04/00-24-58-1400.webp)

## 这一讲在回答什么

MoE 这讲主要在回答三个问题：

1. **为什么 MoE 能在固定 FLOPs 下扩大参数量？**
2. **为什么很多论文都说 MoE 好，但真正做起来却很难？**
3. **为什么 MoE 的难点主要在系统和训练动态，而不是前向公式本身？**

## 中文解读

### 1. 为什么 MoE 有吸引力

先看 dense Transformer 的 FFN。  
对于一个 token，普通 FFN 会完整经过整块 MLP，所以：

- 参数量增加，单 token 计算量也会跟着增加；
- 参数和 FLOPs 基本是强耦合的。

MoE 则把一层 FFN 拆成很多 expert，并让 router 只激活其中 top-k 个。于是：

- **总参数量**可以非常大；
- **单 token 激活的参数量**只占其中一小部分；
- **单步 FLOPs**不必和总参数同比例增长。

所以它特别适合一种目标：

> 想要更大模型容量，但又不想每个 token 都支付 dense 全参数计算成本。

### 2. 为什么说 MoE 是“参数扩容器”

你可以把 dense FFN 看成一家只有一个大厨房的餐厅；  
MoE 则像一条美食街：

- 总摊位很多；
- 每个顾客不会吃完整条街；
- 只会被分配到其中几个摊位。

这样做的收益是：

- 总容量变大；
- 每次服务成本相对可控。

但副作用也很明显：

- 分流逻辑变复杂；
- 不同摊位忙闲不均；
- 需要更复杂的物流系统。

这比喻虽然有点接地气，但很像真实 MoE 系统。别笑，工程师脑子里经常就是这种模型。

### 3. Router 为什么是 MoE 的灵魂，也是麻烦源头

对于每个 token，router 要决定它应该去哪些 expert。  
最常见的是 token-choice top-k routing：

1. 用 token hidden state 对所有 expert 打分；
2. 取 top-k；
3. 用这些 expert 的输出按 gate 权重加权求和。

表面上很简单，但问题马上出现：

- 如果 router 过于偏心，少数 expert 会爆满；
- 其他 expert 没有训练信号，慢慢“死掉”；
- 跨设备分发 token 时，通信会很重；
- 如果某个 expert 的容量超了，还得 drop token 或降级处理。

### 4. 为什么没有 balancing 就很容易崩

如果没有负载均衡机制，router 往往会把大部分 token 送到少数几个 expert。结果是：

- 其他 expert 梯度很少，变“死专家”；
- 通信负载不均；
- 容量虽然大，看上去很多参数，实际没用起来。

所以 balancing 的作用有两层：

1. **统计层面**：让更多 expert 真参与学习；
2. **系统层面**：别把几个 device 打爆，其它 device 在摸鱼。

### 5. 为什么 MoE 的难点主要是系统问题

一旦 experts 分散在多卡上，前向过程不再只是本地 MLP 计算，而变成：

1. 对 token 打分；
2. 把 token 路由到不同 expert 所在设备；
3. expert 分别计算；
4. 再把结果送回；
5. 重新按 token 顺序组装。

这背后几乎一定会涉及：

- all-to-all 通信；
- buffer 管理；
- token 排序/重排；
- capacity 限制；
- device-level load balancing。

所以很多人第一次看 MoE 会觉得：

> 公式不就这么短吗，怎么实现起来像搬家？

答案是：因为它真的像搬家，而且是每个 batch 都在搬。

### 6. 为什么有些简单路由也能有效

很有意思的一点是，MoE 的收益不完全来自“高语义智能路由”。  
有些工作发现：

- 简单 hash routing；
- 某些弱语义路由；
- 稍微带点结构性的分配；

也能比 dense baseline 更强。

这说明一件很重要的事：

> MoE 的一部分收益来自“容量分区”本身，而不完全来自 router 学到了多聪明的语义划分。

### 7. 为什么 MoE fine-tuning 更脆

MoE 看起来总参数很多，但激活是稀疏的。  
这会带来两个问题：

- 某些 expert 被少量任务数据过度改写，容易过拟合；
- 路由一旦在下游任务上分布漂移，训练就容易不稳定。

所以实际里经常会看到：

- alternating dense/sparse layers；
- 更多 fine-tuning data；
- 更保守的学习率；
- upcycling（从 dense 模型迁移初始化）。

## 代码拆解：top-k routing 最小示意

```python
import torch

def topk_route(x, router_w, k=2):
    scores = x @ router_w          # [tokens, experts]
    probs = scores.softmax(dim=-1)
    topk_val, topk_idx = probs.topk(k, dim=-1)
    topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)
    return topk_idx, topk_val
```

这段代码里最重要的不是 `topk` 本身，而是它背后的工程后果：

- 你要把 token 分发到不同设备上的 expert；
- 你要防止某些 expert 爆仓；
- 你还要把输出再 gather 回来。

公式很短，系统实现一点都不短。

### 这段代码背后的真实系统流程

在真实系统里，你通常还要补上一堆步骤：

1. 根据 `topk_idx` 对 token 分桶；
2. 按 expert 所在 device 重排；
3. 发起 all-to-all；
4. 每个 device 上 batched expert MLP；
5. 把输出按原 token 顺序 scatter 回去；
6. 再按 `topk_val` 做加权聚合。

所以 MoE 系统优化的关键不只是 router 精度，而是**路由后整条数据移动链路是否高效**。

## 代码拆解：balancing 的直觉

```python
def load_balance_penalty(actual_fraction, router_prob):
    return (actual_fraction * router_prob).sum()
```

这里的思想不是让所有 expert 永远一模一样忙，而是防止训练早期塌成“只有少数 expert 在工作”。

## 工程视角下的 MoE 代价

- all-to-all 通信
- capacity limit 与 token dropping
- 训练不稳定
- fine-tuning 易过拟合
- 推理时延更难控
- device 间负载不均
- 路由带来的非确定性与 batch 依赖

## 面试里怎么讲这一讲

如果面试官问：**“MoE 为什么能在相似 FLOPs 下提升能力？”**

你可以答：

> 因为 MoE 让总参数量和每 token 激活参数量解耦。dense 模型里参数越多，通常每个 token 的计算也越多；MoE 则只激活 top-k experts，所以总参数可以很大，但单 token 的有效计算只覆盖少数子网络，从而在相似 FLOPs 下获得更大模型容量。

如果被问：**“MoE 最难落地的地方是什么？”**

可以答：

> 主要不是前向公式，而是系统实现和训练稳定性。比如 top-k 路由后会有跨设备 token dispatch、all-to-all 通信、capacity overflow、dead experts、负载不均和 fine-tuning 脆弱性，所以 MoE 常常是用系统复杂度换参数效率。

## Troubleshooting：为什么 MoE 理论上更省，实际却更慢

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| FLOPs 漂亮，step time 一般 | all-to-all 吃掉收益 | 看通信时间占比和 rank 级分布 |
| 少数 rank 持续慢 | router 倾斜或 expert 过热 | 看 expert 热度和 rank 级 step time |
| 训练能收敛，但吞吐低 | token dispatch / gather 路径低效 | 看路由后数据重排和通信链路 |
| fine-tuning 特别脆 | expert 过拟合或路由分布漂移 | 对比 dense / MoE 下游稳定性 |

### 一个工程化判断顺序

1. 先分清比较的是总参数量还是每 token 激活参数量。
2. 再看收益到底该体现为能力提升、吞吐改善，还是单位 FLOPs 更高容量。
3. 然后检查 router、capacity、all-to-all 和 dead expert 问题。
4. 最后再决定这是模型收益不足，还是系统实现没把理论收益兑现出来。

## 本讲小结

MoE 的本质可以浓缩成一句话：

> 它不是“更便宜的 dense 模型”，而是“用稀疏激活换取更大容量”的另一条扩展路线。

它很强，但真正决定成败的，通常不是 router 数学是否优雅，而是：

- balancing 做得好不好；
- 通信链路够不够顺；
- 训练是否稳定；
- 推理时延是否可控。

## 推理优化 / AI Infra 视角

如果你更偏系统岗，这页最值得掌握的不是 top-k 公式，而是这条链：

`router -> token dispatch -> all-to-all -> expert 负载 -> 吞吐 / 尾延迟`

能把这条链讲清楚，MoE 题就不会只停在“稀疏激活更省 FLOPs”这一层。

## 复习题

1. 为什么说 MoE 是“用系统复杂度换参数容量”？
2. top-k routing 会带来哪些通信问题？
3. balancing loss 的作用是什么？
4. 为什么没有 balancing 时容易出现 dead experts？
5. 为什么说 MoE 的一部分收益来自容量分区，而不完全来自语义路由？

## 面试常见题目

1. MoE 为什么不是“免费变大模型”？
2. capacity factor 在工程上为什么重要？
3. all-to-all 为什么会成为 MoE 训练和推理里的关键代价？
4. shared expert 和 routed expert 分别在缓解什么问题？
5. 什么时候 MoE 可能不如 dense 模型划算？

## 面试题答题提示

### 1. 回答 MoE 时一定要讲通信

只讲“激活更稀疏”是不够的。MoE 最大的现实代价通常是路由、负载均衡和 all-to-all 通信。

### 2. 把收益分成参数容量和计算成本两层

它的吸引力在于参数量可以很大，但每个 token 的激活计算不必等比例增加。这样表述会更准确。

### 3. 别把 router 讲得太神秘

很多场景下，MoE 的收益不一定来自完美语义路由，而是来自把容量分到不同 expert 上后带来的更高总容量。
