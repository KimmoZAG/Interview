# DeepSeek 总览：从稀疏架构到推理强化

## 关键结论

DeepSeek 这条线最容易被误读成“先做了 MoE，再做了 MLA，再做了 RL”。更准确的理解是：**它一直在围绕同一个问题演进——怎样在有限预算下，把参数效率、推理效率、系统可扩展性和 reasoning 能力一起往上推。**

- `DeepSeekMoE` 先解决“参数变多以后，是否真的变成更高有效容量”；
- `DeepSeek-V2` 再解决“即使 MoE 可训，KV cache 和通信成本能不能继续压下去”；
- `DeepSeek-V3` 继续解决“超大 MoE 的训练、通信、部署能不能一起成立”；
- `DeepSeek-R1` 最后把节省出来的预算继续投向 reasoning 行为本身，而不只是继续堆预训练 [DeepSeekMoE, Section 9; DeepSeek-V2, Section 5; DeepSeek-V3, Section 6; DeepSeek-R1, Section 6]。

所以这套文档的主线不是“论文串读”，而是：**架构怎么省、系统怎么撑、训练怎么放大 reasoning。**

## 背景：为什么 DeepSeek 值得单独拉成一条主线

### 旧的理解为什么不够

如果只看 benchmark 或模型规模，很容易把 DeepSeek 理解成一条普通的“大模型继续做大”路线；但它真正特别的地方在于，比较早就把这些问题放到同一张图里考虑：

- 稀疏架构要不要做；
- KV cache 和推理吞吐怎么降本；
- all-to-all、负载均衡、FP8、并行调度怎么一起协同；
- 后训练到底是“礼貌微调”，还是可以直接变成 reasoning 能力放大器。

也就是说，DeepSeek 的每一代不只是“多了一个新技巧”，而是在回答不同层面的瓶颈。

### 这一页真正想解决什么

这一页不打算复述论文细节，而是先帮你搞清楚三件事：

1. DeepSeek 这条路线到底在优化什么；
2. 这套系列文档应该按什么顺序读才不容易迷路；
3. 如果你只关心某一条线，应该从哪篇开始。

## DeepSeek 具体怎么做

### 第一条线：先把架构做成“更省算、更能扩”的形态

DeepSeek 在架构层最核心的两步是：

- 用 `DeepSeekMoE` 改造传统 MoE，让专家更专门化，减少“参数很多但知识混杂”的情况；
- 用 `MLA` 改写 attention 的 KV 状态表示，让推理成本不再被传统多头 KV cache 牵着走 [DeepSeekMoE, Sections 1, 3; DeepSeek-V2, Section 2.1]。

这条线回答的是：**如果单位计算不够值钱，那后面加再多系统优化也很难救。**

### 第二条线：再把系统做成“真的能把这些收益兑现出来”

只有好架构还不够，因为 MoE 一旦放大，通信、路由、负载均衡和并行调度立刻会成为主问题。

所以从 V2 到 V3，DeepSeek 持续把系统问题抬到主线：

- `device/node-limited routing`
- `auxiliary-loss-free load balancing`
- `DualPipe`
- `cross-node all-to-all kernels`
- `FP8`
- `prefilling / decoding 分离部署` [DeepSeek-V2, Sections 2.2-3.1; DeepSeek-V3, Sections 2-3]

这条线回答的是：**论文里的结构收益，怎样才能真的落到训练成本和在线吞吐上。**

### 第三条线：最后把节省出来的预算投向 reasoning

到了 `DeepSeek-R1`，重点已经不是继续改 base 架构，而是把后训练本身变成 reasoning 放大器：

- 先用 `R1-Zero` 验证 pure RL 是否能诱导出更强推理行为；
- 再用 cold-start SFT、拒绝采样和第二阶段 RL，把 raw reasoning 收束成更可读、更可用的模型 [DeepSeek-R1, Sections 1-3]。

这条线回答的是：**当 base model 已经很强时，额外算力是不是更应该投到 reasoning 行为，而不是只投到更长预训练。**

### 这一整套路线带来的直接优点

把 DeepSeek 主线压缩成结果，大概就是三件事：

1. **单位计算更值钱**：稀疏架构和 MLA 让参数效率、KV 效率更高；
2. **系统收益更可兑现**：V3 把通信、调度、精度和部署真正拉成一个闭环；
3. **后训练不再只是修口风**：R1 证明 RL 可以直接成为 reasoning 的主要增长引擎。

## 数据怎么说明这些优点

### 证据一：每一代都在解决新的主瓶颈，而不是重复加料

从论文定位就能看出这条线的连续性：

- `DeepSeekMoE` 关心 expert specialization；
- `DeepSeek-V2` 关心 MLA 与路由约束；
- `DeepSeek-V3` 关心超大规模系统协同；
- `DeepSeek-R1` 关心 reasoning 行为能否被显式诱导 [DeepSeekMoE, Section 9; DeepSeek-V2, Section 5; DeepSeek-V3, Section 6; DeepSeek-R1, Section 6]。

这说明 DeepSeek 不是在堆平行技巧，而是在持续解决“当前最贵的问题”。

### 证据二：V2 和 V3 明确把系统收益写成主结果

V2 不只是提出 MLA，还明确给出 KV cache 降幅和吞吐提升；V3 也不只是给出模型规模，而是把 load balance、通信、FP8、训练成本与部署路径一起写进主文 [DeepSeek-V2, Abstract; Section 3.2.3; DeepSeek-V3, Abstract; Sections 3.2-3.4]。

这说明 DeepSeek 从中期开始，就不再把系统优化当作“附录工程细节”，而是当作模型能力的一部分。

### 证据三：R1 证明后训练可以成为主能力来源，而不是收尾动作

R1 最重要的证据不只是最终 benchmark，而是它明确展示：

- pure RL 可以诱导更长、更强的 reasoning；
- reasoning 轨迹还可以再被收束、蒸馏、迁移到更小模型 [DeepSeek-R1, Sections 2-4]。

这意味着 DeepSeek 路线里，后训练已经不只是“让回答更顺眼”，而是“继续生产能力”。

## 阅读导航

<div class="grid cards" markdown>

- :material-family-tree: **架构专题**

    ---

    如果你最关心“DeepSeek 的模型到底改了什么”，先看专家怎么切细、MLA 为什么能压缩 KV、为什么路由和长上下文会一起影响系统。

    [:octicons-arrow-right-24: 从架构起步](architecture/deepseek_moe.md)

- :material-brain: **训练与对齐**

    ---

    如果你最关心“reasoning 到底怎么被训出来”，这里会串起预训练、GRPO、reward/verifier、RL 基础设施、distillation 和 failure modes。

    [:octicons-arrow-right-24: 从训练主线开始](training/rl_and_alignment.md)

- :material-memory: **工程系统**

    ---

    如果你更关心 FP8、DualPipe、all-to-all、显存和带宽约束，直接从系统页切入最有效。

    [:octicons-arrow-right-24: 进入系统优化专题](engineering/infra_optimization.md)

- :material-chart-timeline-variant: **代际对比**

    ---

    如果你只想先抓主线，不想一上来钻细节，这一页最适合快速建立“MoE → MLA → V3 系统协同 → R1 推理强化”的时间轴。

    [:octicons-arrow-right-24: 进入代际演进页](comparison/v1_to_v3_evolution.md)

</div>

## 推荐阅读路径

### 如果你想先建地图

先看 `comparison/v1_to_v3_evolution.md`，再回来读本页导航，会更容易把每一篇放到正确位置。

### 如果你想先看模型怎么变强

按这个顺序读：

- `architecture/latest_deepseek_v32_architecture.md`
- `architecture/deepseek_moe.md`
- `architecture/mla_attention.md`
- `architecture/routing_and_load_balancing.md`
- `architecture/long_context_and_yarn.md`

如果你想先用一页抓住“最新版 DeepSeek 现在长什么样”，优先读 `architecture/latest_deepseek_v32_architecture.md`；如果你想把每个部件拆开学，再按后面的专题页往下读。

### 如果你想先看 reasoning 怎么训出来

按这个顺序读：

- `training/pretraining_strategies.md`
- `training/rl_and_alignment.md`
- `training/reward_design_and_verifiers.md`
- `training/rl_infrastructure.md`
- `training/distillation_and_transfer.md`
- `training/failure_modes_and_limitations.md`

### 如果你想先看系统怎么把收益兑现

直接读 `engineering/infra_optimization.md`，然后回看训练和架构页，你会更容易理解为什么 DeepSeek 一直把“模型设计”和“系统设计”绑在一起讲。

## 思考问题

- 如果只能保留 DeepSeek 路线中的一个环节，你会选稀疏架构、MLA、V3 系统协同，还是 R1 的 RL 路线？为什么？
- DeepSeek 更像“先把预算省出来，再重新分配”，而不是“继续无脑堆大”。这种策略在哪类团队里最有效？
- 如果你要继续扩写这一系列，下一篇最该补的是结构化输出/工具使用，还是训练—部署闭环复盘？
