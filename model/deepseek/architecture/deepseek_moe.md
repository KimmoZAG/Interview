# DeepSeekMoE：为什么要把专家切得更细

## 关键结论

DeepSeekMoE 最重要的贡献，不是简单地把 dense FFN 换成了 MoE，而是回答了一个更尖锐的问题：**MoE 的参数很多，为什么它们常常没有真的变成更高的有效容量。**

- 它指出传统 coarse-grained MoE 容易让 routed experts 学到彼此重叠的知识，参数看起来很多，但分工不够清楚 [DeepSeekMoE, Sections 1, 3.1]。
- 它通过 `finer-grained experts + shared experts`，把“应该专门化的知识”和“所有 token 都会用到的共通知识”拆开处理 [DeepSeekMoE, Sections 3.1-3.2]。
- 它的目标不是把每个 token 送进更多专家，而是让**被激活的那一小部分计算更值钱** [DeepSeekMoE, Sections 3-4]。

所以这一页的核心结论是：**DeepSeek 走 MoE，不只是为了省算，而是为了让参数真正形成可分工的容量。**

## 背景：为什么传统 MoE 还不够好

### 旧做法为什么不够

MoE 的经典卖点是“总参数很大，但每个 token 只激活一小部分计算”。问题在于，真正把它训起来后，往往会出现三个现实麻烦：

- 不同专家学到的东西并没有想象中那么不同；
- 很多共通知识被 routed experts 重复学习，浪费容量；
- 如果只看总参数，很容易误以为容量线性上涨，但有效容量并没有同步上涨。

换句话说，传统 MoE 的问题不在于它不能省算，而在于它常常**省了计算，却没有把省下来的预算充分转成更清晰的功能分工**。

### 这一页真正想解决什么

这一页主要想讲清楚四件事：

1. DeepSeek 为什么认为传统 MoE 的专家粒度还不够细；
2. `shared experts` 解决的到底是什么问题；
3. 为什么这套设计比“单纯增大 top-k MoE”更像有效容量升级；
4. 这些结构收益后来怎样接到 V2/V3 的系统与推理主线上。

## DeepSeekMoE 具体怎么做

### 第一步：把 routed experts 切得更细

DeepSeekMoE 的第一步不是把专家做得更大，而是把它们切得更细，让每个 expert 更容易形成专门分工 [DeepSeekMoE, Sections 3.1, 4.1]。

直觉上可以把它理解成：

- 旧式 MoE 更像“几个大部门”；
- DeepSeekMoE 更像“更多、更小、更专门的小组”。

这样做的好处是，路由器更容易把不同 token 分到真正不同的功能区，而不是让几个大专家都学成“什么都懂一点”。

### 第二步：把共通知识交给 shared experts

如果所有能力都交给 routed experts，会出现一个问题：很多基础能力其实几乎每个 token 都需要，于是不同 routed experts 会重复学习这些公共模式。

DeepSeekMoE 因此额外引入 `shared experts`，专门承担更通用、更稳定的公共知识通道 [DeepSeekMoE, Section 3.2]。

可以把单层输出写成一个直观形式：

$$
\mathbf{y} = \mathbf{x} + \sum_{i \in \mathcal{T}(x)} g_i(x) E_i(\mathbf{x}) + E_{\text{shared}}(\mathbf{x})
$$

其中：

- $\mathcal{T}(x)$ 是路由器为当前 token 选出的 top-k routed experts；
- $g_i(x)$ 是对应路由权重；
- $E_{\text{shared}}$ 则表示始终存在的共享专家通道。

这背后的逻辑很务实：**该分工的能力继续分工，该共享的能力不要重复学。**

### 第三步：把 MoE 的目标从“更多参数”改成“更高参数利用率”

DeepSeekMoE 真正强调的是参数利用率，而不是总参数数字本身 [DeepSeekMoE, Sections 1, 3]。

也就是说，它关心的问题不是：

- “我是不是能堆出更大的稀疏模型？”

而是：

- “这些额外参数有没有被组织成更清晰的专家分工？”

这也是为什么 DeepSeekMoE 的叙事重心一直放在 `expert specialization` 上，而不是只放在 FLOPs 节省上。

### 第四步：让这套结构仍然可训练、可部署

更细的 experts 会带来一个副作用：路由、负载均衡和并行执行都会变复杂 [DeepSeekMoE, Sections 3.3, 4.1.2]。

所以 DeepSeekMoE 并不是只给出一个更“漂亮”的结构，还配套处理了：

- expert-level balance；
- device-level balance；
- sparse training 的工程实现；
- 推理阶段的部署可行性。

这点很关键，因为 DeepSeek 后面整条路线都建立在这个前提上：**架构上的参数效率收益，必须能被系统侧兑现。**

### 这套设计带来的直接优点

把 DeepSeekMoE 的收益压缩成几句人话，大概就是：

- **专家更容易专门化**：减少“大家都学一点同样东西”的冗余；
- **公共能力不再被重复学习**：shared expert 负责通用底盘；
- **同样预算下有效容量更高**：激活参数不大，但总容量更能转成实际能力；
- **为后续 V2/V3 铺路**：后面 MLA、路由约束和系统协同，都是建立在“MoE 值得继续做大”这个前提上。

## 数据怎么说明这些优点

### 证据一：16B 模型用更少计算拿到接近甚至更好的能力

DeepSeekMoE 16B 总参数是 `16.4B`，但每个 token 只激活约 `2.8B` 参数；在预训练计算量约为 DeepSeek 7B 的 `40.5%` 时，它已经能达到与 7B dense 模型相当甚至更强的综合表现 [DeepSeekMoE, Sections 5.1-5.2]。

这说明一件事：**更多总参数只有在组织方式更好时，才会真的变成能力。**

### 证据二：推理吞吐也体现出稀疏结构的实际价值

论文里给出的结果显示，DeepSeekMoE 16B 可以在单卡 `40GB` 显存条件下部署，并在适当优化后接近 `2.5×` 于同级 7B dense 模型的推理速度 [DeepSeekMoE, Section 5.2.1]。

这说明 DeepSeekMoE 的收益不只是离线训练图表好看，而是已经开始进入部署层面的性价比讨论。

### 证据三：145B 阶段继续说明这不是小模型偶然现象

DeepSeekMoE 还把这套思路扩展到更大模型，并开始认真处理 device-level balance 等更大规模问题 [DeepSeekMoE, Sections 7.1-7.2]。

这很重要，因为它说明 DeepSeek 并不是只在“中小规模稀疏模型”上碰巧做对了，而是在验证：**如果专家真的能分工清楚，MoE 路线是可以继续放大的。**

## 思考问题

- 如果没有 `shared experts`，DeepSeekMoE 最可能先在哪个问题上吃亏：专家冗余、训练不稳，还是部署复杂度？
- 把 experts 切得更细，为什么不一定意味着每个 token 要激活更多专家？
- DeepSeekMoE 的核心更像“节省算力”，还是“提升参数组织效率”？为什么？
