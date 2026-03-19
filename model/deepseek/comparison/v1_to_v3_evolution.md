# DeepSeek 代际演进：从稀疏架构到推理强化

## 关键结论

DeepSeek 这条路线最容易被误读成“每一代都只是更大一点、分数更高一点”。更准确的理解是：**它几乎每一代都在优先处理当时最贵的瓶颈，然后把省下来的预算继续投到下一层能力上。**

- `DeepSeekMoE` 先解决“参数怎么真正变成有效容量”，重点是专家专门化 [DeepSeekMoE, Sections 1, 3]。
- `DeepSeek-V2` 再解决“即使 MoE 有效，推理缓存和通信成本会不会先把收益吃掉”，重点是 `MLA + device-limited routing` [DeepSeek-V2, Sections 2.1-2.2]。
- `DeepSeek-V3` 继续解决“当规模上到 671B / 37B 激活时，训练、通信和部署还能不能一起成立”，重点是系统协同 [DeepSeek-V3, Sections 2-3]。
- `DeepSeek-R1` 最后把重心推进到 reasoning：如果 base model 已足够强，能不能用 RL 直接放大推理行为本身 [DeepSeek-R1, Sections 1-3]。

所以这一页最重要的结论是：**DeepSeek 的代际演进不是单纯 scaling，而是“架构稀疏化 → KV 压缩 → 系统协同 → RL reasoning” 的连续接力。**

## 背景：为什么只看排行榜会读错 DeepSeek

### 旧的理解为什么不够

做模型代际比较时，最常见的误区有两个。

第一个误区是只看 benchmark 排名，于是把演进理解成：

- 参数更大；
- 分数更高；
- 所以只是普通的规模升级。

第二个误区是只看局部创新，于是把 DeepSeekMoE、MLA、DualPipe、FP8、GRPO 看成几篇彼此平行的论文技巧。

但 DeepSeek 真正特别的地方恰恰在于：**这些动作不是平行堆叠，而是在接力解决不同层面的主瓶颈。**

如果只看单篇论文，很容易觉得每一代都在换主题；如果沿着“哪一层成本最贵”去看，主线就会清楚很多。

### 这一页真正想解决什么

这一页主要想讲清楚四件事：

1. DeepSeek 每一代到底在优先解决什么问题；
2. 为什么这些问题会按这个顺序出现；
3. 每一代的收益是怎样接到下一代上的；
4. 为什么 R1 并不是突然横空出世，而是前三代工程积累的自然延伸。

## DeepSeek 具体怎么做

### 第一步：DeepSeekMoE 先回答“参数怎么变成有效容量”

DeepSeekMoE 的重点不是“终于也用了 MoE”，而是先指出传统 coarse-grained MoE 容易让 routed experts 学到重叠知识，导致参数很多，但有效分工不够清楚 [DeepSeekMoE, Sections 1, 3.1-3.2]。

因此它优先做了两件事：

- 把 experts 切得更细，让专家更容易专门化；
- 引入 `shared experts`，把共通知识从 routed experts 里拆出来。

这一步的意义在于：**先让更多参数真的变得“有用”，后面才值得继续把稀疏架构做大。**

### 第二步：DeepSeek-V2 再回答“模型怎么高效训练、也高效推理”

即使 MoE 已经有效，如果 attention 还是传统 KV 结构，长上下文和高并发生成场景下的缓存与带宽压力，仍然会快速把系统吃满。

所以到了 V2，DeepSeek 把主问题切到另一层：

- 用 `MLA` 改写 attention 状态表示，先压 `KV cache`；
- 用 `device-limited routing` 和相关 balance 机制，让 MoE 路由开始接受系统成本约束 [DeepSeek-V2, Sections 2.1-2.2]。

这一步很关键，因为它意味着 DeepSeek 不再只问“模型能不能更强”，而是开始问：**模型变强以后，推理和训练侧的账还能不能算得过来。**

### 第三步：DeepSeek-V3 把问题推进到“超大规模系统能否闭环”

到了 V3，重点已经不是证明某个结构点子成立，而是证明一整套超大规模系统可以稳定工作：

- `auxiliary-loss-free load balancing`
- `DualPipe`
- cross-node all-to-all kernels
- `FP8`
- memory saving 与部署协同 [DeepSeek-V3, Sections 2.1.2, 3.2-3.4]

也就是说，V3 的核心不只是“模型更大”，而是：**当规模上到 671B / 37B activated 后，路由、通信、精度、并行调度和部署是不是还能一起成立。**

### 第四步：DeepSeek-R1 把节省出来的预算真正投向 reasoning

到了 R1，研究主线进一步变化：重点不再是 base 架构本身，而是 reasoning 能不能被显式放大。

它先用 `R1-Zero` 检验 pure RL 是否足以诱导更长、更强的推理行为；然后再用冷启动数据、拒绝采样、SFT、第二阶段 RL 和 reward model，把 raw reasoning 收束成更可读、更可用的模型 [DeepSeek-R1, Sections 2-3]。

这一步的真正含义是：**前三代省出来的不只是训练成本，更是把预算重新分配到 reasoning 行为本身的空间。**

### 这条主线带来的直接优点

把四代主线压缩一下，可以得到四个判断：

- **DeepSeekMoE**：让参数更值钱；
- **DeepSeek-V2**：让强模型更能高效跑起来；
- **DeepSeek-V3**：让超大 MoE 真的可扩展；
- **DeepSeek-R1**：让后训练不再只是修口风，而是直接生产 reasoning 能力。

## 数据怎么说明这些优点

### 证据一：四代论文的“主结果”本身就落在不同层的瓶颈上

从论文定位就能看出接力关系：

- DeepSeekMoE 的主结果围绕 expert specialization 和参数效率 [DeepSeekMoE, Sections 3-5]；
- V2 的主结果围绕 MLA、KV cache 降幅与吞吐提升 [DeepSeek-V2, Abstract; Section 3.2.3]；
- V3 的主结果围绕超大规模训练系统、FP8、通信与部署协同 [DeepSeek-V3, Abstract; Sections 3.2-3.4]；
- R1 的主结果围绕 reasoning 行为增强与蒸馏迁移 [DeepSeek-R1, Abstract; Sections 4, 6]。

这说明 DeepSeek 的每一代都在回答“当前最贵的问题是什么”，而不是重复做同一种升级。

### 证据二：V2 与 V3 都把系统收益写成一等结果

V2 不只是说模型更强，还明确写出：

- `KV cache` 降低 `93.3%`；
- 最大生成吞吐提升 `5.76×` [DeepSeek-V2, Abstract; Section 3.2.3]。

V3 也不只是给参数规模，而是把：

- `FP8`
- DualPipe
- cross-node all-to-all
- 训练成本和部署路径

一起写进主文 [DeepSeek-V3, Sections 3.2-3.4; Table 1]。

这说明对 DeepSeek 来说，系统收益不是附录工程，而是模型路线的一部分。

### 证据三：R1 证明后训练可以变成主能力来源

R1 的关键不只是最终 benchmark，而是它明确展示：

- pure RL 可以诱导更强 reasoning；
- reasoning 轨迹还可以被再收束、再蒸馏、再迁移到更小模型 [DeepSeek-R1, Sections 2-4]。

这意味着 DeepSeek 在 R1 阶段已经不再把后训练视为“最后补一层偏好对齐”，而是把它视为真正继续生产能力的主环节。

## 思考问题

- DeepSeek 四代里，最关键的转折是哪一次：从 MoE 到 MLA，还是从 V3 到 R1？为什么？
- 如果没有 V3 的系统协同，R1 的 reasoning 放大还会成立吗？成立的是上限，还是只是实验可行性？
- 如果你要把 DeepSeek 这条路线翻译成自己团队 roadmap，你会先投架构、先投系统，还是先投 post-training？
