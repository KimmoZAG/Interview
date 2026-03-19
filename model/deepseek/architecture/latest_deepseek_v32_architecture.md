# 最新 DeepSeek 架构（V3.2）：为什么它像一台已经拼好的整机

## 关键结论

如果前面的几篇页面是在分别讲 `DeepSeekMoE`、`MLA`、路由和长上下文，那么这一页要回答的是另一个更直接的问题：**到了最新一代，DeepSeek 到底把哪些东西已经做成了默认架构常量。**

- 从 `DeepseekV32ForCausalLM` 配置看，DeepSeek 的主体仍然是 `MLA + DeepSeekMoE`，并没有回到常规 dense Transformer 路线 [DeepSeek-V3.2 config]。
- 长上下文、细粒度 MoE、受约束路由、`MTP` 和 `FP8` 已经不再像附加能力，而更像一起打包进入模型默认形态 [DeepSeek-V3.2 config]。
- 这说明 `DeepSeek-V3.2` 的重点，不是再单独发明一个全新部件，而是把前几代验证过的路线收敛成一台更完整的“默认机器”。

所以这一页最重要的结论是：**如果 `DeepSeek-V2` 解决的是 MLA，`DeepSeek-V3` 解决的是超大 MoE 的系统协同，那么 `DeepSeek-V3.2` 更像是在说——这些东西现在要一起工作，而不是各自成立就算完成。**

## 背景：为什么还需要一页“最新版整机导读”

### 旧的阅读方式为什么不够

前面的架构页大多是按模块拆开的：

- `deepseek_moe.md` 看专家怎么分工；
- `mla_attention.md` 看 KV cache 为什么能压下来；
- `routing_and_load_balancing.md` 看 MoE 为什么不会先死在通信里；
- `long_context_and_yarn.md` 看长上下文为什么能做长。

这种拆法很适合把每个机制讲清楚，但不太适合回答一个更现实的问题：**当你拿到最新版配置时，应该怎样快速判断这是不是一条前后自洽的架构路线。**

因为配置文件最有价值的地方，不在于它花样多，而在于它会暴露很多“已经被默认化的设计选择”：

- 层数和 hidden size 说明 backbone 还在往什么方向扩；
- `q_lora_rank`、`kv_lora_rank` 和 QK 分解维度说明 attention 侧仍在坚持什么；
- routed/shared experts、top-k、group 相关字段说明 MoE 已经被组织成什么运行形态；
- `YaRN`、`163840`、`FP8`、`num_nextn_predict_layers` 则说明长上下文、低精度和训练目标已经进入默认配置预算 [DeepSeek-V3.2 config]。

### 这一页真正想解决什么

这一页主要想讲清楚四件事：

1. 最新配置里最值得看的信号有哪些；
2. 这些信号分别对应前几代 DeepSeek 的哪条主线；
3. 为什么说 V3.2 更像“整机收敛”，而不是“再多一个局部创新”；
4. 这对训练、推理和面试表达分别意味着什么。

## DeepSeek-V3.2 具体怎么做

### 第一步：继续用 MLA 压 attention 状态，而不是回退到常规 KV 结构

从配置字段可以直接看到：

- `q_lora_rank = 1536`
- `kv_lora_rank = 512`
- `qk_rope_head_dim = 64`
- `qk_nope_head_dim = 128`
- `num_attention_heads = 128` [DeepSeek-V3.2 config]

这些字段最重要的含义不是数字本身，而是它们说明 DeepSeek 仍然在坚持 MLA 路线：

- Q 和 KV 的压缩强度并不对称；
- 位置相关和位置无关的 QK 子空间继续拆开；
- attention 的主任务仍然是把长上下文下最贵的状态成本压下来。

也就是说，到了 V3.2，DeepSeek 仍然没有放弃那条很核心的判断：**attention 里最该先优化的，不是实现优雅度，而是 KV cache 的长期成本。**

### 第二步：继续用细粒度 MoE 扩容量，但不把所有层都一刀切稀疏化

配置里另一组关键字段是：

- `n_routed_experts = 256`
- `num_experts_per_tok = 8`
- `n_shared_experts = 1`
- `moe_layer_freq = 1`
- `first_k_dense_replace = 3` [DeepSeek-V3.2 config]

这组信息连起来看，会得到一个很清晰的结构判断：

- DeepSeek 仍然押注“专家更多、分工更细”的路线；
- 每个 token 依然不是极端稀疏地只走一两个专家，而是保留相对更宽的激活；
- 但最前面的少数层并没有直接做成 MoE，而是保留 dense block。

这说明 V3.2 的思路不是“所有地方越稀疏越好”，而是更像：

- 浅层先稳住共享表示；
- 中后层再把容量交给更细的专家分工；
- shared expert 继续承担共通知识底盘。

换句话说，**V3.2 不是把 MoE 做得更粗暴，而是把“哪里该共享、哪里该分流”做得更精细。**

### 第三步：让路由继续受系统约束，而不是自由竞争式 top-k

配置里和路由有关的字段包括：

- `scoring_func = sigmoid`
- `topk_method = noaux_tc`
- `topk_group = 4`
- `n_group = 8`
- `norm_topk_prob = true`
- `routed_scaling_factor = 2.5` [DeepSeek-V3.2 config]

即使不去假装还原完整实现，也足够看出一个方向：**这不是“256 个专家一起打分，谁高谁上”的完全自由式路由。**

更合理的理解是：

- experts 会被组织在 group 里；
- top-k 选择要接受 group 或系统友好的约束；
- 最终概率还会做归一化与缩放。

这正好延续了 DeepSeek 在 V2/V3 里反复强调的路子：MoE 的上限不只是专家够不够多，而是路由是不是足够系统友好 [DeepSeek-V2, Sections 2.2.2-2.2.3; DeepSeek-V3, Section 2.1.2]。

### 第四步：把长上下文、MTP 和 FP8 都写成默认预算的一部分

V3.2 最能体现“整机感”的，反而不是某一个单独字段，而是下面这几组配置同时出现：

- `max_position_embeddings = 163840`
- `rope_scaling.type = yarn`
- `rope_scaling.factor = 40`
- `num_nextn_predict_layers = 1`
- `quant_method = fp8`
- `fmt = e4m3` [DeepSeek-V3.2 config]

这组组合的含义非常强：

- 长上下文已经不是外挂模式，而是默认工作区间；
- 训练目标仍然在为更快推理和更密训练信号服务；
- 低精度也已经不只是部署小技巧，而是模型分发形态的一部分。

因此，V3.2 这份配置最像的不是“模型又多了几个参数项”，而是：**DeepSeek 把长上下文、推理友好训练目标和低精度运行条件都并入了同一套默认架构预算。**

### 这套设计带来的直接优点

把 V3.2 的结构收益压缩一下，大概就是四条：

- **整条主线更闭环**：MLA、MoE、路由、长上下文、MTP、FP8 已经开始一起定义模型形态；
- **默认部署感更强**：不是“先训完再想怎么跑”，而是配置本身已经在考虑运行条件；
- **更适合长上下文与大规模推理**：attention、路由和低精度都在为系统成本让路；
- **更容易作为后续 reasoning 底座**：训练目标和系统形态没有脱节。

## 数据怎么说明这些优点

### 证据一：配置本身已经把前几代主线固化在一起

如果只看 V2 和 V3 的论文，你会分别看到 MLA、超大 MoE、FP8、系统协同这些主线；而到了 V3.2 的配置里，这些东西已经不再散落在不同章节，而是直接一起出现在默认模型字段中 [DeepSeek-V2, Section 2.1; DeepSeek-V3, Sections 2.1-3.3; DeepSeek-V3.2 config]。

这说明 V3.2 的价值，不只是“某项能力更强”，而是**前几代验证过的方向已经被收敛成标准形态。**

### 证据二：长上下文扩展倍率说明它不是温和微调

从原始位置长度到当前上限，可以得到一个很直观的倍率：

$$
\frac{163840}{4096} = 40
$$

这与 `YaRN factor = 40` 是对齐的 [DeepSeek-V3.2 config]。这说明 V3.2 的长上下文不是象征性上调，而是整数量级级别的预算重写。

### 证据三：激活方式说明它仍然在追求“高总容量 + 系统可兑现”

仅从配置直觉出发，单个 token 在一个 MoE block 中大致会经过：

$$
n_{\text{active}} \approx 8 + 1 = 9
$$

也就是 `8` 个 routed experts 加 `1` 个 shared expert [DeepSeek-V3.2 config]。

这说明 V3.2 既没有回退到 dense，也没有走到“极端稀疏得只剩一条细线”的另一端，而是在继续追求：

- 专家足够多；
- 激活仍然可控；
- 总容量能被系统侧兑现出来。

## 思考问题

- 如果你把 V3.2 看成一台整机，而不是一组技巧拼盘，那么它最核心的主轴到底是 `MLA + MoE`，还是“训练—推理一体化预算设计”？
- 前 `3` 层保留 dense，这更像表达稳定性考虑，还是在给后面的 MoE 路由减压？
- 当 `160K+`、`MTP` 和 `FP8` 都进入默认配置后，你觉得未来讨论“模型架构”时，还能把系统与部署完全放到页外吗？
