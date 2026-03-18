# Transformer 推理所需的最小知识

## 一句话先讲清

这一页只回答一个问题：**作为推理优化工程师，你到底需要懂 Transformer 到什么程度，才能把结构图翻译成真实的 FLOPs、显存、带宽、吞吐和延迟问题。**

重点不是把论文框图背得像唱段子，而是知道：**哪些模块是真正热点，哪些超参数在改系统成本，为什么 prefill 和 decode 会像两种不同的负载。**

## 关联知识网络

- 结构成本放大器：[`Attention、KV cache 与吞吐/延迟`](02-attention-kv-cache.md)
- 数值与 pointwise 链：[`常见层：Norm/激活/残差 与数值稳定性`](03-norm-activation-stability.md)
- 输入输出配置：[`Tokenizer 与采样`](04-tokenization-and-sampling.md)
- 线上落地：[`推理栈全景`](../02-inference-engine/01-inference-stack-overview.md)、[`LLM Serving`](../02-inference-engine/04-llm-serving.md)
- 算子实现：[`FlashAttention 与 IO-aware`](../01-operator-optimization/06-flashattention-io-aware.md)

## 为什么值得先学这一页

- 不理解 Transformer 最小结构，后面看 serving、kernel、cache、profiling 就容易像在雾里找楼梯。
- 推理优化里的热点通常不是“模型”这个抽象词，而是 QKV、attention、MLP、KV cache 和 pointwise 链。
- 很多性能问题并不是“模型大”，而是某个超参数把某类成本突然放大了。

## 最小结构：只保留系统最关心的部分

站在 decoder-only 推理视角，可以把 Transformer 粗略看成：

- Embedding：token → hidden states
- 重复 $L$ 层 block：`Norm → Attention → Residual → Norm → MLP → Residual`
- 最后输出 logits

如果只是面试口述，最短版本可以说：

- 每层 block 的计算大头是 **Attention + MLP**
- 系统热点通常是 **QKV / output projection / MLP 的 GEMM**，以及 **attention 与 KV cache 的读写**

## 为什么它在系统里重要

| 模块 | 模型作用 | 系统侧代价 |
|---|---|---|
| Attention | 读取上下文信息 | prefill 计算重、decode 依赖 KV 读取 |
| MLP | 提升非线性表示能力 | 大 GEMM，吞吐热点常驻 |
| Norm / Residual | 稳定深层训练与推理 | pointwise、memory-bound、适合融合 |
| KV Cache | 复用历史上下文 | 长上下文下显存与访存压力显著 |

所以推理优化时，你真正优化的往往不是“Transformer”，而是这些成本项的具体实现。

## 超参数一变，哪类成本会跟着变

设：

- batch = $B$
- sequence length = $S$
- hidden size = $H$
- heads = $N_h$
- head dim = $D_h$
- layers = $L$
- MLP hidden = $D_{ff}$

这些参数决定的不是“风格”，而是成本账本：

| 超参数 | 更直接影响什么 |
|---|---|
| $H$ | 线性层规模、激活张量大小、单层 GEMM 重量 |
| $D_{ff}$ | MLP 参数量与 FLOPs |
| $S$ | prefill 成本、attention 压力、长上下文风险 |
| $L$ | 参数量、激活开销、KV cache 总规模近似线性放大 |
| $N_h, D_h$ | attention 结构、张量布局、实现效率 |

一个很实用的工程直觉是：

- 短上下文时，成本更多像“堆 hidden size 和 layers”
- 长上下文时，attention 与 KV cache 的成本会更早跳出来抢戏

## 一个 block 到底怎么讲才像工程师

如果只讲一个 decoder block，可以按下面顺序说：

1. 输入 hidden states 先做 norm
2. 进入 attention，结合上下文生成新的表示
3. 做 residual
4. 再进入 MLP，做更强的非线性变换
5. 再做 residual，进入下一层

从系统角度最关键的是：

- **Attention 更依赖上下文和 cache**
- **MLP 更像规整的大矩阵计算**

这也意味着不同 workload 的痛点可能完全不同：有些是 attention 更贵，有些反而是 MLP 更重。

## 关键张量形状，至少要有这个手感

假设：

- 输入 hidden：`[B, S, H]`
- Q/K/V：常见可视为 `[B, S, N_h, D_h]`

一个够用的形状直觉是：

- attention 的数学形式很重要，但**layout、transpose、reshape、pack/unpack** 往往更决定真实性能
- 这也是为什么同一数学公式，换一种实现后速度可能差很多

## 每层里真正贵的计算在哪里

一层 decoder block 的热点通常包括：

1. QKV 投影：本质是大 GEMM
2. Attention 分数与加权求和
  - prefill 时更像大矩阵乘
  - decode 时更像频繁读取历史 KV
3. 输出投影：仍然是 GEMM
4. MLP：两次线性层 + 激活

一句翻译成人话：

- **Prefill 更像批量 dense compute**
- **Decode 更像小 shape + 高访存 + 调度开销**

## Prefill vs Decode：一定要分开看

| 阶段 | 负载特征 | 常见瓶颈 |
|---|---|---|
| Prefill | 一次处理整段上下文，$S$ 大 | GEMM、attention、大量 prompt token |
| Decode | 每步通常只生成少量 token，形状更碎 | KV 读取、小 shape kernel、同步与 launch 开销 |

这是推理系统里最值得背熟的一条线：**prefill 和 decode 不是同一个问题的轻重版本，而是两种几乎不同的系统负载。**

## 最小工程例子

假设：

- `B = 8`
- `S = 2048`
- `H = 4096`
- `L = 32`

你至少应该立刻想到：

- prefill 很重，因为整段上下文要一次进模型
- KV cache 不小，因为每层都要存历史 K/V
- layers 越多，同样的 block 成本会被重复很多次

如果把 `S` 从 2048 拉到 8192：

- prefill 会显著变重
- attention 与访存压力会更明显
- decode 侧的 KV cache 占用也会持续上升

所以很多“长上下文模型”的讨论，最后都会落回系统成本，而不只是模型结构本身。

## Troubleshooting：为什么离线看还好，一上并发 TTFT 和 TPOT 一起变差

| 现象 | 第一怀疑点 | 如何拆解 |
|---|---|---|
| TTFT 明显升高 | 长 prompt prefill 或 queue 堵塞 | 看 prompt 长度分布、queue wait、prefill profile |
| TPOT 明显升高 | decode 小 shape、KV 读取、调度开销 | 看 decode 阶段 profile、KV cache 访问 |
| 长上下文下显存吃紧 | KV cache 与 attention 成本被放大 | 联合看 KV 占用和上下文长度分布 |
| GPU 利用率不差但体验不佳 | prefill / decode 瓶颈不在同一层 | 分段看阶段指标，不要只看平均值 |

## 推理优化工程师视角

这一页最重要的价值，是帮你建立 4 个本能：

1. 看到 hidden size、layers、sequence length，能立刻想到成本往哪里走。
2. 看到 prefill 变慢，先想到 attention / GEMM / prompt 长度。
3. 看到 decode 变慢，先想到 KV cache / 小 shape kernel / 调度。
4. 不把“理解模型结构”停留在论文图，而是落到张量形状、热点算子和访存模式。

## 面试高频问法

### 初级

1. 一个 decoder-only Transformer block 主要由哪些部分组成？
2. 为什么 attention 和 MLP 是最主要的两类计算热点？

### 中级

1. hidden size、layers、sequence length 分别更直接影响哪些成本？
2. 为什么 prefill 和 decode 的瓶颈常常不同？

### 高级

1. 如果一个模型在长上下文下性能明显变差，你会优先怀疑哪些成本项？
2. 为什么“训练最优结构”和“推理最优结构”不一定完全一致？

## 易错点

- 只按训练视角理解模型，忽略推理里的 cache 与 batching
- 混淆 attention 数学公式和真实实现的 layout / reshape 成本
- 把超参数当经验值，不追问它到底改了哪些张量与成本
- 只会画结构图，不会把结构图翻译成 shape、FLOPs、显存和访问模式

## 排查 checklist

- [ ] 你的实现中 QKV 是 fused 还是分开算？shape 怎么变化？
- [ ] decode 阶段的热点更偏 attention 还是 MLP？
- [ ] KV cache 的 layout 是连续还是分页？
- [ ] hidden size、$D_{ff}$、num_heads 变化后，哪个成本涨得最快？

## 参考资料

- Transformer / decoder-only LLM 基础资料
- CS336 相关讲义与笔记
- 建议串读：[`Attention、KV cache 与吞吐/延迟`](02-attention-kv-cache.md)、[`FlashAttention 与 IO-aware`](../01-operator-optimization/06-flashattention-io-aware.md)、[`LLM Serving`](../02-inference-engine/04-llm-serving.md)
