# Transformer 推理所需的最小知识

## 要点

- 推理视角只关心：张量形状、数据流、算子热点、KV cache、prefill/decode 两阶段差异
- 绝大多数算力集中在：线性层（GEMM）与 attention 相关子图
- 从 CS336 的视角看，理解 Transformer 不能只停在结构图上，还要能把超参数映射到 **FLOPs、显存、带宽、吞吐**
- 对推理优化工程师来说，最重要的不是把结构图背出来，而是知道：**哪几个超参数在改系统成本，哪几个模块会成为真正热点**。

## 通用知识

### 它是什么

站在推理视角，Transformer 可以粗略看成：

- 输入 token 先变成 hidden states
- hidden states 经过多层 block 反复更新
- 每层 block 里主要由 attention 和 MLP 两大部分组成

如果只保留和系统最相关的对象，可以把一个 decoder-only Transformer 简化成：

- Embedding
- 多层 `Attention + MLP + Residual + Norm`
- 最后输出 logits

### 它解决什么问题

从模型语义上说，Transformer 让每个 token 可以：

- 通过 attention 读取上下文信息
- 通过 MLP 做更强的非线性表示变换

从系统角度说，它决定了：

- 哪些张量会反复被读写
- 哪些子图会变成 GEMM 热点
- 为什么 prefill 和 decode 会呈现两种完全不同的负载形态

### 为什么在 AI 系统里重要

因为你做推理优化时，优化对象通常不是“模型”这个抽象词，而是：

- QKV 投影 GEMM
- attention kernel
- MLP 两个大线性层
- KV cache 的读写和布局
- norm / activation 这类 pointwise 链

如果不先理解 Transformer 的最小计算结构，后面看 serving、kernel、cache、量化、profiling 都会像在雾里开灯——有光，但不一定照到路。

### 它的收益与代价

收益：

- 结构统一，易扩展
- attention + MLP 的组合非常适合现代硬件做矩阵计算优化
- 很多系统优化都能围绕固定热点展开

代价：

- 上下文变长时，attention 与 KV cache 成本会迅速放大
- decode 阶段 workload 很碎，小 shape 和访存问题会冒出来
- 同样一个结构，在训练和推理下的瓶颈并不相同

## 组件速记（推理视角）

- Embedding：token → hidden
- Block（重复 N 层）：
  - Attention（QKV 投影 + 注意力）
  - MLP（两层线性 + 激活，如 GELU/SwiGLU）
  - Residual + Norm

如果只为了面试口述，最短版本可以说成：

- Transformer block 的大头是 attention 和 MLP
- 系统里最常见热点是 QKV / output projection / MLP 的 GEMM，以及 attention 和 KV cache 相关访问

## 训练/推理都要能看懂的超参数

设：

- batch = $B$
- seq = $S$
- hidden = $H$
- heads = $N_h$
- head dim = $D_h$
- layers = $L$
- MLP hidden = $D_{ff}$

这些参数决定的不是“风格”，而是成本：

- $H$ 决定绝大多数线性层与激活张量规模
- $D_{ff}$ 直接决定 MLP 的参数量与 FLOPs
- $S$ 决定 prefill 成本，也决定 attention 的 $S^2$ 部分何时开始主导
- $L$ 近似线性放大参数量、激活开销和 KV cache 大小

一个很实用的工程直觉：

- 短上下文时，很多成本更像是“堆 block 数量和 hidden size”
- 长上下文时，attention 的二次项会迅速放大

再补一个很常用的判断：

- hidden size 往往决定单层大矩阵乘有多重
- layers 往往决定“这个成本被重复多少次”
- sequence length 往往决定 prefill 和 KV cache 压力会不会突然跃迁

## 一个最小 block 应该怎么讲

如果只讲一个 decoder block，可以按下面顺序说：

1. 输入 hidden states 先做 norm
2. 进入 attention，产生新的上下文相关表示
3. 做一次 residual
4. 再进入 MLP，提升表示能力
5. 再做 residual，进入下一层

从系统角度最关键的是：

- attention 更依赖上下文与 cache
- MLP 更像规整的大矩阵计算

所以有些 workload attention 更痛，有些 workload 反而 MLP 更重，不能一概而论

## 关键张量形状（示例）

假设：

- batch = B
- seq = S
- hidden = H
- heads = Nh
- head_dim = Dh，通常 $H = Nh \times Dh$

常见：

- 输入 hidden：`[B, S, H]`
- Q/K/V：`[B, S, Nh, Dh]`（实现里可能会打平或换维度以便 GEMM）

再往下讲一个够用的形状直觉：

- attention 本质上是在 head 维度上，把 query 和历史 key/value 组织成更适合局部并行的结构
- 所以系统里经常会看到 transpose、reshape、pack/unpack，这些虽然不是“数学公式”的主体，却常常决定真实性能

## 每层里真正贵的计算在哪里

从实现角度，一层 decoder block 的热点通常是：

1. QKV 投影：本质是 3 个大 GEMM
2. Attention 分数与加权求和：
   - prefill 时更像大矩阵乘
   - decode 时更像“读很多历史 KV”
3. 输出投影：又是 GEMM
4. MLP：两次线性层 + 激活

一个够用的判断方式：

- Prefill：更像批量 dense compute
- Decode：更像小 shape + 高访存负载

更进一步的工程翻译是：

- prefill 容易让大 GEMM 吃满硬件
- decode 更容易被 kernel launch、KV 读取、同步和小矩阵效率拖住

## Prefill vs Decode

- Prefill：一次处理整段上下文，`S` 大 → GEMM 更“饱满”
- Decode：每步 `S≈1` → 更容易被 kernel launch、访存与同步开销主导

这是推理系统里最值得背熟的一条分界线：

- prefill 和 decode 不是同一个问题的轻重版本
- 它们常常是两种完全不同的系统负载

## 最小例子

假设：

- `B = 8`
- `S = 2048`
- `H = 4096`
- `L = 32`

那么你至少可以马上做出几个判断：

- prefill 会很重，因为整段上下文一次进模型
- KV cache 也会不小，因为每层都要存 K/V
- layers 越多，同样的 block 成本会被重复很多次

如果把 `S` 再从 2048 提到 8192：

- prefill 成本会显著变重
- attention 相关内存和访存压力会更明显
- decode 侧的 KV cache 占用也会明显上升

这就是为什么很多“长上下文模型”讨论，最后都会落回系统成本，而不只是模型结构本身。

## 工程例子

一个非常典型的线上现象：

- 同一模型在离线单请求 benchmark 下看起来没问题
- 一上线上高并发，TTFT 和 TPOT 同时变差

这通常不是一句“Transformer 很大”就能解释的。

更合理的拆法是：

- TTFT 可能被长 prompt 的 prefill 和 queue 拖高
- TPOT 可能被 decode 阶段的 KV 读取和小 shape kernel 拖高
- 如果 context 再拉长，attention 和 KV cache 会一起变得更贵

这就是为什么理解 Transformer 时，必须把：

- 结构
- shape
- cache
- workload 阶段

放在一张图里看。

## 你需要具备的最小架构判断

- 为什么 RoPE 比绝对位置编码更适合现代 decoder-only LLM
- 为什么 Pre-Norm / RMSNorm 更容易稳定深层训练
- 为什么 MLP 宽度、attention 头数和 context length 不应孤立讨论
- 为什么“训练最优结构”和“推理最优结构”不一定完全一致

## 推理优化工程师视角

如果你以后主要做的是推理引擎、serving、kernel 或 profiling，那这篇至少要帮你建立 4 个本能：

1. 看到 hidden size、layers、seq length，要能立刻想到成本往哪里走
2. 看到 prefill 变慢，要先想到 attention / GEMM / prompt 长度
3. 看到 decode 变慢，要先想到 KV cache / 小 shape kernel / 调度
4. 不把“模型结构理解”停留在论文图，而是落到张量形状与热点算子

会这一层，你读后面的：

- `02-attention-kv-cache.md`
- `04-llm-serving.md`
- `06-flashattention-io-aware.md`

就会顺很多。

## 常见面试问题

### 初级

1. 一个 decoder-only Transformer block 主要由哪些部分组成？
2. 为什么说 attention 和 MLP 是最主要的两类计算热点？

### 中级

1. hidden size、layers、sequence length 分别更直接影响哪些成本？
2. 为什么 prefill 和 decode 的瓶颈常常不同？

### 高级

1. 如果一个模型长上下文下性能明显变差，你会优先怀疑 Transformer 的哪些成本项？
2. 为什么“训练最优结构”和“推理最优结构”不一定一致？

## 易错点

- 只按“训练视角”理解模型，忽略推理的缓存与动态 batching
- 混淆 attention 的数学公式与实际实现的 layout/reshape
- 把超参数当成“经验配置”，而不去问它们具体改变了哪些张量与成本
- 只会画结构图，不会把结构图翻译成 shape、FLOPs、显存和访问模式

## 排查 checklist

- [ ] 你的实现中 QKV 是 fused 还是分开算？shape 如何变化？
- [ ] Decode 阶段的热点在哪：attention 还是 MLP？
- [ ] KV cache 的 layout（按 token 连续 vs 分页）是什么？
- [ ] hidden size、d_ff、num_heads 变化后，哪个算子成本涨得最快？

## 参考资料

- Transformer / decoder-only LLM 基础资料
- CS336 相关讲义与笔记
- 后续建议串读：`02-attention-kv-cache.md`、`06-flashattention-io-aware.md`、`04-llm-serving.md`
