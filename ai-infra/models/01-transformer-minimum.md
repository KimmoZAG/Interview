# Transformer 推理所需的最小知识

## 要点

- 推理视角只关心：张量形状、数据流、算子热点、KV cache、prefill/decode 两阶段差异
- 绝大多数算力集中在：线性层（GEMM）与 attention 相关子图
- 从 CS336 的视角看，理解 Transformer 不能只停在结构图上，还要能把超参数映射到 **FLOPs、显存、带宽、吞吐**

## 组件速记（推理视角）

- Embedding：token → hidden
- Block（重复 N 层）：
  - Attention（QKV 投影 + 注意力）
  - MLP（两层线性 + 激活，如 GELU/SwiGLU）
  - Residual + Norm

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

## 关键张量形状（示例）

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

这也是为什么同一个模型，在训练/prefill 与在线 decode 的瓶颈完全可能不同。

## Prefill vs Decode

- Prefill：一次处理整段上下文，`S` 大 → GEMM 更“饱满”
- Decode：每步 `S≈1` → 更容易被 kernel launch、访存与同步开销主导

从 CS336 的语言讲：

- Prefill 更接近“吞进去一段序列做并行计算”
- Decode 更接近“单步自回归生成”，系统开销和缓存设计开始压过纯数学公式

## 训练视角下最容易被忽略的资源账

虽然这篇是推理最小知识，但如果不理解训练资源账，很多架构判断会失真：

- 参数：决定静态权重内存
- 激活：决定训练时的主要动态显存之一
- 梯度：与参数同阶
- 优化器状态：AdamW 常常还要再额外保存一到两倍参数规模的状态

因此，现代 LLM 架构讨论里“这个设计更好”往往隐含着另一个问题：**它是否还能在给定显存和吞吐预算下训练得动**。

## 你需要具备的最小架构判断

- 为什么 RoPE 比绝对位置编码更适合现代 decoder-only LLM
- 为什么 Pre-Norm / RMSNorm 更容易稳定深层训练
- 为什么 MLP 宽度、attention 头数和 context length 不应孤立讨论
- 为什么“训练最优结构”和“推理最优结构”不一定完全一致

## 易错点

- 只按“训练视角”理解模型，忽略推理的缓存与动态 batching
- 混淆 attention 的数学公式与实际实现的 layout/reshape
- 把超参数当成“经验配置”，而不去问它们具体改变了哪些张量与成本

## 排查 checklist

- [ ] 你的实现中 QKV 是 fused 还是分开算？shape 如何变化？
- [ ] Decode 阶段的热点在哪：attention 还是 MLP？
- [ ] KV cache 的 layout（按 token 连续 vs 分页）是什么？
- [ ] hidden size、d_ff、num_heads 变化后，哪个算子成本涨得最快？

## CS336 对照

- 官方 lecture 对应：Lecture 2（resource accounting）、Lecture 3（architectures, hyperparameters）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://rd.me/cs336
  - https://www.rajdeepmondal.com/blog/cs336-overview
  - https://bearbearyu1223.github.io/posts/cs336-transformer-architecture-overview/
