# Transformer 推理所需的最小知识

## 要点

- 推理视角只关心：张量形状、数据流、算子热点、KV cache、prefill/decode 两阶段差异
- 绝大多数算力集中在：线性层（GEMM）与 attention 相关子图

## 组件速记（推理视角）

- Embedding：token → hidden
- Block（重复 N 层）：
  - Attention（QKV 投影 + 注意力）
  - MLP（两层线性 + 激活，如 GELU/SwiGLU）
  - Residual + Norm

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

## Prefill vs Decode

- Prefill：一次处理整段上下文，`S` 大 → GEMM 更“饱满”
- Decode：每步 `S≈1` → 更容易被 kernel launch、访存与同步开销主导

## 易错点

- 只按“训练视角”理解模型，忽略推理的缓存与动态 batching
- 混淆 attention 的数学公式与实际实现的 layout/reshape

## 排查 checklist

- [ ] 你的实现中 QKV 是 fused 还是分开算？shape 如何变化？
- [ ] Decode 阶段的热点在哪：attention 还是 MLP？
- [ ] KV cache 的 layout（按 token 连续 vs 分页）是什么？
