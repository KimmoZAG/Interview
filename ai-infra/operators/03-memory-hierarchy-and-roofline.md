# 内存层级与性能模型（Roofline）

## 要点

- 用 roofline 把问题先粗分：**带宽受限** vs **算力受限**
- 性能优化要“有证据”：先测，再改，再验证

## 内存层级（概念级）

- CPU：L1/L2/L3、NUMA、主存
- GPU：寄存器、shared memory、L2、HBM（以及可能的 L1/texture cache）

## Roofline 的最小用法

- 算术强度：$AI = \frac{FLOPs}{Bytes}$
- 经验法：
  - AI 低：多半带宽瓶颈 → 减少访存、提高复用、融合
  - AI 高：可能算力瓶颈 → 用更高效 kernel、使用更低精度、提高并行度

## 推理里常见现象

- Decode 阶段常见：小矩阵/小 batch → launch 与访存占主导
- Prefill 阶段常见：大 GEMM → 更接近算力瓶颈

## 排查 checklist

- [ ] 你能给出这个算子的 AI 粗估吗？
- [ ] profiler 显示的 achieved bandwidth / achieved FLOPs 大概多少？
- [ ] 优化后是否同时验证“正确性 + 性能回归曲线（不同 shape）”？
