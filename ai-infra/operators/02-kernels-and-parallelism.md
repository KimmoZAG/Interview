# Kernel 与并行基础（SIMD/SIMT）

## 要点

- GPU 上常见瓶颈：访存（带宽/缓存命中）与同步（block 间/warp 内）
- 很多“慢”的原因不是算子复杂，而是 **launch 次数太多** 或 **融合不足**

## 你需要能解释的最小执行模型

- CPU：SIMD（向量化）、cache、分支预测
- GPU：SIMT（warp/线程块）、内存合并访问（coalescing）、共享内存/寄存器

## Kernel 性能的常见决定因素

- 并行度：线程/warp 是否足够填满 SM
- 内存访问：是否连续、是否对齐、是否复用（cache/shared memory）
- 计算密度：FLOPs/Byte（算术强度）
- 同步与原子：是否引入大量 serialization

## 最小例子建议（后续你可补代码/伪代码）

- GEMM（矩阵乘）是很多算子的核心
- LayerNorm / Softmax 的数值稳定写法（减 max）

## 易错点

- 只看 FLOPs 忽略带宽：算子明明“计算不多”却很慢
- 小 batch / 小 shape 下 kernel launch 开销占主导

## 排查 checklist

- [ ] 单次 kernel 时间 vs kernel 次数（是不是太碎）
- [ ] 带宽是否接近峰值？若很低，访问模式是否不连续
- [ ] 是否能通过融合/合并算子减少 launch
