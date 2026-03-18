# 算子优化索引

定位：回答 **单个算子 / 子图 / kernel 为什么快、为什么慢、如何定位和优化**。

## 建议学习顺序

1. 张量、Shape 与内存布局
2. Kernel 执行模型：SIMD / SIMT / warp / block
3. 内存层级与 Roofline
4. GEMM / Reduction / Pointwise 三类热点
5. 图融合与调度
6. 量化：FP16 / BF16 / INT8 / INT4
7. Attention Kernel 与 FlashAttention
8. 算子性能分析方法

## 存量内容映射

- [张量、shape 与内存布局](01-tensors-shapes-layout.md)
- [Kernel 与并行基础（SIMD/SIMT）](02-kernel-execution-model.md)
- [内存层级与性能模型（Roofline）](03-memory-hierarchy-and-roofline.md)
- [计算图、融合与调度](04-graph-fusion-scheduling.md)
- [量化基础（INT8/INT4）与误差](05-quantization-basics.md)
- [FlashAttention 与 IO-aware Attention](06-flashattention-io-aware.md)

## 已完成“实战型重构”的核心页

- [张量、shape 与内存布局](01-tensors-shapes-layout.md)
- [Kernel 与并行基础（SIMD/SIMT）](02-kernel-execution-model.md)
- [内存层级与性能模型（Roofline）](03-memory-hierarchy-and-roofline.md)
- [计算图、融合与调度](04-graph-fusion-scheduling.md)
- [量化基础（INT8/INT4）与误差](05-quantization-basics.md)
- [FlashAttention 与 IO-aware Attention](06-flashattention-io-aware.md)

## 后续补强建议

- 新增一篇：`04-gemm-reduction-pointwise.md`
- 新增一篇：`08-operator-profiling-playbook.md`
- 把所有章节统一成：**通用知识 → 例子 → 常见面试问题 → 易错点 → 排查 checklist**
