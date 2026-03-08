# 图编译：TVM / MLIR / XLA（概念对齐）

## 要点

- 图编译的目标：把“高层算子图”变成“更少、更快的 kernel/代码”，并适配目标硬件
- 你需要把三个概念分清：**IR 表达 → 优化 pass → 代码生成/调度**

## 共同概念（不绑定实现）

- IR：中间表示（算子图/循环/张量表达等）
- Pass：融合、常量折叠、dead code elimination、layout 变换
- Lowering：从高层算子逐步降到更接近硬件的表示
- Codegen：生成 CUDA/HIP/CPU 向量化代码等

## 工程落地关注点

- 动态 shape：需要 shape-specialization 或多版本 cache
- 自动调优：代价与收益（编译时间 vs 运行收益）
- 可调试性：IR dump、pass 可视化、回归定位

## 易错点

- 编译缓存策略不当导致线上抖动（冷启动慢）
- 过度 specialization 导致 cache 爆炸

## 排查 checklist

- [ ] 编译耗时与运行收益是否量化？
- [ ] 是否能稳定复现某个 shape 的最优路径？
- [ ] 是否有 IR/pass 级别的回归对比手段？
