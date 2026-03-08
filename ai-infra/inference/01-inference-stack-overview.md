# 推理栈全景：前端→图→kernel→执行

## 要点

- 把推理系统拆成 4 层更容易定位问题：**模型前端 → 中间表示(IR/图) → 编译/优化 → 执行时(runtime)**
- 性能与稳定性问题，往往发生在“层与层交界处”（shape/dtype/layout/动态批处理）

## 典型数据流

1. 前端（PyTorch/TF/JAX/自研）：定义模型与权重
2. 导出/表示：ONNX / TorchScript / StableHLO 等
3. 优化与编译：图优化、算子选择、融合、量化、代码生成
4. Runtime：内存管理、kernel 调度、stream 同步、并发与 batching

## 你需要能画出来的“最小架构图”（建议你后续补）

- 请求进入 → 预处理(tokenize) → prefill → decode 循环 → 后处理
- 动态 batching 在哪里做？cache 在哪里存？

## 易错点

- 只在模型层看问题，忽略 runtime 的同步与内存抖动
- 只看单次推理，忽略并发与排队造成的尾延迟

## 排查 checklist

- [ ] 把耗时拆解到：预处理 / prefill / decode / 后处理
- [ ] 确认“图优化是否生效”（kernel 数量是否下降）
- [ ] 确认“shape/dtype/layout 是否符合预期”
