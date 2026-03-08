# 计算图、融合与调度

## 要点

- 推理加速常见路径：**图级优化（融合/常量折叠）+ kernel 级优化（高效实现）**
- 融合的收益主要来自：减少中间张量写回、减少 kernel launch、提高缓存复用

## 计算图（抽象）

- Op graph：节点是算子，边是张量
- 关键属性：shape/dtype/layout、是否动态 shape、是否可重排

## 融合的常见形态

- pointwise 链：`bias + gelu + dropout`（推理里 dropout 通常关）
- Norm + scale/bias
- attention 子图（QKV 投影、softmax、matmul）中的部分融合

## 调度（scheduling）关注点

- tile/blocking（提高数据复用）
- 并行策略（线程块映射）
- 内存分配与重用（buffer reuse）

## 易错点

- 融合后数值误差变化（尤其低精度）
- 动态 shape 导致某些优化失效或需要多个编译 cache

## 排查 checklist

- [ ] 图中是否存在大量小 op（pointwise）导致 launch 爆炸？
- [ ] 是否能接受对部分子图做 ahead-of-time 编译缓存？
- [ ] 融合前后是否对比了中间张量的最大误差？
