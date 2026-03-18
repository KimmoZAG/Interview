# 图编译：TVM / MLIR / XLA

## 一句话先讲清

图编译回答的是：**怎么把“高层模型图”系统性地变成“更少、更快、更贴硬件”的执行计划或代码。**

你可以把 Runtime 理解为“真正跑起来的那一层”，把图编译理解为“决定能不能跑得更顺、更少绕路的那条优化流水线”。

## 为什么值得单独学

- 很多性能收益并不是改模型结构得到的，而是来自图层融合、layout 调整、lowering 与 codegen。
- 编译收益如果不能稳定命中真实流量，就只是离线 demo，不是系统能力。
- 一旦接触动态 shape、缓存、多版本 specialization，图编译就会直接影响冷启动、显存、线上波动。

## 关联知识网络

- 上游地图：[`推理栈全景`](01-inference-stack-overview.md)
- 相邻执行层：[`Runtime：ONNX Runtime / TensorRT`](02-runtime-onnxruntime-tensorrt.md)
- 线上落地：[`LLM Serving`](04-llm-serving.md)
- 诊断动作：[`推理优化 Playbook`](05-optimization-playbook.md)
- 证据体系：[`可观测性与调试`](06-observability-and-debugging.md)
- 模型侧成本背景：[`Transformer 推理所需的最小知识`](../03-llm-architecture/01-transformer-minimum.md)、[`Attention、KV cache 与吞吐/延迟`](../03-llm-architecture/02-attention-kv-cache.md)

## 图编译到底在干什么

图编译通常分成三步：

1. **IR 表达**：先把高层模型表示成适合机器处理的中间表示。
2. **优化 Pass**：在 IR 上做融合、常量折叠、layout 变换、死代码消除等。
3. **Lowering / Codegen**：把高层表示降到更接近硬件的循环、指令或执行计划。

真正的价值不在于术语多，而在于它给了你一条**从模型图到硬件执行之间可控、可调、可对比**的优化链路。

## 共同概念，一次分清

| 概念 | 它解决什么问题 | 你应该观察什么 |
|---|---|---|
| IR | 用统一形式表达图、张量或循环结构 | 信息是否足够支持后续优化 |
| Pass | 系统性变换图结构或数据布局 | 是否真的减少 kernel / 访存 / 中间张量 |
| Lowering | 从高层算子逐步降到低层表示 | 关键语义有没有在降级中丢失 |
| Codegen | 为目标硬件生成执行代码或计划 | 最终是否命中目标后端的高性能路径 |

## 最小例子：为什么 `matmul -> bias add -> gelu` 值得编译

如果没有优化，这一段可能拆成多个独立 kernel：

- 一个做 matmul
- 一个做 bias add
- 一个做 gelu

这样会带来：

- 更多 kernel launch
- 更多中间张量写回与读回
- 更多不必要的显存带宽浪费

如果图编译器完成融合，可能把后两步甚至更多逻辑合到更少的执行单元里，从而：

- 减少 launch 次数
- 减少中间结果落地
- 提高数据复用

这也是图编译最常见的“体感收益来源”。

## TVM / MLIR / XLA 应该怎么记

不要急着背产品名，先按“抽象层级”记：

- **TVM**：更强调张量表达、调度与面向后端的优化落地
- **MLIR**：更强调多层 IR 组织和可组合编译基础设施
- **XLA**：更强调面向特定执行栈的图优化与 lowering 路线

对于面试和工程判断，更重要的是知道它们都在解决类似问题：

- 如何表达图
- 如何组织 pass
- 如何做 shape / layout / hardware-aware 优化
- 如何把收益稳定交付到运行时

## 工程落地时最该看的三张账

| 账本 | 典型问题 | 常见误区 |
|---|---|---|
| 编译时间账 | 编译耗时是否可接受？冷启动是否会爆？ | 只看运行时收益，不看编译代价 |
| 缓存命中账 | shape specialization 是否能覆盖真实流量？ | benchmark 命中最优路径，线上却频繁 miss |
| 运行时收益账 | kernel 数、latency、throughput 是否稳定改善？ | 只看单 shape 的最佳结果 |

这三张账必须一起看。不然非常容易出现“离线测得像神，线上表现像谜”的情况。

## 动态 shape 为什么是高发区

一个常见线上现象：

- 某个 shape 非常快
- shape 稍微一变，性能突然回退

这通常意味着：

- 当前最优路径依赖 shape specialization
- 编译缓存只覆盖了部分 shape
- 某些 pass 在动态 shape 下无法稳定生效
- runtime 层被迫切到更保守的执行路径

所以图编译问题常常不是“编译器不够聪明”，而是**真实流量比分布假设复杂得多**。

## Troubleshooting：编译收益很好看，但线上不稳定怎么办

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 某些输入特别快，另一些突然变慢 | shape specialization 覆盖不全 | 统计线上 shape 分布与 cache 命中率 |
| 冷启动很慢 | 编译/加载成本被挪到了请求路径上 | 看首次请求编译时间、warmup 是否到位 |
| kernel 数没降多少 | pass 没命中或底层没真正融合 | 对比 IR dump、profile 中 kernel 数 |
| 吞吐提升不稳定 | 编译收益没有稳定落到 runtime | 联合 runtime profiling 与 serving 指标 |
| 显存或 cache 压力上升 | 过度 specialization、缓存变体过多 | 观察 cache 规模、workspace、allocator 状态 |

### 一个实用排障顺序

1. 先看**收益是否只存在于某些特定 shape**。
2. 再看**IR / pass 层到底发生了什么变化**。
3. 然后确认**底层 kernel 数和访存模式是否真的变好了**。
4. 最后把编译收益放回 runtime / serving 场景里验证，防止“实验室胜利”。

## 推理优化工程师视角

读图编译的重点，不是为了当编译器作者，而是为了学会把性能收益拆开看：

- 是 kernel 数变少了？
- 是 layout 更顺了？
- 是某些 shape 被专门优化了？
- 还是只是把成本从运行时挪到了编译时和冷启动？

当你能把这些账说清楚，图编译就不再是“黑箱加速器”，而是系统设计里一个可验证的决策点。

## 面试高频问法

### 初级

1. IR、pass、lowering、codegen 分别是什么？
2. 为什么图编译常常能减少 kernel 数？

### 中级

1. 为什么动态 shape 会让图编译和缓存策略变复杂？
2. 图级融合和 kernel 级融合有什么联系与区别？

### 高级

1. 如果编译后某些 shape 很快、另一些很慢，你会从哪里开始定位？
2. 为什么图编译收益不能只看单次运行时间，还必须看编译成本和冷启动？

## 易错点

- 把“编译成功”误当成“收益已经稳定上线”
- 过度 specialization 导致 cache 爆炸
- 只看高层 IR，不验证底层 kernel 是否真的减少
- 只看吞吐，不看编译时延和可维护性成本

## 排查 checklist

- [ ] 编译耗时与运行收益是否被同时量化？
- [ ] 是否能稳定复现某个 shape 的最优路径？
- [ ] 是否有 IR / pass 级别的前后对比手段？
- [ ] 底层 kernel 数量、访存模式是否真的改善？
- [ ] 编译收益是否在真实 serving 流量中仍然成立？

## 参考资料

- TVM / MLIR / XLA 官方资料
- 图编译、lowering 与 codegen 相关实践文档
- 动态 shape 与编译缓存优化资料
