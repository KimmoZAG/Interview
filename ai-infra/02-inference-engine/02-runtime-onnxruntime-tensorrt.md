# Runtime：ONNX Runtime / TensorRT

## 一句话先讲清

这一页回答的是：**模型图已经准备好了，接下来到底由谁把它稳定、高效地跑在目标硬件上？**

如果说 [`推理栈全景`](01-inference-stack-overview.md) 负责给你地图，Runtime 就是地图里真正负责“发车”的那一层；而 [`图编译`](03-graph-compiler-tvm-mlir-xla.md) 则更像是负责把路线修得更顺的那批工程队。

## 为什么值得单独学

- 很多“模型本身没变，但线上性能完全不一样”的差异，实际发生在 runtime 层。
- Runtime 决定了 execution provider、engine、memory planner、cache、stream 等执行细节。
- 选错 runtime，最常见的后果不是“完全跑不起来”，而是**能跑，但吞吐、延迟、冷启动、可维护性全面拧巴**。

## 关联知识网络

- 上游入口：[`推理栈全景`](01-inference-stack-overview.md)
- 下游配合：[`图编译：TVM / MLIR / XLA`](03-graph-compiler-tvm-mlir-xla.md)
- 线上落地：[`LLM Serving`](04-llm-serving.md)
- 诊断方法：[`推理优化 Playbook`](05-optimization-playbook.md)、[`可观测性与调试`](06-observability-and-debugging.md)
- 资源约束：[`Paged KV 与 Allocator`](07-paged-kv-and-allocator.md)
- 模型侧背景：[`Attention、KV cache 与吞吐/延迟`](../03-llm-architecture/02-attention-kv-cache.md)

## 它到底做什么

Runtime 负责把导出的模型图真正执行起来，常见职责包括：

- 加载图与权重
- 选择 execution provider / engine
- 调度 kernel 与 stream
- 处理动态 shape、memory plan 与 workspace
- 维护编译缓存、engine cache 或 plan 文件
- 暴露 profiling / tracing / fallback 信息

它比框架层更接近执行细节，但又比单个 kernel 高一个层级。你可以把它理解为：**把“模型图”翻译成“实际执行路径”的那一层总控**。

## ORT vs TRT：别只问“谁更快”

| 维度 | ONNX Runtime（ORT） | TensorRT（TRT） |
|---|---|---|
| 定位 | 通用型 runtime，provider 生态广 | NVIDIA GPU 上的高性能推理 engine |
| 优势 | 接入快、跨平台、兼容性更友好 | 固定/半固定 shape、FP16/INT8 场景下常有更高上限 |
| 风险 | 极致性能不一定追到最深 | 对算子覆盖、版本兼容、动态 shape 更敏感 |
| 动态 shape | 通常更稳，性能未必最优 | 能做但维护成本更高，engine 变体容易膨胀 |
| 部署维护 | 多后端、多平台更省心 | 需要更严格的版本、硬件、engine 生命周期管理 |

真正的工程问题通常不是“ORT 慢 / TRT 快”，而是：

1. 你的真实 shape 分布是什么？
2. 你的算子是否都能命中理想路径？
3. 你是否能接受 engine 预构建、cache 预热、版本锁定？

这三个问题没答好，离线 benchmark 再漂亮也容易线上翻车。

## 最小工程判断：什么时候更偏 ORT，什么时候更偏 TRT

### 更偏 ORT 的场景

- 模型迭代快，图结构或算子组合常变
- 动态 shape 很多，难以稳定预构建 engine
- 希望跨平台、多后端统一部署
- 团队更重视稳定接入和维护成本

### 更偏 TRT 的场景

- 目标硬件集中在 NVIDIA GPU
- shape 相对稳定，或可控在少量 profile 范围内
- FP16 / INT8 / 特定融合路径的性能收益明确
- 可以接受 plan/engine 的预构建与版本绑定

## 性能收益通常从哪里来

| 性能来源 | 具体动作 | 常见副作用 |
|---|---|---|
| 图级优化 | 常量折叠、融合、消除冗余 reshape/transpose | 调试路径更绕，fallback 更隐蔽 |
| 执行路径优化 | provider 选择、I/O binding、stream 利用 | 容易出现“理论能走，线上没走” |
| 低精度优化 | FP16 / INT8 / calibration | 精度回归、校准链路变复杂 |
| 缓存优化 | engine cache / timing cache / warmup | 冷启动、cache miss、版本兼容问题 |

## 一个最常见的线上坑

现象往往长这样：

- 离线 benchmark 很快
- 上线后 TTFT 很高
- 显存占用比预期大
- 某些请求一慢就慢得离谱

根因常常不是“runtime 理论性能不够”，而是下面几件事在捣乱：

- 动态 shape 触发了多个 engine / profile 变体
- 请求没命中已有 cache，现场重新构建
- 某些算子没覆盖，偷偷 fallback 到别的 provider
- 实际瓶颈在 serving queue / batching，而不是 runtime 本身

这也是为什么 runtime 问题常常要和 [`LLM Serving`](04-llm-serving.md) 一起看：**车很快，不代表路上不堵。**

## Troubleshooting：线上吞吐没有离线 benchmark 那么好，先怀疑什么

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 首次请求特别慢 | engine / plan 现场构建 | 看 cold start 日志、cache miss、build 耗时 |
| 某些输入长度突然很慢 | 动态 shape 没命中高性能 profile | 对照 shape 分布与 profile 配置 |
| GPU 利用率不低但吞吐不涨 | 上层 batching / queue 才是瓶颈 | 联合看 TTFT、batch size、queue wait |
| 整体 latency 抖动大 | provider fallback 或多路径执行 | 打开 provider / node 级 profiling |
| 显存异常升高 | 多 engine 变体、workspace 过大 | 对比 engine 数量、workspace、allocator 状态 |

### 一个排障顺序

1. 先确认**实际跑的是哪个 provider / engine**。
2. 再确认**哪些 shape 命中了理想路径，哪些没命中**。
3. 然后区分是 runtime 层问题，还是 serving / scheduler 层在拖后腿。
4. 最后再看有没有必要回到 [`图编译`](03-graph-compiler-tvm-mlir-xla.md) 层重做融合或 shape specialization。

## 推理优化工程师视角

这页最重要的不是背 ORT 和 TRT 的产品特性，而是建立**执行路径意识**：

- 模型最后到底落在哪个 provider / engine 上跑？
- 高性能路径覆盖的是不是你的真实流量，而不是 benchmark 样本？
- 纸面性能优势是否已经被冷启动、cache miss、fallback 吃掉？

很多“runtime 切换后收益不稳定”的问题，本质都不是算子变慢，而是**线上 workload 和你离线假设根本不是同一批东西**。

## 面试高频问法

### 初级

1. ONNX Runtime 和 TensorRT 的定位差别是什么？
2. 为什么 runtime 选型不能只看单次 benchmark？

### 中级

1. 为什么动态 shape 会让 TRT 类 engine 更难维护？
2. provider fallback 为什么会让性能突然塌掉？

### 高级

1. 如果线上 GPU 利用率不低，但吞吐不符合预期，你如何区分 runtime 问题和 serving 调度问题？
2. 某模型在固定 shape 下 TRT 很快，但线上收益不稳定，你会优先检查哪些证据？

## 易错点

- 把 runtime 选型理解成“换个后端就自动更快”
- 只看理想 shape，不看真实流量分布
- 忽略 provider fallback，误以为一直在高性能路径上
- 把冷启动 / cache miss 问题误判成 kernel 性能问题

## 排查 checklist

- [ ] 实际跑的是哪个 execution provider / engine？
- [ ] engine / cache 命中率是多少？
- [ ] 动态 shape 是否导致 profile 或 engine 变体爆炸？
- [ ] latency 问题真的发生在 runtime，而不是 queue / batching？
- [ ] profiling 里 kernel 数量是否异常偏多（说明融合没生效）？

## 参考资料

- ONNX Runtime 官方文档
- TensorRT 官方文档
- 各类 runtime / engine 调优实践
