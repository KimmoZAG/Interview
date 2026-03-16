# Runtime：ONNX Runtime / TensorRT（要点清单）

## 要点

- ONNX Runtime（ORT）：偏通用 runtime，生态好、易集成
- TensorRT（TRT）：偏 NVIDIA GPU 极致优化，常见于静态/半静态图与 FP16/INT8 场景
- 选 runtime 不只是看“谁更快”，而是看：**图是否稳定、算子是否覆盖、动态 shape 是否友好、部署链路是否能维护**。

## 通用知识

### 它是什么

runtime 负责把导出的模型图真正执行起来。它通常要处理：

- 图加载
- execution provider / engine 选择
- kernel 调度
- 内存管理
- 编译缓存或 engine 复用

### 它解决什么问题

runtime 解决的是“模型如何稳定、高效地跑在目标硬件上”。

它比框架层更贴近执行细节，但又比单个 kernel 更高一层。

### 为什么在 AI 系统里重要

因为同一个模型，换一个 runtime 后，可能会在这些地方表现不同：

- kernel 融合程度
- 动态 shape 性能波动
- 量化支持能力
- 冷启动与编译缓存表现

### 它的收益与代价

ORT 的常见收益：

- 通用性更强
- provider 生态丰富
- 更适合多平台与多后端部署

TRT 的常见收益：

- 在 NVIDIA GPU 上更容易拿到极致性能
- FP16 / INT8 / engine 优化更成熟

代价则往往是：

- TRT 对动态 shape、算子覆盖和版本兼容更敏感
- ORT 在极致性能上未必总能追平深度定制 engine

## 选型维度（工程视角）

- 支持的算子覆盖与 fallback 路径
- 动态 shape 支持与性能波动
- 量化与校准工具链
- 部署形态：C++/Python、跨平台要求、版本锁定成本

## 最小例子

假设同一个模型有两种部署方案：

- 用 ORT 直接跑 ONNX，优点是接入快、兼容性强
- 用 TRT 构建 engine，优点是在固定 shape 和低精度下可能更快

如果你的线上 workload：

- shape 非常稳定
- 都跑在 NVIDIA GPU
- 可以接受 engine 预构建

那么 TRT 常更有吸引力。

如果你的 workload：

- 动态 shape 多
- 算子变化频繁
- 平台更复杂

那么 ORT 往往更稳。

## 常见性能手段

- 图级：常量折叠、融合、消除冗余 reshape/transpose
- runtime：内存池、I/O binding、异步拷贝与 stream
- 编译缓存：engine/cache 的构建与复用

## 工程例子

一个常见坑是：

- 离线 benchmark 很快
- 一上线上，冷启动很慢，显存也更高

根因经常是：

- 动态 shape 触发了多个 engine / cache 变体
- 某些请求 shape 没命中已有编译缓存
- fallback 到非预期 provider，导致局部 CPU/GPU 混跑

所以 runtime 问题经常不是“理论性能不够”，而是“真实 workload 没走你以为的那条执行路径”。

## 推理优化工程师视角

从推理优化工程师的角度，这一章最重要的不是记住 ORT 和 TRT 的产品差异，而是建立一个执行路径意识：

- 模型最后到底落在哪个 provider / engine 上跑
- 哪些 shape 命中了高性能路径，哪些请求掉回了普通路径
- 冷启动、cache 命中率、fallback 是否已经吞掉了纸面性能优势

很多“runtime 切换后没有预期收益”的问题，本质不是算子变慢，而是线上流量分布、动态 shape 和 engine 生命周期管理没有和 benchmark 假设对齐。

所以做 runtime 选型时，最好先问：

1. 我追求的是峰值性能，还是稳定可维护的性能？
2. 我的真实 shape 分布是否适合这条执行链？
3. 我是否能持续观测 provider / engine / cache 的命中情况？

能把这几件事量化清楚，runtime 选型才是工程决策，而不是跑分崇拜。

## 常见面试问题

### 初级

1. ONNX Runtime 和 TensorRT 的定位差别是什么？
2. 为什么 runtime 选型不能只看单次 benchmark？

### 中级

1. 为什么动态 shape 常让 TRT 类 engine 更难维护？
2. fallback path 为什么会让性能突然变差？

### 高级

1. 如果线上 GPU 利用率不低，但整体吞吐不符合预期，你会如何判断是 runtime 选型问题还是上层调度问题？
2. 如果某个模型在离线固定 shape 下 TRT 很快，但线上收益不稳定，你最先怀疑什么？

## 易错点

- 动态 shape 导致生成多个 engine/cache，显存与构建时间暴涨
- fallback 到非预期 provider（CPU/GPU 混跑）
- 只看理想 shape，不看真实流量分布
- 把 runtime 性能问题和上层 serving 调度问题混为一谈

## 排查 checklist

- [ ] 实际跑的是哪个 execution provider / engine？
- [ ] engine/cache 命中率是多少？
- [ ] kernel 数量是否异常偏多（融合没生效）？

## 参考资料

- ONNX Runtime 官方文档
- TensorRT 官方文档
- 相关 runtime / engine 调优资料
