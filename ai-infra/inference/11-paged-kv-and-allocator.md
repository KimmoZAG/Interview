# Paged KV 与 Allocator：为什么 KV cache 不是“开个数组”这么简单

## 要点

- 从 CS336 Lecture 10 的工程落点看，KV cache 不只是一个模型概念，而是推理系统最关键的状态对象之一。
- 一旦进入真实 serving，问题立刻从“KV 要不要存”变成“怎么分配、怎么复用、怎么回收、怎么避免碎片”。
- paged KV 的核心价值不是概念新奇，而是让长短不一、动态到达的请求可以更稳定地管理 KV 状态。
- allocator 设计会直接影响：显存碎片、扩容成本、尾延迟、请求驱逐和系统稳定性。

## 1. 为什么连续 KV 布局会出问题

最直观的做法是：

- 为每个请求分配一段连续显存
- 把每一层的 K/V 依次写进去

这在静态 batch 或短请求场景下很好理解，但真实 serving 很快会出现问题：

- 不同请求长度差异很大
- 请求到达和结束时间不一致
- 有的请求生成很短，有的请求生成很长

于是连续布局容易带来：

- 扩容拷贝
- 显存空洞
- 碎片累积

## 2. Paged KV 在解决什么

paged KV 的思路是：

- 不要求每个请求独占一整段连续大块显存
- 而是把 KV 按 page 或 block 切成固定粒度的小块
- 请求只持有这些块的映射关系

这样做的好处是：

- 更容易增长和回收
- 减少大块连续显存需求
- 更适合 continuous batching 和动态请求生命周期

## 3. 为什么 allocator 会成为关键组件

一旦用 page/block 管理 KV，系统就必须回答：

- 新请求来了，从哪里拿 block
- 请求结束后，哪些 block 可以回收
- block 不够时，驱逐谁
- 如何避免频繁分配释放造成抖动

这就从“模型缓存”问题升级成了“内存分配器”问题。

## 4. 一个最小心智模型

可以把 Paged KV 系统想成三层：

### 层 1：逻辑 token 序列

- 用户看到的是一串上下文 token

### 层 2：KV pages / blocks

- 系统把这些 token 对应的 KV 切成固定大小块

### 层 3：映射和调度

- 每个请求维护逻辑位置到物理 block 的映射
- 调度器和 kernel 根据映射读取所需 KV

## 5. Paged KV 的主要收益

### 收益 1：更好的内存复用

- 请求结束后可以只回收对应 block

### 收益 2：更少的扩容拷贝

- 序列变长时不必整体搬迁已有 KV

### 收益 3：更适合动态 batching

- 不同请求可以独立增长，不必预先按最大长度保留大块空间

## 6. 它的代价是什么

Paged KV 不是免费午餐，它通常带来：

- 地址映射开销
- 更复杂的 kernel 读取逻辑
- block 粒度带来的内部碎片
- allocator 元数据与调度复杂度

所以工程上不能只问“支不支持 paged KV”，还要问：

- block 大小是否合理
- 映射访问是否破坏了访存局部性
- allocator 是否成为新的延迟来源

## 7. 什么时候 allocator 会伤你最深

最容易出问题的场景包括：

- 长短请求混跑
- 高频突发流量
- 多模型或多租户共享 GPU
- 长上下文请求占满大量 KV 页

在这些场景下，allocator 设计不好会直接表现为：

- p99 抖动
- 偶发 OOM
- throughput 不稳定

## 8. 应该观测哪些指标

除了常规的 TTFT、TPOT、吞吐，还应特别盯：

- 活跃 KV 页数
- block 利用率
- 分配/释放频率
- 请求驱逐率
- 显存碎片与保留内存

如果不观测这些，你就只能看到“系统偶尔抖”，但无法定位是调度问题还是内存管理问题。

## 9. 和长上下文有什么关系

上下文越长：

- KV cache 越大
- allocator 压力越高
- page/block 数量越多

因此长上下文推理几乎一定会把 KV 管理和内存分配推到系统核心位置。

## 易错点

- 把 KV cache 只当成 attention 的附属缓存
- 只比较总显存占用，不看碎片和分配行为
- 只知道 paged KV 更灵活，却说不清它增加了哪些映射和调度成本
- 不做分桶分析，就把所有线上抖动归因给模型本身

## 排查 checklist

- [ ] 当前 KV 布局是连续、分页，还是混合方案？
- [ ] block 大小是否和请求长度分布匹配？
- [ ] 是否记录了 allocator 相关指标，而不只是总显存？
- [ ] p99 抖动时，究竟是 decode、分配器，还是驱逐策略在出问题？

## CS336 对照

- 官方 lecture 对应：Lecture 10（inference）
- 推荐搭配阅读：
  - [../inference/04-llm-serving.md](../inference/04-llm-serving.md)
  - [../models/02-attention-kv-cache.md](../models/02-attention-kv-cache.md)
  - [../inference/06-observability-and-debugging.md](../inference/06-observability-and-debugging.md)