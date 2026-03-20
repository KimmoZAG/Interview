---
tags:
  - AI Infra
  - LMCache
  - vLLM
  - SGLang
  - Serving Engine
description: 解释 LMCache 如何把 KV 复用内核与 serving engine 生命周期钩子解耦，并分别适配 vLLM 与 SGLang。
---

# 第 7 章：与 vLLM / SGLang 的集成与解耦设计

配套入口：

- [README.md](README.md)
- [00-index.md](00-index.md)
- [01-overall-architecture-and-core-abstractions.md](01-overall-architecture-and-core-abstractions.md)
- [04-gpu-connector-and-async-store-load.md](04-gpu-connector-and-async-store-load.md)
- [06-disaggregated-prefill-and-control-plane.md](06-disaggregated-prefill-and-control-plane.md)

前面几章你已经知道，LMCache 的内核主要是三层：

- `LMCacheEngine`
- `TokenDatabase`
- `StorageManager`

再往下是 GPU connector、lookup client、offload server、controller 等围绕内核展开的能力。

但如果你站在系统设计面试官的视角，会马上追问一句：

**既然每个 serving engine 的 KV layout、调度生命周期、block/page 管理都不一样，LMCache 为什么还能同时接 vLLM 和 SGLang？**

这一章要回答的就是这个问题。

先给结论：

- LMCache 并没有假装不同引擎完全一样；
- 它真正抽象出来的是 **KV 复用内核**；
- 引擎特有的生命周期钩子、metadata 生成、GPU layout 访问方式，放在 integration 层做适配；
- 因此它不是“零耦合”，而是 **把耦合压缩到最小必要面**。

这就是 LMCache 集成设计里最有工程价值的地方。

## 技术背景

### 1. 为什么 KV cache 系统一定会和 serving engine 发生耦合

很多人第一次看 LMCache，会希望它像 Redis client 一样，给所有引擎一个统一的 `put/get` 接口就完了。

现实不是这样。KV cache 这件事天然深度侵入 serving engine，原因很直接：

- KV 在 GPU 上的物理布局由引擎决定；
- scheduler 知道哪些 token 已算、哪些 block 已分配；
- attention 层知道何时可以 layerwise load/save；
- request 生命周期由引擎管理，不由 LMCache 管理。

所以一个 KV cache 系统如果真想高效，不可能完全脱离引擎内部语义。

### 2. 但如果完全写死到单一引擎里，代价会非常大

另一种极端是把所有逻辑都塞进 vLLM 或某个 engine 的内部实现里。

这样做短期看集成简单，长期问题很大：

- 迁移到别的 engine 基本等于重写；
- 引擎升级时容易被内部 API 变化拖垮；
- 存储、P2P、controller、offload 这些共性能力被困在单一 runtime 内。

从平台化视角看，这是很差的资产沉淀方式。

### 3. 所以 LMCache 的目标不是消灭耦合，而是控制耦合的位置

这正是 integration 层的意义。

LMCache 的做法可以概括成两句话：

1. **把可复用的核心能力收敛到 engine-agnostic 内核里。**
2. **把不可避免的引擎差异，限制在 adapter / service factory / GPU connector 这些边界层。**

这个思路很像数据库驱动或存储引擎接口：

- 核心事务语义相对稳定；
- 每个 backend 再去补自己的连接、编码、执行细节。

### 4. 面试里真正该问的不是“能不能集成多引擎”，而是“最小适配面在哪里”

一个系统声称支持多个 runtime，不值钱。

真正值钱的是它能说清：

- 哪些模块是共用的；
- 哪些模块必须按引擎重写；
- 如果引擎升级，受影响面有多大；
- scheduler、worker、attention 层分别在什么点接入。

LMCache 在这方面其实做得比较清楚，这也是这一章最应该学的部分。

## 技术核心（结合代码）

### 1. 先看总边界：`LMCacheManager + BaseServiceFactory` 负责把内核和引擎装配起来

如果你只看第 4 章的 GPU 路径，很容易以为 LMCache 主要是一个 engine object。

但从 `lmcache/v1/manager.py` 和 `lmcache/integration/base_service_factory.py` 看，真正的总装配逻辑是：

- `LMCacheManager` 负责生命周期管理；
- `BaseServiceFactory` 负责“这个引擎/这个角色该创建哪些组件”；
- manager 不直接写死 vLLM 或 SGLang 逻辑。

`BaseServiceFactory` 定义的抽象方法很有代表性：

- `get_or_create_metadata()`
- `get_or_create_lmcache_engine()`
- `maybe_create_lookup_client()`
- `maybe_create_lookup_server()`
- `maybe_create_offload_server()`
- `maybe_create_internal_api_server()`
- `maybe_create_health_monitor()`

这里的关键设计点是：**manager 只管理组件生命周期，不关心这些组件是怎么按引擎创建出来的。**

这一步把“系统编排逻辑”和“引擎特定创建逻辑”拆开了。

### 2. `LMCacheManager` 本身是引擎无关的

`LMCacheManager` 构造时只拿两样东西：

- `LMCacheEngineConfig`
- `BaseServiceFactory`

然后它按统一顺序初始化：

- metadata
- engine
- lookup client/server
- offload server
- runtime plugin launcher
- internal API server

再在 `post_init()` 阶段做 engine ready 之后的收尾，例如：

- `engine.post_init(...)`
- 异步 lookup server 接线
- health monitor 初始化

这里最重要的工程意义是：**引擎集成层不需要重复实现整套服务生命周期。**

也就是说，vLLM 和以后别的 runtime 真正要负责的是“怎么创建组件”，而不是“怎么重写整套 LMCache 管理逻辑”。

### 3. vLLM 为什么需要最重的一层适配

vLLM 这边的入口主要在：

- `integration/vllm/lmcache_connector_v1.py`
- `integration/vllm/vllm_v1_adapter.py`
- `integration/vllm/vllm_service_factory.py`
- `integration/vllm/vllm_multi_process_adapter.py`
- `integration/vllm/utils.py`

它之所以最复杂，不是因为代码写得啰嗦，而是因为 vLLM 本身提供了非常明确的 KV connector 生命周期，LMCache 需要精确嵌进去。

这套生命周期同时横跨：

- scheduler 侧 lookup；
- worker 侧 load/save；
- attention layer 内的 layerwise wait/load；
- request finish 和 block 回收前的异步传输收尾。

### 4. `LMCacheConnectorV1Dynamic`：对外暴露的是 vLLM 期望的 connector 形状

`integration/vllm/lmcache_connector_v1.py` 里的 `LMCacheConnectorV1Dynamic` 直接继承 vLLM 的 `KVConnectorBase_V1`。

这说明 LMCache 在 vLLM 里不是外挂脚本，而是明确接入了官方 connector 协议。

这个类本身很薄，它主要做一件事：

- 把 vLLM 调进来的标准接口，转发给 `LMCacheConnectorV1Impl`。

对应的方法包括：

- `register_kv_caches()`
- `start_load_kv()`
- `wait_for_layer_load()`
- `save_kv_layer()`
- `wait_for_save()`
- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `build_connector_meta()`
- `request_finished()`

从设计上看，这是一层典型的 **协议适配器**。

它的价值在于：

- 上游保持 vLLM 原生接口；
- 下游保留 LMCache 自己的内部实现空间；
- 当 vLLM connector API 变化时，修改面优先收敛在这层和 adapter 层，而不是污染整个内核。

### 5. 真正复杂的逻辑在 `LMCacheConnectorV1Impl`

`integration/vllm/vllm_v1_adapter.py` 才是 vLLM 集成的核心。

这个文件里你会看到几个非常重要的数据结构：

- `LoadSpec`
- `SaveSpec`
- `DisaggSpec`
- `RequestTracker`
- `ReqMeta`

这些结构说明了一个事实：

**vLLM 集成的难点，不在于“调用一次 store/retrieve”，而在于把 request 调度状态翻译成 LMCache 能理解的 load/save/transfer 语义。**

比如 `RequestTracker` 会维护：

- 当前 request 的 token 序列；
- block 分配状态；
- 已经保存到哪里；
- 是否进入 decode phase；
- multimodal hash 和位置；
- request 级别配置；
- 是否绑定了 disagg 传输信息。

这其实就是 adapter 层在做的关键工作：**把 engine 内部 request 状态机，映射成 LMCache 的缓存生命周期状态机。**

### 6. 为什么 `RequestTracker` 这种对象不能下沉到内核层

一个常见误判是：既然 `RequestTracker` 很重要，为什么不直接做成 `LMCacheEngine` 的一部分？

答案是不能，因为这里面大量字段都属于 vLLM 语义：

- scheduler 分配的 block ids；
- 新请求与 running request 的调度行为；
- preempt 后 token/block 恢复逻辑；
- sampling_params 里的 extra args；
- multimodal placeholder 的抽取方式。

这些都不是 LMCache 内核通用语义。

如果把它们下沉到内核层，LMCache 会迅速被 vLLM 内部机制绑死。

所以这块放在 adapter 里，恰恰说明分层是清醒的。

### 7. `VllmServiceFactory`：同一套引擎，不同角色创建的组件并不相同

`integration/vllm/vllm_service_factory.py` 很值得细看，因为它揭示了 vLLM 集成不是单一进程视角，而是 **按角色装配**。

`role` 至少区分：

- `scheduler`
- `worker`

不同角色创建的东西不同：

- scheduler 侧通常创建 lookup client；
- worker 侧创建 engine、lookup server、offload server；
- DP rank 0 还会创建 internal API server 和 runtime plugin launcher；
- health monitor 则在统一 post-init 之后创建。

这件事背后的系统含义是：**LMCache 适配的不是一个“抽象引擎对象”，而是一个多角色 serving runtime。**

这和真实线上部署更接近，也解释了为什么 service factory 比简单 adapter 更重要。

### 8. metadata 生成为什么必须放在引擎适配层

`VllmServiceFactory.get_or_create_metadata()` 做了很多看起来“偏底层”的事情：

- 从 `model_config` / `parallel_config` / `cache_config` 推导 KV dtype；
- 计算 `num_layer`、`num_kv_head`、`head_size`；
- 处理 MLA、draft layers；
- 计算 worker/local_worker/world_size；
- 带上 role、served_model_name、engine_id、extra config。

这说明 metadata 不是纯粹的 LMCache 配置，而是 **引擎拓扑 + 模型配置 + cache 布局** 的交汇点。

因此 metadata 不能凭空在内核里推导，它必须由 integration 层来构造。

### 9. `CreateGPUConnector(...)` 说明 GPU layout 访问方式也是引擎相关的

在 `VllmServiceFactory.get_or_create_lmcache_engine()` 和 `integration/sglang/sglang_adapter.py` 里，你都会看到：

- 先构造 metadata；
- 再调用 `CreateGPUConnector(config, metadata, EngineType.XXX)`；
- 最后把 connector 注入 `LMCacheEngineBuilder.get_or_create(...)`。

这说明 LMCache 的一个关键抽象是：

**内核不直接假设 GPU 上 KV 怎么排布，而是通过 engine-specific GPU connector 读写。**

所以你可以把整个集成层理解成三种不同粒度的适配：

1. 协议级适配：connector API。
2. 生命周期适配：scheduler/worker/request hooks。
3. 存储布局适配：GPU connector。

这三层一起，才让“同一个 LMCache 内核接不同 engine”变得可行。

### 10. vLLM 的 scheduler bypass lookup 体现了集成层和内核之间的性能协作

`VllmServiceFactory.get_or_create_lmcache_engine()` 里有一段很有意思的逻辑：

- 如果是 scheduler 且没有 `enable_scheduler_bypass_lookup`，甚至不一定需要 engine；
- 如果开启 bypass lookup，才要求 engine 满足更强约束，例如 `save_only_first_rank` 等条件。

这说明 integration 层不是单纯“翻译接口”，它还在参与性能路径上的策略选择。

也就是说，适配层除了负责接线，还负责回答：

- lookup 是在 scheduler 侧做还是 worker 侧做；
- 哪些配置组合是安全可行的；
- 某些优化是否要求特殊的 rank 语义。

这正是成熟 integration 层该承担的职责。

### 11. `vllm_multi_process_adapter.py` 说明 LMCache 连“部署形态变化”也做了专门隔离

这一层很容易被忽略，但其实很关键。

`integration/vllm/vllm_multi_process_adapter.py` 处理的是 multiprocess 模式下：

- 通过消息队列跟 LMCache server 交互；
- 懒启动 heartbeat；
- 维护健康状态；
- 两阶段 lookup/prefetch 结果；
- 不健康时进入 degraded mode。

这说明 LMCache 不仅在抽象“不同 engine”，还在抽象 **同一 engine 的不同部署模式**。

从工程角度讲，这非常值钱，因为它避免了把：

- 单进程 connector 逻辑；
- 多进程 server/client 通信逻辑；
- 健康检查与降级策略；

全部混到一份 adapter 代码里。

### 12. SGLang 集成明显更薄，这恰恰说明 LMCache 没有强迫所有引擎接受同一种接法

SGLang 入口主要在：

- `integration/sglang/sglang_adapter.py`
- `integration/sglang/utils.py`

跟 vLLM 相比，它没有一整套 service factory + connector protocol 的复杂层次，而是更直接：

- 用 `init_lmcache_engine(...)` 构造 metadata 和 engine；
- `LMCacheConnector` 直接暴露 `load_kv()` / `store_kv()` / `get_kv_events()` / `reset()`；
- `LMCacheLayerwiseConnector` 在此基础上再做 layerwise retrieve/store。

这背后不是“功能更弱”，而是接入点不同。

SGLang 这边更像是：

- 提供已有的 KV pool；
- LMCache 直接基于这些 tensor 做 load/store；
- 少一些 scheduler 协议抽象，多一些直接操作 GPU cache 的逻辑。

这再次说明 LMCache 的目标不是统一所有引擎的表面 API，而是统一底层复用内核。

### 13. `LMCacheConnector` 和 `LMCacheLayerwiseConnector` 说明 SGLang 适配更偏“直接调用内核”

在 `sglang_adapter.py` 里你会看到非常直接的调用：

- `lmcache_engine.retrieve(...)`
- `lmcache_engine.store(...)`
- `lmcache_engine.lookup(...)`
- `lmcache_engine.retrieve_layer(...)`
- `lmcache_engine.store_layer(...)`
- `lmcache_engine.lookup_unpin(...)`

这说明 SGLang 适配的核心工作更偏向：

- 把 token ids、slot mapping、kv_indices 整理好；
- 选择普通路径还是 layerwise 路径；
- 在 TP 组里对可加载 token 数做 `global_min_tokens` 对齐；
- 管理 lookup pin/unpin 生命周期。

从分层角度看，这是一种更薄、更直接的 integration 方式。

### 14. 为什么 vLLM 和 SGLang 的适配层厚度不同，反而说明架构是健康的

很多人看到两个引擎的适配写法不一样，会本能觉得“不统一”。

但从架构角度，这恰恰是健康信号。

原因很简单：

- vLLM 暴露了更成熟的 connector 生命周期，所以 LMCache 适配就可以更正式、分层更多；
- SGLang 的接入点更直接，因此适配可以更薄；
- 如果硬要把两者揉成完全一样的代码形状，通常只会引入额外抽象噪音。

真正应该统一的，不是表层代码长相，而是：

- `LMCacheEngine` 的语义；
- key/metadata 的正确性；
- GPU connector 的边界；
- lookup/store/retrieve 的结果语义。

LMCache 在这点上做得是务实统一，不是形式统一。

### 15. 这一整套设计最终把“适配新引擎”的问题压缩成几个固定问题

如果未来要接一个新的 serving engine，你真正要回答的核心问题其实已经被 LMCache 框住了：

1. 这个引擎的 KV cache 在 GPU 上怎么访问。
2. scheduler 和 worker 生命周期里，哪些时机适合 lookup/load/save。
3. 如何提取稳定的 request token 序列、slot mapping、block/page 信息。
4. metadata 里哪些字段能唯一标识这套 KV 的拓扑和布局。
5. 需要哪些额外服务组件：lookup client/server、offload server、API server、健康监控。

这就是这套集成架构最大的价值：

**它没有消除新引擎适配成本，但把成本变成了清晰、边界明确、可拆分的工程任务。**

### 16. 把第 7 章压成一条总数据流

从“引擎集成”视角，可以把 LMCache 的工作流压成下面这条线：

```text
Serving Engine 暴露自身生命周期钩子
  -> integration 层提取 request / block / slot / KV tensor 语义
  -> service factory 构造 metadata / engine / lookup / offload 等组件
  -> GPU connector 屏蔽具体引擎的 KV 物理布局
  -> LMCacheEngine 执行统一的 lookup / retrieve / store / layerwise pipeline
  -> manager 负责服务生命周期、健康检查和降级
```

这条线里最重要的认识是：

- **integration 层负责翻译引擎语义**；
- **LMCache 内核负责执行统一缓存语义**。

这就是它能同时支持 vLLM 和 SGLang 的根本原因。

## 面试可能问到的问题

### 问题 1：为什么 LMCache 不能只提供一个统一 `put/get` API，让所有引擎自己适配？

**满分回答思路：**

因为 KV cache 不只是对象存取问题，还涉及：

- request 生命周期；
- block/page/slot 映射；
- GPU layout；
- layerwise load/save 时机；
- scheduler 可见的局部命中信息。

如果 LMCache 只提供一个极薄的 `put/get` API，绝大多数复杂性就会泄漏到各引擎接入代码里，导致：

- 每个引擎重复写一遍生命周期管理；
- 共性的控制面、存储层、健康检查无法复用；
- 很难保证行为一致。

所以更合理的做法是像 LMCache 这样：内核统一缓存语义，integration 层统一接入框架，但保留必要的引擎特定适配。

### 问题 2：为什么 `RequestTracker` 这类 request 状态对象应该留在 vLLM adapter，而不是下沉到 `LMCacheEngine`？

**满分回答思路：**

因为它承载的是 vLLM 的调度语义，不是 LMCache 的通用缓存语义。

例如：

- block ids 的分配和变更；
- preemption 后如何恢复 token/block 对齐；
- sampling params 里的 request extra args；
- multimodal placeholder 的抽取方式。

这些都依赖 vLLM 的 request 模型。如果下沉到内核层，会让 LMCache 核心逻辑直接依赖 vLLM 内部结构，失去多引擎扩展能力。

### 问题 3：如何判断一个缓存系统是真正“支持多引擎”，还是只是写了几份不同的接入胶水？

**满分回答思路：**

关键看三点：

- 核心缓存语义是不是共用一套内核；
- 引擎差异是不是被约束在清晰边界内；
- 新引擎接入时，受影响面是不是局部可控。

LMCache 的答案比较清楚：

- `LMCacheEngine`、`StorageManager`、`TokenDatabase`、controller、backend 等是共用的；
- 差异主要落在 service factory、adapter、GPU connector、metadata 构造；
- 因此它不是简单复制几份 glue code，而是有明确分层的多引擎架构。

---

这一章读完，你应该把 LMCache 的“多引擎支持”理解成一句话：

1. **内核统一的是缓存语义，不是引擎表面 API。**
2. **integration 层负责把不同 runtime 的 request 生命周期、KV 布局和角色模型翻译进来。**
3. **真正的解耦，不是没有耦合，而是把耦合限制在 adapter、factory 和 GPU connector 这些最小必要边界里。**

你发送“继续”，下一章我会写 **第 8 章：可观测性、健康检查、部署形态与工程化落地**，把 LMCache 从“能跑”讲到“能上线”。