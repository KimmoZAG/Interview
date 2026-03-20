---
tags:
  - AI Infra
  - LMCache
  - 存储系统
  - KV Cache
description: 解释 LMCache 如何把 GPU KV 变成多层存储对象，以及对象写入、回迁、pin、refcount 和淘汰如何协同工作。
---

# 第 3 章：存储层、分层卸载与对象生命周期

配套入口：

- [README.md](README.md)
- [00-index.md](00-index.md)
- [01-overall-architecture-and-core-abstractions.md](01-overall-architecture-and-core-abstractions.md)
- [02-chunking-hashing-and-metadata.md](02-chunking-hashing-and-metadata.md)

如果第 2 章回答的是“**什么东西可以复用**”，那第 3 章回答的就是“**这些可复用对象到底存哪、怎么搬、什么时候删**”。

LMCache 真正体现系统味道的地方，就在这一章。

因为 KV Cache 一旦离开模型 forward 内部，它就不再只是张量，而会变成一个标准的基础设施问题：

- 内存池怎么分配；
- 本地 CPU 热缓存怎么维护；
- 远端存储怎么异步写入；
- 命中后是否要 write-back 到本地；
- 对象什么时候允许被淘汰；
- 并发读取和回迁时怎样避免死锁、重复释放和热点抖动。

## 技术背景

### 1. 为什么单 GPU 持有所有 KV 是不现实的

没有 LMCache 或类似体系时，最直观的做法就是：KV 留在 GPU，能命中就命中，放不下就删。

这个方案在短上下文、低并发时没问题，一旦进入长上下文和真实线上场景，马上暴露三个结构性瓶颈。

第一是 **显存贵**。

KV Cache 的体积增长和上下文长度几乎线性相关。长对话、RAG、大系统 prompt 都会把显存迅速打满。显存一旦被历史 KV 占住，新请求就更难进入，最终拖低吞吐。

第二是 **GPU 的职责不该是长期存档层**。

GPU 最值钱的是算 attention 和 matrix multiply，而不是长期保存历史状态。把稀有但不一定马上要用到的 KV 一直留在 GPU，本质上是用最贵的资源干最不划算的事情。

第三是 **跨实例复用根本没解**。

即便某个 worker 的 GPU 里还留着 KV，只要下一次请求跑到别的 worker，这些数据仍然等于不存在。也就是说，单 GPU 缓存最多解决“局部短时间命中”，解决不了“系统级复用”。

所以 KV 一旦要真正复用，必然会走向多层存储。

### 2. 为什么“把 KV 放 CPU”依然不是完整答案

很多人会把 offload 简化成一句话：把 GPU KV 拷到 CPU。

这句话只说对了最表面的一层，真正的工程问题才刚开始。

你马上会碰到：

- CPU 内存怎么分配，是否 pinned，是否 NUMA 感知；
- 对象是一个整块 tensor，还是分层/分页对象；
- 多个 backend 并存时，谁是 allocator，谁只是落盘/远传层；
- 远端拿回来之后要不要自动写回本地 CPU；
- lookup pin 期间如何避免正在被取走的对象被淘汰；
- 异步 batched get 并发时如何避免 allocator 死锁。

所以真实系统里，“CPU offload” 不是一个动作，而是一整套对象生命周期管理。

### 3. 为什么 LMCache 会自然长成“L1/L2/remote/P2P/PD”这种多层形态

一旦把问题视为存储系统，而不是单机缓存技巧，多层结构几乎是自然结果。

- **L1 / local CPU**：延迟最低，适合作为热缓存和 write-back 层。
- **disk / remote**：容量更大，适合保存不常驻但仍有复用价值的对象。
- **P2P**：适合同节点/对等 worker 之间直接共享 KV，减少中心化存储绕行。
- **PD backend**：服务于 prefill/decode 分离场景，本质上是“边算边传”的专用数据通道。

LMCache 的一个强项，就是它没有把这些能力缝死在一起，而是通过统一 `StorageManager` 把它们编排起来。

## 技术核心（结合代码）

### 1. 先抓住核心对象：LMCache 存的不是 tensor，而是 `MemoryObj`

核心定义在 `lmcache/v1/memory_management.py`。

这一步非常关键。LMCache 没有让后端直接持有“某个引擎里的原始 GPU tensor”，而是先统一抽象成 `MemoryObj`。

`MemoryObj` 携带两类信息：

- **逻辑信息**：shape、dtype、memory format、token 数量；
- **生命周期信息**：地址、物理大小、`ref_count`、`pin_count`。

其中元信息由 `MemoryObjMetadata` 描述：

```python
class MemoryObjMetadata:
    shape
    dtype
    address
    phy_size
    ref_count
    pin_count
    fmt
    shapes
    dtypes
```

这个设计解决了一个根问题：

**一旦 KV 被抽成 `MemoryObj`，存储后端就只需要理解标准对象接口，而不需要理解 vLLM 的 paged KV 内部结构。**

这就是 LMCache 能同时支持 local CPU、remote、P2P、PD 等后端的基础。

### 2. `MemoryFormat` 说明 LMCache 不是只存一种 KV 布局

`MemoryFormat` 枚举里有多种布局：

- `KV_2LTD`
- `KV_T2D`
- `KV_2TD`
- `KV_MLA_FMT`
- `BINARY` / `BINARY_BUFFER`

这说明 LMCache 的存储对象并不假设所有 KV 都是同一种物理形态。它把“怎么摆放这块内存”作为元数据的一部分显式带着走。

这是很成熟的系统设计：

- key 负责身份；
- metadata 负责结构；
- format 负责解释方式。

这样后面不论是 layerwise、MLA 还是压缩/二进制流，都有空间挂进去。

### 3. `TensorMemoryObj` 的 `ref_count` 和 `pin_count` 是对象生命周期的核心

`TensorMemoryObj` 是最重要的 `MemoryObj` 实现。

它的生命周期控制逻辑很值得你记住：

- `ref_count_up()` / `ref_count_down()` 表示对象当前被多少地方持有；
- `pin()` / `unpin()` 表示对象是否被暂时禁止淘汰；
- `can_evict` 的判断是：**未 pin 且 `ref_count == 1`**。

这套规则看起来简单，但非常工程化。

为什么 `ref_count == 1` 才能淘汰？

因为通常这“最后一个引用”就是 allocator/backend 自己手上的持有；只有没有外部使用者了，才允许回收。

为什么还要 `pin_count`？

因为 lookup 命中、控制面 pin、异步加载中的对象，都可能还不该被逐出。单靠 refcount 无法表达“逻辑上暂时锁住、但不一定多了一个真正消费者引用”这种状态。

这正是很多缓存系统线上最容易踩坑的地方：**只有引用计数，没有 pin 语义，最终会在预取和淘汰并发时出错。**

### 4. `StorageManager` 是多层存储的统一编排器

`lmcache/v1/storage_backend/storage_manager.py` 里的 `StorageManager` 是本章最重要的类。

它做的事情不是“某个 backend 的实现”，而是统一调度所有 backend：

- 创建 backend；
- 维护 allocator backend；
- 组织批量 put/get/contains；
- 管理异步 lookup/prefetch；
- 控制 freeze / bypass / hot cache。

它的构造里就暴露了几个设计重点：

- 自己开一个 event loop 线程处理异步存储任务；
- 区分 `allocator_backend` 和普通 backend；
- 对 async loading 路径使用 serializer 避免并发取数据时 allocator 死锁；
- 在 CUDA worker 上准备内部 copy stream。

这说明 LMCache 很清楚：**多层存储的难点不只是后端种类多，而是并发状态和对象所有权复杂。**

### 5. `allocator backend` 的设计很关键：不是每个 backend 都负责分配内存

这是第 3 章一定要抓住的抽象。

在 `StorageManager.batched_put(...)` 里，它会先构造一个 `obj_dict`，并默认把原始 `keys/memory_objs` 归属给 allocator backend。然后对于其他 backend：

1. 看它对应的 allocator backend 是谁；
2. 如果需要，调用 `allocate_and_copy_objects(...)` 先分配新的对象并把数据拷过去；
3. 再调用该 backend 的 `batched_submit_put_task(...)`。

伪代码可以写成：

```python
obj_dict[allocator_backend] = (keys, memory_objs)

for backend in storage_backends:
    allocator = backend.get_allocator_backend()
    if allocator not in obj_dict:
        obj_dict[allocator] = allocate_and_copy_objects(allocator, keys, memory_objs)

    ks, objs = obj_dict[allocator]
    backend.batched_submit_put_task(ks, objs)

for all objs in obj_dict:
    obj.ref_count_down()
```

这个设计的系统意义是：

- 数据真正的物理落点可以和写入后端解耦；
- backend 可以共享同一个 allocator 产出的对象；
- 不同 backend 不必重复理解内存布局和分配细节。

说白了，**LMCache 不只是有多个 storage backend，它还有一个“对象分配层”和“对象存储层”的分离。**

### 6. Local CPU backend 本质上是 LMCache 的 L1 热缓存

`lmcache/v1/storage_backend/local_cpu_backend.py` 很值得仔细看，因为它不是简单的“本地内存字典”。

它本质上承担的是 L1 热缓存角色：

- `hot_cache` 里维护 key -> `MemoryObj`；
- 支持 `contains()` 时 pin 对象；
- `submit_put_task()` 会同步放入本地热缓存；
- `touch_cache()` 根据请求顺序更新 cache policy；
- `remove()` 触发对象的引用回收。

这里你会看到 LMCache 把本地 CPU 层看得非常重：

- 命中最快；
- 可作为 allocator backend；
- 可作为远端数据的 write-back 目标；
- 可在 freeze 模式下单独保留为“只读热层”。

这在系统上很合理，因为真正高价值的多层缓存体系，通常都离不开一个低延迟的本地热层。

### 7. cache policy 并不是装饰品，而是本地热层的核心策略

`local_cpu_backend.py` 里通过 `get_cache_policy(config.cache_policy)` 注入策略，支持：

- `LRU`
- `LFU`
- `FIFO`
- `MRU`

虽然默认和最常见的是 `LRU`，但这个设计已经表明 LMCache 把 local CPU hot cache 明确看成一个需要策略治理的缓存层，而不是单纯的哈希表。

你面试里可以顺手带出一点：

**纯 LRU 在某些长请求和批量预取场景并不总是最优。**

原因是：

- 大对象一次进入就可能挤掉大量小热点对象；
- 流式扫描类请求会污染 cache；
- 预取对象可能还没真正被消费就占住热层。

这也是为什么“缓存策略”在 KV 系统里不是一个微不足道的参数。

### 8. `RemoteBackend`：容量层/远端层的主路径是异步写入和按需回迁

`lmcache/v1/storage_backend/remote_backend.py` 代表的是远端存储这条路。

它的关键特征是：

- 初始化远端 connector；
- 使用 serializer / deserializer 处理对象；
- `submit_put_task()` 和 `batched_submit_put_task()` 异步写远端；
- `contains()` / `batched_contains()` 做远端存在性查询；
- 失败时做连接重建和计数统计。

这条路径的工程重点不在“能不能 put/get”，而在于：

- **写入不能阻塞主链路**；
- **远端失败不能把本地服务拖死**；
- **拿回来的对象最好能写回 LocalCPUBackend，形成更快的后续命中。**

这最后一点在 `StorageManager.get()` / `batched_get()` 里能看到：

- 如果对象是从 `LocalCPUBackend` / `PDBackend` 之外的 backend 取回来的；
- 且本地有 `LocalCPUBackend`；
- 则会自动写回本地热层。

这就是非常标准的 **read-through + write-back** 思想。

### 9. 为什么 `StorageManager.get()` 要对远端结果做本地 write-back

这个点非常面试化，值得单独说。

如果远端命中后不写回本地，那么同一批热点 KV 后续请求每次都要走远端。这样：

- 远端 RTT 持续压着 TTFT；
- 远端存储会被热点反复打；
- 本地层永远学不到真实热点。

而有了 write-back，本地 CPU 会逐渐形成热点镜像：

- 第一次 miss 本地、hit 远端；
- 第二次及后续请求很可能直接命中本地 L1。

这是典型的层级缓存思路。LMCache 在这里不是把 backend 当平铺列表，而是实际上形成了层次化访问路径。

### 10. `async_lookup_and_prefetch()`：真正把多层存储串成“连续前缀加载链”

`StorageManager.async_lookup_and_prefetch(...)` 是很关键的一段逻辑。

它做的事情不是简单“并发查多个后端”，而是按 tier 去拼出一个**连续前缀**：

1. 对每层 backend 做 `batched_async_contains()`。
2. 得到每层能连续命中的 chunk 数。
3. 对命中的 chunk 提交 `batched_get_non_blocking()`。
4. 用 `event_manager` 记录加载事件。
5. 汇总真实返回结果，计算最终连续可用 token 数。

这里最漂亮的一点是它对“前缀完整性”的坚持。

源码里的注释举了非常典型的例子：

- Tier0 预期有 3 个 chunk，但实际只成功拿回 2 个；
- 即便 Tier1 / Tier2 又拿回了后面的 chunk，这些结果也不能算作连续可用前缀；
- 因为中间已经断了。

这说明 LMCache 宁愿损失一些表面上的命中数字，也要保证“最终返回给 scheduler 的，是逻辑上连续可加载的前缀长度”。

这和第 2 章的前缀哈希设计是完全一致的。

### 11. `AsyncSingleSerializer` / `WeightedSemaphore`：为什么异步加载里还要主动串行化

很多人看到异步加载，直觉是“并发越高越好”。但 `StorageManager` 里专门实现了：

- `AsyncSingleSerializer`
- `AsyncMultiSerializer`
- `WeightedSemaphore`

本质原因是：**本地 CPU allocator 和批量 get 并不是无穷并发安全的。**

源码注释明确写了，多个 `batched_get` 并发执行时，local CPU backend 可能同时分配 memory object，进而死锁。

这在工程上非常真实：

- 异步并发增加了吞吐潜力；
- 但 allocator、pinned memory 和 copy 路径的内部状态可能不是可无限并发的；
- 所以必须在“看起来异步”与“底层资源安全”之间加一道节流器。

这也是为什么优秀的存储系统常常不是“全并发”，而是“受控并发”。

### 12. `P2PBackend`：本质上是对等节点间的直接 KV 共享层

`lmcache/v1/storage_backend/p2p_backend.py` 把 P2P 也包装成一个标准 backend，这点很有设计感。

它内部做的事情包括：

- 向 controller 发 batched P2P lookup；
- 维护 target peer info 和 lookup socket；
- 使用 transfer channel 直接在 peer 间搬运数据；
- 结合本地 `LocalCPUBackend` 的 allocator 作为对象承接层。

你可以把它理解成：

- 命中信息先通过控制面发现；
- 真正的数据则尽量走对等直连通道，而不是都绕统一中心存储。

这在系统上非常重要，因为热点 KV 一旦能直接 P2P 命中，往往比统一远端存储更低延迟，也更省中心层带宽。

### 13. `PDBackend`：不是普通冷存储，而是 Prefill/Decode 之间的专用传输层

`lmcache/v1/storage_backend/pd_backend.py` 和前面那些 backend 的气质完全不同。

它更像一个“带存储接口外观的数据传输 backend”。

原因很简单：在 PD 分离里，prefiller 的目标不是长期保存 KV，而是**及时把刚算出来的 KV 送到 decoder**。所以 `PDBackend`：

- sender 侧更像直接写对端；
- receiver 侧才真正持有数据和 allocator；
- 通过 side channel 做 allocation 协调；
- 再通过 transfer channel 走实际数据通道。

这说明 LMCache 对 backend 的抽象非常务实：

**backend 不一定非得是“长期存储介质”，也可以是“符合相同 put/get 契约的传输承载层”。**

### 14. `distributed/storage_manager.py`：多进程模式下已经显式采用 L1/L2 控制器架构

除了 v1 主路径的 `storage_backend/storage_manager.py`，仓库里还有 `lmcache/v1/distributed/storage_manager.py`。

这一套更明确地把多层存储写成了：

- `L1Manager`
- 多个 `L2Adapter`
- `StoreController`
- `PrefetchController`
- `EvictionController`

它的 API 也很像标准存储系统：

- `reserve_write(...)`
- `finish_write(...)`
- `submit_prefetch_task(...)`
- `read_prefetched_results(...)`
- `finish_read_prefetched(...)`

这说明 LMCache 的演化方向已经很清晰：**从“一个服务引擎的缓存外挂”，逐步走向更完整的分布式多层存储控制面。**

这一点面试时说出来很加分，因为它反映你看到了项目的架构趋势，而不是只盯住单个 Python 文件。

### 15. 对象生命周期总结：一块 KV 从出生到死亡的完整路径

把前面的细节压成一条完整生命周期，最容易建立整体感。

#### 阶段 1：出生

- `LMCacheEngine.store()` 根据 chunk key 决定要保存哪些对象。
- `StorageManager.allocate()` / `batched_allocate()` 从 allocator backend 分配 `MemoryObj`。
- `gpu_connector.batched_from_gpu(...)` 把 GPU KV 写入这些对象。

#### 阶段 2：进入各层

- `StorageManager.batched_put()` 把对象分发给一个或多个 backend。
- Local CPU 可能直接同步持有一份。
- Remote/P2P/PD 可能拿到复制后的对象或序列化后的内容。

#### 阶段 3：被查找与命中

- `contains()` / `batched_contains()` 判断对象是否存在。
- lookup 时可能给对象 `pin()`，防止在请求处理中被淘汰。

#### 阶段 4：被回迁

- `get()` / `batched_get()` / `batched_get_non_blocking()` 返回 `MemoryObj`。
- `gpu_connector.batched_to_gpu(...)` 把对象写回 GPU KV buffer。
- 若来自远端层，可能自动 write-back 到 `LocalCPUBackend`。

#### 阶段 5：释放与淘汰

- 使用方执行 `ref_count_down()`。
- lookup pin 结束后执行 `unpin()`。
- 当对象不再被持有、也没有 pin 时，allocator 才能真正 free。
- cache policy / eviction controller 决定谁先被逐出。

这整条链其实就是一句话：

**LMCache 管的不只是“有没有这个 KV”，而是“它当前在哪一层、谁在用、能不能删”。**

## 面试可能问到的问题

### 问题 1：CPU offload 为什么经常不是算力瓶颈，而是内存带宽和拷贝路径瓶颈？

**满分回答思路：**

因为 offload 的主成本通常不在算，而在搬数据。

KV Cache 是大块张量，写入和回迁的主路径通常是：

- GPU -> pinned CPU
- CPU 内存重排/序列化
- CPU -> remote 或 CPU -> GPU

这里真正决定上限的是：

- PCIe / NVLink 带宽；
- pinned memory 分配与管理；
- NUMA 亲和性；
- CPU 内存带宽；
- 序列化/压缩的开销。

LMCache 里大量设计都在回应这个现实，比如 pinned allocator、internal copy stream、batched put/get、以及把 `MemoryObj` 作为统一对象减少无谓转换。结论就是：**KV offload 更像一个 IO 系统问题，而不是计算优化问题。**

### 问题 2：为什么缓存系统里仅靠 refcount 不够，还必须有 pin 语义？

**满分回答思路：**

refcount 只表达“当前有多少使用者持有这个对象”，但不能完整表达“这个对象逻辑上暂时不允许被淘汰”。

典型场景：

- scheduler/lookup 已经决定后面要用它；
- 异步 prefetch 已经发出，但消费者还没真正拿到；
- 控制面正在做 move/prefetch/evict 协调。

这时对象可能并没有很多外部引用，但如果允许 eviction，就会在取数与淘汰并发时出错。LMCache 通过 `pin_count` 把“暂时保活”与“真实引用”分开表达，这能显著减少 race condition 和 use-after-free 风险。

### 问题 3：多 backend 并存时，如何避免写放大和回迁抖动？

**满分回答思路：**

这是分层缓存设计的核心难题。写放大和回迁抖动通常来自两个地方：

- 同一对象被无差别复制到多层，造成带宽和容量浪费；
- 远端对象每次被读回都重复走远端，没有在本地形成稳定热层。

LMCache 的思路大致是：

- 用 `allocator backend` + backend 分层，把对象分配和后端写入解耦；
- 通过 LocalCPUBackend 充当 L1 热层；
- 从远端 get 回来后自动 write-back 到本地；
- 通过 cache policy、freeze、backend bypass、async serializer 控制抖动和失败传播。

如果继续延伸到更理想的答案，你可以说真正成熟的系统还会进一步加入 admission control、cost-aware eviction 和热点限流，否则高频热点仍然可能在层间来回震荡。

---

这一章读完，你应该已经真正看懂 LMCache 为什么不是“KV 存一下”这么简单。

1. **它先把 KV 抽象成 `MemoryObj`，让对象身份、结构和生命周期都可管理。**
2. **它用 `StorageManager` 把 Local CPU、remote、P2P、PD 编排成分层体系。**
3. **它靠 refcount、pin、write-back、serializer 和 cache policy 保住正确性与性能。**

你发送“继续”，下一章我会写 **第 4 章：GPU Connector 与异步 Store/Load 主链路**，把 GPU tensor 如何进入 LMCache、又如何异步回到引擎 KV buffer 这条热路径彻底讲透。