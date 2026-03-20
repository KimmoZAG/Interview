---
tags:
  - AI Infra
  - LMCache
  - Disaggregated Prefill
  - 控制平面
description: 解释 LMCache 在 PD 场景下如何通过 controller、worker、full sync 和执行器协调跨实例 KV 生命周期与控制操作。
---

# 第 6 章：Disaggregated Prefill 与控制平面

配套入口：

- [README.md](README.md)
- [00-index.md](00-index.md)
- [03-storage-hierarchy-offload-and-lifecycle.md](03-storage-hierarchy-offload-and-lifecycle.md)
- [05-p2p-lookup-and-network-communication.md](05-p2p-lookup-and-network-communication.md)

前面几章你已经看到，LMCache 不只是“能存 KV”，它还能：

- 把 KV 分层放到不同 backend；
- 做跨实例 lookup；
- 通过 P2P 或 remote 路径拉取对象。

但一旦进入 **Disaggregated Prefill** 场景，问题会再升级一层。

因为这时候系统不只是“谁有缓存，别人去借一下”，而是：

- prefill 节点和 decode 节点本来就被拆开部署；
- prefill 生产 KV，decode 侧要尽快消费 KV；
- 整个集群需要知道哪些 worker 在线、哪些 key 在哪、哪些 key 正在搬运、哪些 key 可以淘汰。

所以 LMCache 才会引入一整套控制平面：

- `LMCacheControllerManager`
- `LMCacheWorker`
- `KVController`
- `LMCacheClusterExecutor`
- `FullSyncSender`

这不是为了“做复杂”，而是因为只靠数据面 API，已经不足以把 PD 系统管稳。

## 技术背景

### 1. 为什么 prefill 和 decode 会被拆开

Prefill 和 decode 在系统特征上几乎是两种完全不同的负载。

Prefill 阶段的特点是：

- 一次性处理长上下文；
- attention 和 KV 写入吞吐大；
- 更像大批量算力任务。

Decode 阶段的特点是：

- 每步只处理极少 token；
- 极端延迟敏感；
- 更像细粒度在线服务。

把两者混在同一套资源池里跑，常见结果是：

- prefill 抢占 decode 的延迟预算；
- decode 的碎片化调度拖累 prefill 吞吐；
- 调度器很难同时把两种目标都优化好。

所以 PD 分离本质上是：**把高吞吐阶段和低延迟阶段分开放到更适合的资源形态上。**

### 2. 一旦 PD 分离，最大问题就不再是算，而是 KV 的生命周期协同

很多人会把 PD 系统理解成：“prefill 节点算完，把 KV 发给 decode 节点”。

这句话不算错，但仍然太浅。真正难的不是“能发”，而是下面这些事情怎么协同：

- decoder 什么时候确认这段 KV 已经可用；
- prefiller 什么时候可以安全释放它；
- 如果 controller 重启或状态丢失，集群怎么恢复；
- 某个 worker 超时或失联时，全局 key 视图怎么更新；
- 一批 admit/evict 事件与 full sync 并发时，怎样避免控制面状态错乱。

这已经是标准的控制平面问题，而不是单纯数据拷贝问题。

### 3. 为什么 LMCache 需要 controller，而不是让 worker 彼此直接感知所有状态

可以想象一种最“去中心化”的设计：

- 每个 worker 自己维护局部 key 集；
- 所有 worker 两两同步 admit/evict；
- 发生 move / pin / full sync 时自行协商。

这个方案在 worker 数量稍大后几乎一定会失控：

- 状态同步复杂度高；
- 两两通信很多；
- 故障恢复难；
- 很难形成统一的全局查找和调度视图。

所以 LMCache 的选择很务实：

- **控制状态集中到 controller**；
- **数据本体仍尽量在 worker 之间直传或分层存放**。

这和第 5 章的 P2P 设计一脉相承：控制面集中，数据面分布。

### 4. 为什么线上控制面真正怕的是“不一致”而不是“慢一点”

一个 KV 控制面即使慢一点，最坏通常是命中率变差、回退重算。

但如果控制面不一致，后果会更严重：

- controller 以为某 key 还在，实际已经被 evict；
- worker 正在 full sync，controller 却继续接纳旧增量消息；
- 某实例重启后本地热层还在，但全局 registry 丢了；
- move/pin 请求作用到了不完整或已过期的数据集。

所以 LMCache 在控制面设计里非常强调：

- 注册与心跳；
- freeze 模式；
- full sync；
- 同步状态查询；
- 显式的 worker request / worker message / orchestration message 分层。

## 技术核心（结合代码）

### 1. `LMCacheControllerManager`：控制平面的总路由器

主入口在 `lmcache/v1/cache_controller/controller_manager.py`。

这个类的角色很像控制平面的前门：

- 建立 PULL / ROUTER / heartbeat socket；
- 接收来自 worker 的 push 消息和 req/reply 消息；
- 把不同消息路由给不同微控制器；
- 管理 cluster executor；
- 跑健康检查线程。

它内部最重要的两个组件是：

- `RegistrationController`
- `KVController`

再加一个：

- `LMCacheClusterExecutor`

所以它本质上不是“一个控制器实现所有事情”，而是一个 **消息总线 + 微控制器路由器**。

### 2. LMCache 把控制面消息显式分成了三类

`cache_controller/message.py` 里的注释非常关键。它把消息分成：

1. `WorkerMsg`：worker -> controller，push 模式，不要求返回。
2. `WorkerReqMsg`：worker -> controller，req/reply，需要返回。
3. `OrchMsg`：外部编排/运维 -> controller，req/reply。

这个分类非常有系统味道，因为它明确区分了：

- **增量状态上报**，如 admit/evict/full sync batch；
- **需要即时回答的控制请求**，如 register/heartbeat/batched P2P lookup；
- **面向控制操作的编排命令**，如 clear/pin/move/compress。

如果没有这层分型，控制面很容易退化成一个“所有消息都走一个 socket 和一个 handler”的大泥球。

### 3. `LMCacheWorker`：每个 worker 既是数据面执行者，也是控制面的本地代理

`lmcache/v1/cache_controller/worker.py` 里的 `LMCacheWorker` 非常关键。

它承担的职责包括：

- 启动时向 controller `register()`；
- 建立 push / req / heartbeat / reply socket；
- 把本地 KV admit/evict 等事件 push 到 controller；
- 接收 controller 返回的控制命令或查询结果；
- 必要时执行 full sync。

这说明 worker 不是被动存储后端，而是控制面在节点上的常驻代理。

从系统视角看，它是连接 controller 和本地 `LMCacheEngine` 的桥：

- 往上和 controller 通信；
- 往下调用 engine / backend 执行真实操作。

### 4. 注册和心跳为什么是控制面最先要解决的问题

`LMCacheWorker.register()` 发送的是 `RegisterMsg`，内容包含：

- `instance_id`
- `worker_id`
- `ip`
- `port`
- `peer_init_url`

controller 回复 `RegisterRetMsg`，里面还能带 `heartbeat_url` 等配置。

这一步其实是在完成两件事：

1. 把 worker 纳入 controller 的全局视图。
2. 给 worker 下发后续控制所需的连接信息。

随后 heartbeat 会通过专门的 socket 独立发送。

这里最重要的设计点是：**heartbeat 不和其他控制请求共用同一条易阻塞路径。**

这非常合理，因为心跳是 liveness 保障。如果它和其他重请求共用 socket，一旦排队或超时，就会把活节点误判成死节点。

### 5. `KVController`：全局 KV 视图的维护者

`lmcache/v1/cache_controller/controllers/kv_controller.py` 是控制面的核心状态机之一。

它背后的核心数据观是：

- 哪个 `(instance_id, worker_id)`
- 在哪个 `location`
- 当前持有哪些 `chunk_hash`

注释里明确说了，当前 registry 近似是：

```text
(instance_id, worker_id) -> [location -> set[chunk_hash]]
```

这意味着 controller 维护的不是数据本体，而是 **全局 key 索引和位置映射**。

这正是控制面的价值所在：

- 查找时不用扫所有 worker 的本地实际缓存；
- move/pin/clear 等操作有全局视图可依赖；
- 故障恢复时知道理论上哪些 key 应该存在。

### 6. 增量 admit/evict 为什么要单独走 batched operation

从 `LocalCPUBackend` 往 controller 发的不是每个操作一个 RPC，而是 `BatchedKVOperationMsg`。

这点在 `KVController.handle_batched_kv_operations(...)` 里能看清楚。

理由非常现实：

- KV 对象 admit/evict 频率可能很高；
- 单条上报网络和序列化开销太碎；
- controller 维护 registry 时更适合按批更新。

而且 controller 在 full sync 期间还会显式丢弃这些增量操作，避免和一次完整状态重建相互污染。

这说明 LMCache 对控制面一致性的优先级非常高：

**当完整状态重建进行中，宁可暂时放弃增量，也不要把新旧状态混在一起。**

### 7. `FullSyncSender` + freeze：这是 controller 重启和状态恢复的关键机制

这一块是第 6 章最该重点记的地方。

`cache_controller/full_sync_sender.py` 定义了 worker 侧 full sync 发送器，流程很清楚：

1. 随机 startup delay，避免 herd effect。
2. `lmcache_engine.freeze(True)` 进入 freeze 模式。
3. 扫描 `LocalCPUBackend` 的所有热 key。
4. 发 `FullSyncStartMsg`。
5. 分批发 `FullSyncBatchMsg`。
6. 发 `FullSyncEndMsg`。
7. 轮询 `FullSyncStatusMsg`，直到 controller 确认可以退出 freeze。

这套机制的重要性不亚于数据面本身。

因为一旦 controller 重启或 registry 丢失，如果没有 full sync：

- worker 本地明明还有大量热 KV；
- 但 controller 完全不知道；
- 后续所有 lookup/p2p/move 决策都建立在错误全局视图上。

所以 full sync 本质上是在做：

**把 worker 本地现实状态重新灌回控制面。**

### 8. 为什么 full sync 期间要 freeze

这也是非常典型的系统设计题。

如果 full sync 期间不 freeze，会发生什么？

- worker 一边在扫旧 key 发给 controller；
- 本地热层一边还在继续 admit/evict；
- controller 收到的视图就可能既不代表旧状态，也不代表新状态。

freeze 模式的语义，在 `LMCacheEngine.freeze()` 和 `StorageManager.set_freeze()` 里很清楚：

- 停止新增 store；
- retrieval 只走 local CPU；
- 不再生成新的 admit/evict 干扰消息。

这就是一种非常标准的恢复期写冻结策略：

**先让状态静下来，再做控制面重建。**

### 9. `KVController.handle_full_sync_*`：controller 端如何保证重建过程可控

KVController 在 full sync 上拆了三步处理：

- `handle_full_sync_start(...)`
- `handle_full_sync_batch(...)`
- `handle_full_sync_end(...)`

核心逻辑是：

1. `start` 时先把 worker 标记成 syncing，并清掉该 worker 旧 registry 记录。
2. `batch` 时把整批 key 作为 admit 事件灌入 registry。
3. `end` 时记录完成状态，并和实际 key 数做核对。

配套还有 `handle_full_sync_status(...)`，让 worker 轮询同步进度和缺失批次。

这里的设计重点是：full sync 不是 fire-and-forget，而是一个带状态跟踪和缺批重发语义的同步会话。

这说明 LMCache 很认真地把控制面恢复当成了一个正式协议，而不是临时脚本。

### 10. `FullSyncTracker` 背后的思想：恢复期最怕的是“看起来完成，实际不完整”

虽然这一章不展开 `FullSyncTracker` 的源码细节，但从 `KVController` 的调用方式已经能看出它做什么：

- 跟踪 worker 是否处于 syncing 状态；
- 记录 batch 是否收到；
- 计算 global progress；
- 判断是否可以退出 freeze；
- 识别 missing batches。

这件事的本质，是避免一种最危险的状态：

**worker 觉得自己已经同步完了，但 controller 的 registry 其实还不完整。**

一旦这种错觉发生，后续所有 lookup 和 move 决策都会带着隐形错误继续跑。

### 11. `batched_p2p_lookup(...)`：controller 在 P2P 场景里扮演的是“全局发现器”

在 `KVController.batched_p2p_lookup(...)` 里，controller 做的事很克制：

1. 看第一个 hash 是否在别的 instance 存在；
2. 若存在，拿到该 instance 的 `peer_init_url` 和当前位置；
3. 再数一数这批 hashes 连续命中了多少 chunk；
4. 返回 `(instance_id, location, num_hit_chunks, peer_init_url)`。

这说明 controller 在 P2P 路径里的定位不是数据代理，而是全局发现器。

这非常关键，因为如果 controller 也负责转发真实 KV，很快就会成为中心化数据瓶颈。

LMCache 在这里显然有意识把 controller 控制流和真实数据流分离开了。

### 12. `LMCacheClusterExecutor`：控制平面不只是查，还会下发真正的 cluster 操作

`cache_controller/executor.py` 的 `LMCacheClusterExecutor` 很容易被忽略，但它非常有代表性。

它会把外部编排请求（`OrchMsg`）翻译成对各 worker 的操作，比如：

- `clear`
- `pin`
- `compress`
- `decompress`
- `move`

做法很直接：

- 找到目标 instance 的 worker sockets；
- 构造 per-worker message；
- 并发下发；
- 收集各 worker 返回结果；
- 做一致性断言。

这说明 controller 不是只维护一个被动 registry，它还具备主动操作集群 KV 状态的能力。

从面试角度，你可以把它讲成：**LMCache 的控制面已经具备“小型分布式缓存控制器”的雏形。**

### 13. `MoveMsg` / `PinMsg` / `ClearMsg`：为什么这些控制操作必须走 controller，而不是直接点对点调用

在 `message.py` 里你会看到一整套 orchestration 消息。

这些操作之所以走 controller，而不是直接命中某个 worker，本质原因是：

- 谁应该执行、执行哪些 worker，需要全局视图；
- move 这类操作往往涉及源 instance 和目标 instance 配对；
- controller 可以统一生成 event_id、聚合结果、做一致性判断。

如果这些动作都靠外部系统直接手工打到某个 worker，很容易出现：

- 目标选错；
- 多 TP rank 不一致；
- 某些 worker 操作成功，另一些失败却没人聚合感知。

所以 orchestration 经 controller 汇总是非常合理的设计。

### 14. `PDBackend` 和 controller 的关系：一个管数据通道，一个管全局状态

前面第 3 章讲过 `PDBackend` 更像一个专用传输 backend。

到了第 6 章，你应该把它和 controller 放在一起看：

- `PDBackend` 负责 sender/receiver 之间实际的 buffer 分配与传输；
- controller / worker 体系负责谁在线、哪些 key 存在、什么时候 full sync、什么时候 move/pin/clear。

也就是说，PD 场景里 LMCache 其实是两套系统同时工作：

1. **高速数据通路**：尽快把 KV 从 prefill 侧送到 decode 侧。
2. **慢速但关键的控制通路**：保证全局状态、恢复、查找和调度命令一致。

没有前者，性能上不去；没有后者，系统迟早跑乱。

### 15. worker 控制面设计里一个很重要的工程点：错误时要返回默认安全值

`LMCacheWorker._on_request_failure(...)` 里对不同消息定义了默认返回：

- `BatchedP2PLookupMsg` 失败时返回 0 命中；
- `HeartbeatMsg` 失败时返回空心跳回复；
- full sync 请求失败时返回 not accepted / incomplete。

这件事看起来很小，实际上很重要。

因为控制面一旦请求失败，最危险的不是“这次 miss”，而是上层拿到了模糊或非法状态继续做错误决策。

LMCache 这里的策略很务实：**通信失败时优先退回安全默认值，让数据面回退或重试，而不是继续冒险。**

### 16. 把第 6 章压成一条完整时序

从 PD + 控制面的角度，完整时序可以这么讲：

```text
Worker 启动
  -> RegisterMsg 注册到 controller
  -> 获取 heartbeat_url 等控制配置
  -> 持续 heartbeat 上报存活

正常运行中
  -> LocalCPUBackend / 其他 backend 产生 admit/evict 增量
  -> Worker batched push 到 controller
  -> KVController 更新全局 registry

P2P / lookup 场景
  -> Worker/Backend 发 BatchedP2PLookupMsg
  -> KVController 用全局 registry 找最优来源和连续命中长度
  -> 返回 peer_init_url 等 layout_info
  -> 数据面自行直连搬运 KV

控制操作场景
  -> 外部 OrchMsg 发到 controller
  -> LMCacheClusterExecutor 下发到对应 worker
  -> 聚合结果并返回

恢复场景
  -> Controller 状态丢失或要求重建
  -> Worker FullSyncSender 进入 freeze
  -> 发 start / batch / end / status
  -> Controller FullSyncTracker 确认同步完整
  -> Worker 退出 freeze
```

这就是 LMCache 真正意义上的控制面闭环。

## 面试可能问到的问题

### 问题 1：PD 体系里最大的性能风险是 KV 传输延迟，还是一致性/生命周期管理？

**满分回答思路：**

短期看最显眼的是传输延迟，因为它直接影响 decoder 何时能开始消费 KV；但从系统长期稳定性看，更危险的往往是一致性和生命周期管理。

原因是：

- 传输慢通常会表现成延迟上升或退化为重算；
- 但如果 controller 状态错、worker 释放时机错、pin/unpin 不闭环、full sync 不完整，系统会在看似命中的情况下做出错误决策，影响更隐蔽也更难排查。

所以满分回答应该是：**数据传输决定性能上限，一致性和生命周期管理决定系统能不能长期稳定跑。** LMCache 之所以引入 controller、worker、full sync、freeze，就是在补这条更难的链。

### 问题 2：prefill 节点和 decode 节点如何协调“什么时候可以安全释放 KV”？

**满分回答思路：**

不能只靠“数据发完了”这种局部条件决定释放，因为真正安全释放取决于消费者是否已经确认可用、控制面是否还需要追踪这批对象，以及是否还存在异步回迁或重试路径。

在 LMCache 这类系统里，通常要靠多层条件共同保证：

- 数据面上 receiver 已经完成分配和接收；
- 控制面上相关 lookup/pin 生命周期已结束；
- request 对应的 save/load 阶段已经收尾；
- 异常/abort 路径也执行了 cleanup。

所以这是一个“引用 + pin + 协议确认”的组合问题，绝不是 sender 本地看到传输完成就能直接 free。

### 问题 3：为什么控制面一旦抖动，可能把数据面一起拖死？如何设计降级？

**满分回答思路：**

因为很多数据面动作的前提依赖控制面给出的全局判断，例如：

- 哪个 peer 有这段 KV；
- full sync 是否完成；
- 某 worker 是否还被认为在线；
- move/pin/clear 应该作用到哪些节点。

如果控制面抖动却继续返回不可靠信息，数据面就可能把错误状态当成真，造成更严重的错误传播。

合理降级策略应该是：

- lookup/P2P 失败时优先回退到 0 hit 或 recompute；
- full sync 未完成时继续 freeze，不贸然恢复增量路径；
- heartbeat 失败时逐步隔离节点；
- backend/peer 失效时绕过该路径，保本地 L1 和重算路径可用。

核心原则是：**控制面异常时，优先牺牲命中率，不要牺牲正确性。**

---

这一章读完，你应该已经能把 LMCache 的 PD 和控制面讲成一套完整系统：

1. **PD 把问题从“缓存复用”升级成“跨角色 KV 生命周期协同”。**
2. **controller 维护全局视图，worker 负责本地代理与执行，executor 负责集群级操作分发。**
3. **full sync + freeze 是控制面恢复一致性的关键，不是附加功能。**

你发送“继续”，下一章我会写 **第 7 章：与 vLLM / SGLang 的集成与解耦设计**，把 LMCache 为什么能接不同 serving engine 讲透。