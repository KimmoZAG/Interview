# 坑位清单（持续更新）

定位：这不是主线正文，而是一份 **高频误判索引**。适合在复习、面试准备、性能排查时快速翻阅。

## 怎么使用这份清单

- 如果你在看主线时感觉“概念都懂，但一到排查就乱”，先翻这里
- 如果你在面试里被追问“常见误区/怎么定位”，这里可以直接给你答题骨架
- 如果你在做性能分析，优先把现象归到：算子、runtime、serving、通信、训练/评测 其中一类

## 1. 把单算子优化误当成系统优化

- 场景：你换了更快 kernel，或做了 fusion
- 现象：单算子 benchmark 变好，但端到端收益有限
- 根因：系统瓶颈其实在调度、KV cache、allocator、通信或尾延迟
- 解决：分开看 prefill/decode、单算子/端到端、平均值/p99
- 如何验证：同时对比 kernel trace 与整体 TTFT/TPOT/tokens/s
- 相关链接：
	- [../01-operator-optimization/04-graph-fusion-scheduling.md](../01-operator-optimization/04-graph-fusion-scheduling.md)
	- [../02-inference-engine/05-optimization-playbook.md](../02-inference-engine/05-optimization-playbook.md)

## 2. 把 decode 当成 compute-bound 问题

- 场景：LLM 线上生成很慢
- 现象：以为 attention 公式复杂，所以一定是算力不够
- 根因：decode 常常更像小 shape + 读 KV 的 memory-bound 问题
- 解决：优先估算 KV cache、访存压力、layout 和 allocator 影响
- 如何验证：区分 prefill 与 decode 时间占比，并结合显存/带宽观察
- 相关链接：
	- [../03-llm-architecture/02-attention-kv-cache.md](../03-llm-architecture/02-attention-kv-cache.md)
	- [../02-inference-engine/07-paged-kv-and-allocator.md](../02-inference-engine/07-paged-kv-and-allocator.md)

## 3. 只看平均延迟，不看尾延迟

- 场景：服务吞吐看起来不错
- 现象：平均值正常，但用户体验仍差
- 根因：长短请求混跑、动态 batching、allocator 抖动把 p95/p99 拉高
- 解决：按长度分桶统计 TTFT/TPOT/p95/p99，而不是只看平均 tokens/s
- 如何验证：把短请求、长请求、混合请求分开测
- 相关链接：
	- [../02-inference-engine/04-llm-serving.md](../02-inference-engine/04-llm-serving.md)
	- [../02-inference-engine/06-observability-and-debugging.md](../02-inference-engine/06-observability-and-debugging.md)

## 4. 把“总通信量一样”误当成“性能也差不多”

- 场景：多卡训练/推理扩展效率变差
- 现象：觉得总传输字节差不多，性能也该差不多
- 根因：消息大小分布、collective 种类、同步点位置、拓扑不同
- 解决：先区分 latency-bound / bandwidth-bound / topology-bound
- 如何验证：看具体 collective 类型、频率和跨机/机内路径
- 相关链接：
	- [../04-communication/02-communication-foundations.md](../04-communication/02-communication-foundations.md)
	- [../04-communication/04-collectives.md](../04-communication/04-collectives.md)

## 5. 把 benchmark 提升误当成产品一定更好

- 场景：模型升级或后训练后离线分数更高
- 现象：团队想直接上线
- 根因：训练指标、能力指标、系统指标、产品指标不是一回事
- 解决：先确认上线底线指标，再看离线/线上是否同向改善
- 如何验证：同时汇报质量、延迟、成本、用户体验
- 相关链接：
	- [training-metrics-vs-product-metrics.md](training-metrics-vs-product-metrics.md)
	- [post-training-and-alignment.md](post-training-and-alignment.md)

## 6. 把 reward/verifier 当成“只要有就行”

- 场景：做偏好优化、RL 或 verifier-based 训练
- 现象：离线 reward 更高，但真实体验没明显变好
- 根因：奖励信号和真实目标有缝，或者 verifier 太窄、易被钻空子
- 解决：同时检查信号来源、可投机路径和独立评测
- 如何验证：加入对抗样本、成本指标、行为回归样例
- 相关链接：
	- [post-training-and-alignment.md](post-training-and-alignment.md)
	- [reward-and-verifier-design.md](reward-and-verifier-design.md)

## 7. 把数据量增长误当成数据价值增长

- 场景：扩充预训练数据或调整混配
- 现象：token 数更多，但收益不稳定
- 根因：重复内容、低信息密度语料、混配失衡
- 解决：跟踪过滤率、去重率、抽样质量和小实验结果
- 如何验证：对比“数据量变化”和“有效能力变化”而不是只看 loss
- 相关链接：
	- [pretraining-data-engineering.md](pretraining-data-engineering.md)
	- [data-mixing-and-curriculum.md](data-mixing-and-curriculum.md)

## 可复用答题模板

如果面试里被问“你怎么定位/你见过什么坑”，可以优先按这 5 句展开：

1. 先界定场景：训练、推理、服务还是通信
2. 先描述现象：是慢、抖、错，还是成本异常
3. 再给根因分类：算力、带宽、调度、数据、奖励信号
4. 说明验证方法：看哪些指标、trace、样例或对照实验
5. 最后给解决动作：改什么、如何回归验证

## 新增条目模板

## 标题（一句话概括）

- 场景：
- 现象：
- 根因：
- 解决：
- 如何验证：
- 相关链接：
