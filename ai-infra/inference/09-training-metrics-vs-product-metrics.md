# 训练指标 vs 产品指标：为什么 loss 变好不代表产品更好

## 要点

- 从 CS336 Lecture 12 以及后续 alignment 部分的视角看，训练指标、离线 benchmark、线上产品指标回答的是三类不同问题，不能混成同一张成绩单。
- 一个常见误区是：看到 validation loss 或 benchmark accuracy 提升，就默认用户体验、成本和安全性也同步变好。
- 真正有用的评测框架必须同时覆盖：训练质量、任务能力、系统性能、产品行为。
- 如果你的汇报不能回答“这次改动究竟改善了哪一层，伤害了哪一层”，那这套评测体系基本就是失效的。

## 1. 为什么这两个世界经常对不上

训练阶段最自然的指标通常是：

- training loss
- validation loss
- perplexity

它们回答的是：

- 模型是否更会预测训练分布或验证分布中的下一个 token

但产品场景真正关心的往往是：

- 回答有没有帮助
- 格式是否稳定
- 延迟是否可接受
- 成本是否可接受
- 是否符合安全与合规要求

所以一开始就要承认：

- 训练目标和产品目标天然不是同一个目标函数

## 2. 四层指标框架

一个足够实用的框架，是把指标分成四层。

### 第一层：训练指标

- training loss
- validation loss
- perplexity

用途：

- 判断训练是否收敛
- 比较不同超参数、数据配方、模型规模是否在同一方向上改进

### 第二层：能力指标

- benchmark accuracy
- task success rate
- coding / math / reasoning 得分

用途：

- 判断模型是否在特定能力上更强

### 第三层：系统指标

- TTFT
- TPOT
- p95 / p99 latency
- tokens/s
- req/s
- 显存与 GPU 利用率

用途：

- 判断模型是否能以合理成本跑起来

### 第四层：产品与行为指标

- 用户满意度
- 任务完成率
- 人工偏好胜率
- 输出格式遵循率
- 拒答/误答/违规率

用途：

- 判断模型是否真的更像一个可用产品，而不只是分数更高

## 3. 为什么 loss 下降仍然可能伤害产品

几个常见场景：

### 场景 1：模型更会续写，但更不听话

表现：

- validation loss 下降
- 但 instruction following 变差

原因通常是：

- 训练目标优化的是 token 预测，不是行为约束

### 场景 2：能力更强，但成本爆炸

表现：

- benchmark accuracy 提升
- 但 latency 和 GPU 成本大幅恶化

### 场景 3：对齐更强，但基础能力受损

表现：

- 格式更稳定、拒答更稳
- 但 reasoning 或 factual accuracy 下滑

## 4. 一张评测表至少要同时放什么

如果你在做一次重要模型改动，评测表至少应包含：

- 一个训练质量指标
- 一个目标任务指标
- 一个系统性能指标
- 一个资源成本指标
- 一个行为或产品指标

例如：

| 维度 | 示例指标 | 回答的问题 |
|---|---|---|
| 训练 | validation loss | 训练分布拟合是否改善 |
| 能力 | math / code benchmark | 目标能力是否提升 |
| 系统 | TTFT / TPOT / p99 | 是否能在线上稳定服务 |
| 成本 | GPU 小时 / 每千 token 成本 | 代价是否可接受 |
| 产品 | 格式遵循率 / 偏好胜率 | 用户体验是否更好 |

## 5. 为什么要区分离线评测和线上评测

离线评测常见优点：

- 便于复现
- 便于回归比较
- 迭代快

但它天然有几个限制：

- 请求分布可能和线上差很远
- 用户交互是多轮的，而离线集常是单轮的
- 真实产品还受到缓存、调度、并发和限流影响

因此一个成熟流程通常是：

1. 先做离线回归
2. 再做 staging / shadow / A-B 验证
3. 最后看线上真实行为指标

## 6. 为什么 CS336 的后半程特别强调这件事

因为到了 evaluation、data、alignment 这些章节，课程已经不再只问：

- 模型会不会做题

而是在问：

- 模型是否能在目标环境下稳定地产生我们需要的行为

这会强迫你把：

- 训练
- 推理
- 数据
- 对齐
- 产品目标

放到同一张系统图里。

## 7. 一个最小评测闭环

可以把它记成：

1. 用训练指标判断训练是否健康
2. 用能力指标判断目标能力是否增强
3. 用系统指标判断是否还能部署
4. 用产品指标判断用户是否真的受益
5. 用分层回归表定位退化到底发生在哪一层

## 易错点

- 把 validation loss 当成最终产品分数
- 把 benchmark accuracy 当成线上体验代理变量
- 只汇报均值，不汇报 p95 / p99 和成本
- 对齐改动后只看行为指标，不看基础能力是否回退

## 排查 checklist

- [ ] 这次改动改善的是训练层、能力层、系统层，还是产品层？
- [ ] 是否存在某一层变好、另一层明显变差？
- [ ] 离线评测集与真实请求分布差异有多大？
- [ ] 是否同时记录了延迟、成本和行为约束？

## CS336 对照

- 官方 lecture 对应：Lecture 12（evaluation）、Lecture 15-17（alignment / RL）
- 推荐搭配阅读：
  - [../models/05-evaluation-and-benchmarking.md](../models/05-evaluation-and-benchmarking.md)
  - [../models/07-post-training-and-alignment.md](../models/07-post-training-and-alignment.md)
  - [../inference/06-observability-and-debugging.md](../inference/06-observability-and-debugging.md)