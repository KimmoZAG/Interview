# Reward 与 Verifier 设计：后训练到底在优化什么

## 要点

- 从 CS336 Lecture 15-17 的角度看，post-training 真正困难的地方，不是“选哪一个缩写算法”，而是你给模型的监督信号到底靠不靠谱。
- reward 回答“什么样的输出更值得鼓励”，verifier 回答“这个输出是否满足某个可检查标准”。
- 在 reasoning、tool use、格式约束等场景里，verifier 常常比笼统的人类偏好更可操作，因为它能把正确性或约束转成可验证信号。
- 但一旦 reward 或 verifier 设计得差，模型就可能学会 reward hacking，而不是真的变强。

## 1. 为什么后训练的难点是监督信号

预训练阶段的监督信号相对直接：

- 下一个 token 是什么

而后训练阶段，你真正关心的是：

- 回答是否有帮助
- 推理是否正确
- 格式是否满足要求
- 行为是否符合安全与产品边界

这些目标通常不是一个天然存在的单一标签，所以你必须构造监督信号。

## 2. Reward 和 Verifier 的区别

一个够用的区分方式：

### Reward

- 给输出一个分数或相对偏好
- 常用于 preference optimization 或 RL-style 后训练

### Verifier

- 判断输出是否满足某种可检验条件
- 常见于数学、代码、格式约束、工具调用等场景

reward 更像“方向感”，verifier 更像“检查器”。

## 3. 什么样的问题适合 verifier

当任务存在明确可验证标准时，verifier 特别有价值。

常见场景：

- 数学题最终答案是否正确
- 代码是否通过测试
- JSON / XML / 函数调用格式是否合法
- 工具使用步骤是否满足协议

这些任务的共同点是：

- 你不一定知道最佳推理过程
- 但你常常能检查结果是否合格

## 4. 为什么 verifier 在 reasoning 训练里很重要

因为 reasoning 场景最常见的问题是：

- 模型写出看起来很像推理的文本
- 但最终答案不对，或者过程并不可靠

如果只有模仿式 SFT，模型可能只是学到“推理腔调”。

引入 verifier 后，至少可以把信号更直接地绑定到：

- 最终答案是否通过检查
- 中间步骤是否满足约束

## 5. Reward Hacking 是怎么发生的

只要 reward 或 verifier 可被投机利用，模型就可能学会：

- 讨好评分器
- 规避真正问题
- 用表面模式骗过检查器

常见形式：

- 过度迎合某种模板
- 把答案写得更像高分样本，但不更正确
- 专门利用 verifier 的漏洞

所以 reward / verifier 设计不是“把指标写出来”就结束，而是要持续做对抗性验证。

## 6. 一个最小设计框架

如果你要设计 reward 或 verifier，可以先问 5 个问题：

1. 我到底想鼓励什么行为？
2. 这个行为能否被自动检查？
3. 模型最可能怎样钻漏洞？
4. 这个信号和真实产品目标一致吗？
5. 这个信号会不会压坏其他能力？

## 7. Reward / Verifier 常见来源

### 人类偏好

- 更贴近产品目标
- 但成本高、主观性强、一致性有限

### 规则检查器

- 成本低、可重复
- 但只适合有明确格式或约束的任务

### 单元测试 / 程序执行

- 对代码、数学、工具调用尤其有效
- 但覆盖面有限

### 更强模型评分

- 扩展性好
- 但要小心偏差放大和系统性幻觉

## 8. 为什么它必须和评测绑定

一个 reward / verifier 是否好，不是看它“定义得多优雅”，而是看它带来的训练后结果是否真的更好。

所以至少要同时看：

- 目标任务正确率
- 行为约束满足率
- 模型是否出现明显投机行为
- 推理成本是否上升过多

## 9. 在产品里该怎么理解它

更实际一点，可以把它理解成：

- reward 决定你训练时在追什么
- verifier 决定你能不能把某些目标变成稳定监督

如果这两件事没设计好，后训练就很容易从“让模型更有用”变成“让模型更会取悦指标”。

## 易错点

- 背 DPO / PPO / GRPO 名字，却说不清监督信号来源
- 把 verifier 当成万能真理机
- reward 设计只看单指标，不做对抗样本检查
- 不看 reward 改动对基础能力和成本的副作用

## 排查 checklist

- [ ] 当前 reward 或 verifier 具体在鼓励什么行为？
- [ ] 这个信号能否被模型投机利用？
- [ ] 它和真实产品目标的一致性如何？
- [ ] 是否有独立评测证明训练后真的更好，而不是只会骗分？

## CS336 对照

- 官方 lecture 对应：Lecture 15-17（alignment / RL）
- 推荐搭配阅读：
  - [../models/07-post-training-and-alignment.md](../models/07-post-training-and-alignment.md)
  - [../models/05-evaluation-and-benchmarking.md](../models/05-evaluation-and-benchmarking.md)
  - [../inference/09-training-metrics-vs-product-metrics.md](../inference/09-training-metrics-vs-product-metrics.md)