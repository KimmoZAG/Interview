# Post-training 与 Alignment：SFT、偏好优化、RL 的最小框架

## 要点

- 从 CS336 Lecture 15-17 的角度看，预训练模型只是“会续写文本”的 base model，而 post-training 的目标是把它塑造成 **更有用、更稳定、更符合约束** 的 assistant。
- Post-training 不只是“再训练一次”，而是在改变模型输出分布、任务偏好和行为边界。
- 一个足够常见的后训练链路是：**SFT → preference optimization / RL → 评测与对齐回路**。
- alignment 既是行为问题，也是系统问题，因为数据构造、奖励设计、推理采样、评测标准都会反向影响最终结果。

## 为什么预训练后还不够

一个纯预训练模型通常擅长：

- 续写分布内文本
- 模仿训练语料中的模式

但它不一定天然擅长：

- 严格遵循指令
- 保持输出格式稳定
- 避免有害/不合规输出
- 在 reasoning 任务上稳定使用更长推理轨迹

这就是为什么需要 post-training。

## 1. SFT（Supervised Fine-Tuning）

最直观的后训练方式：

- 输入 prompt
- 输出目标 response
- 用监督学习直接让模型拟合目标答案或目标风格

SFT 的价值：

- 快速把 base model 拉到“可用 assistant”区域
- 让模型学会 instruction following、格式遵循、任务风格

局限：

- 它学的是示范分布，不一定真的学会偏好或长期行为稳定性
- 对于复杂 reasoning，可能只学到表面格式

## 2. Preference Optimization

这类方法的核心是：

- 不再直接给唯一标准答案
- 而是让模型学会“这个输出比另一个更好”

常见输入形式：

- 同一 prompt 下的 chosen / rejected response 对

它的价值在于：

- 更直接地注入行为偏好
- 比纯 SFT 更像在学“选择规则”

常见代表包括 DPO 这类直接偏好优化思路。

## 3. RL-based Post-training

再往前走，就是让模型在某种 reward 信号下继续优化。

CS336 后半段关注的重点，不是背所有 RL 公式，而是理解：

- 你要优化的 reward 是什么
- 它是否稳定
- 它会不会诱导 reward hacking

在 reasoning 场景中，reward 有时来自：

- 数学题最终答案是否正确
- 格式是否满足约束
- 更强模型或 verifier 的评分

## 4. Expert Iteration / GRPO 之类方法要抓什么重点

面对这些名字，最重要的不是记缩写，而是问：

- 监督信号来自哪里
- 每轮如何产生新样本
- reward 是否可靠
- 训练是否会把基础能力拉歪

一个够用的理解方式：

- 有些方法更像“不断收集更好的监督样本”
- 有些方法更像“直接在策略分布上做相对优化”

## 5. Alignment 不是只有“安全”

很多人一提 alignment 就只想到安全过滤，但它更宽：

- helpfulness：是否有帮助
- harmlessness：是否减少有害输出
- honesty / calibration：是否少胡说
- format adherence：是否按要求输出结构化结果
- task behavior：是否更愿意展示推理、使用工具、遵守流程

所以更准确地说，alignment 是：

- 让模型行为分布更接近目标产品需求

## 6. 为什么 alignment 是系统问题

这件事不只在 loss function 里发生，还依赖：

- prompt 模板
- 采样参数
- 数据构造方式
- reward / verifier 设计
- 推理阶段是否允许长思维链或多次采样

因此一个后训练结果变差，未必是“算法失败”，也可能是：

- 数据偏了
- reward 漏洞太多
- 评测方式不对
- 推理配置变了

## 7. 一个最小后训练流水线

如果要用最小流程概括，可以写成：

1. 从 base model 出发
2. 用 instruction / reasoning 数据做 SFT
3. 构造 preference 或 reward 信号
4. 用偏好优化或 RL 再训练
5. 用行为评测 + 任务评测 + 安全评测做闭环

这已经足以覆盖大多数现代 assistant 的主线思路。

## 8. Post-training 最常见的风险

### 风险 1：对齐提高了，但基础能力掉了

表现：更会“像助手说话”，但任务正确率下降。

### 风险 2：reward 学歪了

表现：模型开始讨好指标，而不是真的变好。

### 风险 3：评测过于单一

表现：某个对齐指标很好，但线上体验很差。

### 风险 4：推理配置掩盖了训练问题

表现：换了采样参数后，所谓的“对齐效果”突然消失。

## 9. 你至少要会回答的三个问题

### 例 1：为什么 SFT 不等于 alignment 已完成

因为 SFT 主要学的是示范分布，未必足够表达偏好与长期行为稳定性。

### 例 2：为什么 reasoning 后训练常和 verifier / reward 绑在一起

因为仅靠模仿示范轨迹，不一定能稳定提升最终正确率。

### 例 3：为什么 alignment 结果必须和评测一起看

因为它改变的是行为分布，而行为好坏只能通过任务和偏好评测体现。

## 易错点

- 把 alignment 仅理解成安全过滤
- 背算法名字，却说不清监督信号和 reward 从哪里来
- 只看一个 alignment 指标，不看任务正确率和系统代价
- 认为 post-training 与推理配置、评测流程无关

## 排查 checklist

- [ ] 当前阶段是在做 SFT、偏好优化，还是 RL-based post-training？
- [ ] 监督信号或 reward 的来源是否可靠？
- [ ] 是否同时观察了任务能力、行为约束和系统成本？
- [ ] 推理配置变化后，对齐效果是否依然稳定？

## CS336 对照

- 官方 lecture 对应：Lecture 15-17（alignment / RL）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://www.rajdeepmondal.com/blog/cs336-lecture-15
  - https://www.rajdeepmondal.com/blog/cs336-lecture-16
  - https://www.rajdeepmondal.com/blog/cs336-lecture-17