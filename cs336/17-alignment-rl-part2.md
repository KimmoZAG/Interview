# 17｜对齐（三）：Policy Gradient 与 GRPO 代码直觉

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec17/>

课程导航：上一讲 [16 对齐 2](16-alignment-rl.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)

## 先抓住这讲要点

- 在 outcome-reward 场景下，语言模型 RL 最朴素的理解就是：**让高 reward 的回答概率变大，让低 reward 的回答概率变小**。
- baseline 的关键作用不是“改目标”，而是**降低梯度方差，让训练不那么发疯**。
- PPO、GRPO 这类方法本质上都在做同一件事：给 policy gradient 加上更多稳定器，避免更新过猛、方差过大和策略崩坏。

## 这一讲在整门课里的位置

第 16 讲我们已经站在方法层面看了 PPO、DPO、GRPO。  
这一讲更进一步，想把很多“看着像黑魔法”的 RL 公式，重新翻译成语言模型训练里更直观的代码和概率更新逻辑。

它回答的是：

> policy gradient 到底在优化什么？baseline 为什么这么重要？clip 和 KL 为什么几乎总是出现？

## 这讲想训练你什么能力

学完这一讲，你应该能：

- 把 policy gradient 看成 reward-weighted log-prob 更新，而不是抽象公式；
- 理解 baseline 为什么不改期望梯度却能显著改善稳定性；
- 看懂 clip、ratio、reference KL 的存在意义；
- 知道 LLM RL 实现里最常见的工程坑出现在哪里。

## 代表图

![lec17](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec17/00-55-58-1400.webp)

## Policy gradient 在语言模型里的最简直觉

如果把一个 prompt $x$ 下整段回答 $y$ 看成一次完整 action，那么我们的目标可以写成：

$$
\max_\theta \; \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[R(x, y)]
$$

它对应的经典梯度形式是：

$$
\nabla_\theta J(\theta) = \mathbb{E}[R(x,y) \nabla_\theta \log \pi_\theta(y|x)]
$$

这串公式初看很抽象，但翻译成人话其实很简单：

> 如果某个回答拿到高 reward，就增加它的概率；如果拿到低 reward，就降低它的概率。

所以语言模型里的 RL，最原始的骨架可以看成：

- 先采样回答；
- 再打 reward；
- 然后按 reward 对 log-prob 做加权更新。

## 为什么它看起来像 reward-weighted SFT

这是一个非常好的入门直觉。  
在最朴素的 outcome-reward 场景里，你几乎可以把 policy gradient 理解成：

- SFT 里，我们提高目标答案的概率；
- RL 里，我们提高“得到高奖励的答案”的概率。

因此很多人会说：

> LLM RL 的最简形式，像是在做 reward-weighted SFT。

这句话不完全严格，但非常有助于建立第一层直觉。

## 代码拆解：naive policy gradient

```python
def pg_loss(logp, reward):
    return -(reward * logp).mean()
```

这几乎就是 policy gradient 的“极简骨架版”：

- `logp` 是当前 policy 对采样答案的对数概率；
- `reward` 决定这个答案该被推高还是压低；
- 负号是因为我们通常写成最小化 loss。

当然，真实实现会复杂很多，但直觉上就是这么回事。

## baseline 为什么这么重要

如果你直接拿 reward 当权重，会很快遇到一个问题：

- 不同 prompt 难度不同；
- reward 尺度差异大；
- 随机采样噪声也很大；
- 梯度会变得非常 noisy。

这时就要引入 baseline。

### baseline 的核心作用

不是改变“什么是好答案”，而是让更新信号更稳定。  
我们把 reward 改成 advantage：

$$
A(x,y) = R(x,y) - b(x)
$$

其中 $b(x)$ 只依赖 prompt / state，不依赖具体采样动作。  
这有个关键性质：

- **不改变期望梯度方向**；
- 但能显著降低方差。

这就是为什么 baseline 在 RL 里几乎是必备稳定器。

## 代码拆解：带 baseline 的版本

```python
def advantage_loss(logp, reward, baseline):
    advantage = reward - baseline
    return -(advantage * logp).mean()
```

这段代码的启发在于：

- 不是问“这个 reward 高不高”；
- 而是问“它比当前上下文里的平均水平高多少”。

这正是 advantage 的精神。

## 为什么 clip 和 KL 几乎总是出现

如果只靠最原始 policy gradient，更新很容易不稳定。  
模型可能因为一次高 reward 样本就大幅偏移，最后学坏得比学好快。

### 1. clip：防止一步迈太大

PPO 里的 clipping 可以粗略理解成：

- 允许更新；
- 但不允许更新比例超出安全范围；
- 防止某一次梯度把策略拉得过猛。

它像给优化过程装了个“限速器”。

### 2. KL：别离参考策略太远

即便 reward 很诱人，也不能让 policy 完全漂走。  
否则会出现：

- 语言风格崩坏；
- 输出分布异常；
- reward hacking 被无限放大；
- 原本有用的 base behavior 被破坏。

所以 reference KL 的作用，是给策略系一根绳子：

> 可以往高 reward 方向走，但别离原来那个人类可接受的区域太远。

## 为什么 old policy 必须冻结

在 PPO / ratio-based 更新里，常会出现 old policy 和 current policy 的概率比值。  
这个 old policy 的角色，是提供一个稳定参考点。

如果 old policy 也在反向传播里动起来，就会出现：

- ratio 失去稳定比较基准；
- 梯度路径混乱；
- 训练逻辑被破坏。

所以工程实现里，old policy 基本都要 `no_grad` 或显式冻结。

## GRPO / PPO 本质上在解决什么

你可以把这些方法统一理解成：

### 朴素 PG 的问题

- 方差大；
- 容易不稳定；
- 容易一步走太猛；
- 稀疏 reward 时更新很脆弱。

### PPO / GRPO 的改进方向

- 用 baseline / value / group baseline 降方差；
- 用 clip 控制步长；
- 用 KL 防止分布漂移；
- 用多样本比较获得更稳的相对信号。

也就是说，这些方法不是在推翻 policy gradient，而是在给它加护栏。

## 工程实现里的几个坑

这部分很值得记，因为真到实现时，往往不是公式难住你，而是这些细节把你坑住：

### 1. old policy 必须 `no_grad`

否则 ratio 相关梯度会出问题。

### 2. reward 很 sparse 时，可能“全军覆没”

如果一个 batch 里几乎没有正向信号，更新就会非常弱，甚至退化成没学到东西。

### 3. rollout 很贵

语言模型 RL 最大的现实成本之一就是 rollout：

- 采样慢；
- 打分慢；
- 序列长时尤其贵；
- 常常需要外层采样、内层多步优化。

### 4. reference / reward / policy 不一致会制造奇怪行为

一旦三者目标不对齐，你会看到：

- 表面高分、实际行为变差；
- 语言风格怪异；
- judge 喜欢但用户不喜欢；
- 模型专门学会骗奖励。

这也是为什么 RL for LLM 永远不能只盯着一个 loss。

## 面试里可以怎么讲

如果面试官问：**“为什么 baseline 不改变期望梯度？”**

你可以答：

> 因为 baseline 只依赖 state/prompt，不依赖具体采样动作，所以它与动作对数概率梯度项结合后在期望上为零。这样不会改变梯度期望方向，但能显著降低方差。

如果面试官问：**“clip 和 KL 在 PPO 类方法里分别干什么？”**

可以答：

> clip 用来限制单次更新幅度，避免 ratio 过大导致训练不稳定；KL 用来约束当前 policy 不要偏离参考模型太远，防止语言分布崩坏和 reward hacking 被放大。

## 复习题

1. 为什么可以把最朴素的 LLM policy gradient 看成 reward-weighted SFT？
2. baseline 的核心价值为什么是降方差，而不是改目标？
3. clip 和 KL 各自控制了哪类不稳定性？
4. 为什么 old policy 必须被冻结？
5. rollout 成本为什么会成为 LLM RL 的现实瓶颈？

## 面试常见题目

1. 为什么 policy gradient 在直觉上像“按 reward 加权的监督学习”？
2. baseline 为什么不会改偏梯度期望？
3. PPO 里的 clip 和 KL 为什么常常要同时出现？
4. old policy 不冻结会发生什么？
5. 为什么 LLM RL 的成本经常卡在 rollout 而不是反向传播？

## 面试题答题提示

### 1. 这讲要讲清统计直觉

不要只背公式，重点是说明 reward、log-prob、baseline、old policy 各自在控制什么不稳定性。

### 2. baseline 的关键词是降方差

它不是在修改目标，而是在不改变梯度期望方向的前提下，让估计更稳定。

### 3. PPO 问题要落到训练稳定性

clip 控制单步更新过大，KL 控制策略分布漂移，这两者共同防止语言模型训练发散或分布崩坏。
