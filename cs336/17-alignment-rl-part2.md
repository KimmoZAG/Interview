# 17｜对齐（三）：Policy Gradient 与 GRPO 代码直觉

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec17/>

## 这讲的核心结论

- outcome-reward 场景下，语言模型 RL 可以先看成“对整段回答做 reward-weighted SFT”。
- baseline 的意义不是改目标，而是**降方差**。
- GRPO / PPO 这类方法的本质，都是在想办法让 policy gradient 更稳定、更少发疯。

## 代表图

![lec17](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec17/00-55-58-1400.webp)

## 中文解读

### 1. Policy gradient 在 LM 里的最简形式

把整段回答看成一个 action，目标是：

$$
\max_\theta \; \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[R(x, y)]
$$

梯度形式是：

$$
\nabla_\theta J(\theta) = \mathbb{E}[R(x,y) \nabla_\theta \log \pi_\theta(y|x)]
$$

这看起来就像：reward 高的回答，增加概率；reward 低的回答，减少概率。

### 2. baseline 为什么重要

如果不同 prompt 的绝对 reward 尺度差异很大，梯度会非常 noisy。  
减去一个只依赖 state/prompt 的 baseline，不会改期望梯度，但会显著降方差。

### 3. 为什么要 clip 和 KL

- clip：防止一次更新步子迈太大；
- KL：防止 policy 离参考模型飘太远，出现模式崩坏。

## 代码拆解：naive policy gradient

```python
def pg_loss(logp, reward):
    return -(reward * logp).mean()
```

这就是最原始的形式：reward 当权重，log-prob 当可导对象。

## 代码拆解：带 baseline 的版本

```python
def advantage_loss(logp, reward, baseline):
    advantage = reward - baseline
    return -(advantage * logp).mean()
```

再往上发展，就会进入 GRPO / PPO：

- ratio
- clipping
- reference KL
- inner loop / outer loop rollout

## 工程实现里的几个坑

- old policy 必须 `no_grad`，不然 ratio 梯度会出问题；
- reward 很 sparse 时，采样全错可能直接没更新；
- rollout 成本高，常要外层采样、内层多步优化。

## 复习题

1. 为什么 baseline 不改变期望梯度？
2. clip 和 KL 各自在稳定性上起什么作用？
3. 为什么 old policy 需要冻结？
