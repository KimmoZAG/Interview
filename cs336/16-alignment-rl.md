# 16｜对齐（二）：DPO、PPO、GRPO 与可验证奖励

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec16/>

## 这讲的核心结论

- RLHF 的真正难点不是写出目标函数，而是**噪声偏好、奖励错配、过优化、实现复杂度**。
- DPO 之所以火，是因为它把偏好学习改写成了更像监督学习的形式。
- 对数学、代码等可验证任务，reward 来自 verifier 往往比人类偏好更可扩展。

## 代表图

![lec16](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec16/00-29-02-1400.webp)

## 中文解读

### 1. 为什么 PPO 管线重

PPO 通常需要：

- rollout
- reward model
- value model
- advantage estimation
- KL regularization
- clip / ratio 计算

所以它强，但实现成本也高。

### 2. DPO 为什么更工程友好

它绕开了显式 reward model + on-policy RL 的很多复杂环节，直接从 preference pair 出发训练 policy，更接近监督式优化。

### 3. 为什么 verifiable reward 很重要

在数学、代码、可执行任务里，正确与否常常可以自动判定。  
这比“人类觉得哪个好”更便宜、更稳定，也更适合大规模 RL。

## 代码拆解：DPO 的数据视角

```python
pair = {
    "x": "Solve 2+2",
    "y_pos": "4",
    "y_neg": "5",
}
```

你可以把 DPO 理解为：  
**让模型把 `y_pos` 的概率相对 `y_neg` 提高，同时别离 reference policy 太远。**

## 代码拆解：GRPO 的 group baseline 直觉

```python
import numpy as np

def group_advantages(rewards):
    rewards = np.array(rewards)
    return rewards - rewards.mean()
```

对同一个 prompt 采样多条回答，再做组内中心化，能减少“题目难度不同”带来的方差。  
这也是 GRPO 不需要 value model 的关键直觉之一。

## 复习题

1. PPO 相比 DPO 的主要实现复杂度来自哪里？
2. 为什么 reward hacking 会出现？
3. verifiable reward 适合哪些任务，不适合哪些任务？
