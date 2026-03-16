# 16｜对齐（二）：DPO、PPO、GRPO 与可验证奖励

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec16/>

## 先抓住这讲要点

- RLHF 难的从来不是把目标函数写出来，而是**偏好噪声、奖励错配、过优化、方差控制和实现复杂度**。
- DPO 之所以流行，是因为它把“从偏好学习”改写成了更像监督训练的过程，工程门槛比 PPO 管线低得多。
- 对数学、代码、可执行任务等可验证场景，来自 verifier 的 reward 往往比纯人类偏好更便宜、更稳定、更容易扩展。

## 这一讲在整门课里的位置

第 15 讲回答了：为什么需要 post-training，以及 SFT / RLHF 这条经典路线怎么来的。  
这一讲进一步解决的是：

> 当我们已经有偏好数据或奖励信号之后，具体该怎么优化 policy？为什么 PPO 很重，DPO 为什么更“工程友好”，GRPO 又在改什么？

## 这讲想训练你什么能力

学完这一讲，你应该能：

- 理解 RLHF 真正困难的地方不只是数学，而是系统实现和目标错配；
- 分清 PPO、DPO、GRPO 各自的代价和适用场景；
- 看懂“verifiable reward”为什么正在成为代码、数学等领域的重要方向；
- 知道为什么 reward hacking、过优化、训练不稳定会反复出现。

## 代表图

![lec16](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec16/00-29-02-1400.webp)

## 为什么 RLHF 真正难的不是公式

很多人初看 RLHF，会觉得难点在于符号太多：reward model、advantage、KL、ratio、clip……  
但真正做起来会发现，更难的是这些现实问题：

- 偏好标签本身有噪声；
- reward model 只能近似人类偏好；
- policy 会学会“钻奖励函数空子”；
- on-policy rollout 成本很高；
- 训练方差大、容易不稳定。

所以 RLHF 的难，不只是理论推导，而是：

> 你在用一个不完美的目标，驱动一个强模型持续优化，而它非常擅长把你的漏洞放大。

## PPO 为什么强，但为什么管线很重

PPO 在语言模型 RL 里长期重要，因为它是一个相对成熟、稳健、受控的 policy gradient 路线。  
但它的工程负担确实不轻。

### 一个典型 PPO 管线往往要有

- rollout；
- reward model；
- value model；
- advantage estimation；
- old policy / reference policy；
- KL regularization；
- ratio 与 clipping。

这意味着什么？

- 模块多；
- 状态多；
- 调试面广；
- rollout 成本高；
- 稳定性问题复杂。

所以 PPO 的代价不是“代码长一点”，而是整条训练链条都更重。

## DPO 为什么被很多团队喜欢

如果 PPO 的问题是太重，那 DPO 的魅力就在于：

> 它尽量绕过显式 reward model + on-policy RL 的复杂闭环，直接用偏好对来优化 policy。

这使它在工程上更像监督学习：

- 数据是 preference pairs；
- 优化过程更直接；
- 不需要实时 rollout 那么复杂的环节；
- 调试成本通常更低。

这也是为什么很多团队在有成对偏好数据时，会优先考虑 DPO 一类方法。

## 代码拆解：DPO 的数据视角

```python
pair = {
    "x": "Solve 2+2",
    "y_pos": "4",
    "y_neg": "5",
}
```

从这段数据出发，DPO 的核心直觉可以表述成一句话：

> 对同一个输入，让模型相对更偏好 `y_pos` 而不是 `y_neg`，同时别离参考策略太远。

它妙的地方在于：

- 不用先训练一个显式 reward model 再做 RL；
- 直接把 preference signal 用在 policy 更新上；
- 因而在很多场景下显得更简单、更稳。

## reward hacking 为什么会出现

这是 RLHF 里非常值得警惕的现象。  
一旦你给模型定义了某个 reward，它就会尽力去最大化它。  
问题在于：

- reward 不一定等于你真正想要的行为；
- reward model 可能有漏洞；
- verifier 可能只覆盖部分正确性；
- 某些“看起来高分”的行为，其实不是人真正想要的。

于是模型可能学会：

- 讨好 judge；
- 利用格式偏见；
- 重复某些高奖模式；
- 针对奖励函数投机取巧。

这就是所谓的 reward hacking。  
本质上，它是“代理目标”和“真实目标”不一致的结果。

## 为什么 verifiable reward 很重要

如果任务本身可自动验证，事情会简单很多。  
比如：

- 数学题是否算对；
- 代码测试是否通过；
- SQL 是否执行出正确结果；
- 某些结构化任务是否满足约束。

这些场景里的 reward 不再完全依赖“人类觉得哪个好”，而可以来自：

- 单元测试；
- 执行结果；
- 规则校验器；
- 程序化 verifier。

这有几个巨大优势：

- 更便宜；
- 更一致；
- 更易扩展；
- 更适合大量 rollout。

所以在代码、数学等任务上，verifiable reward 往往比纯偏好信号更适合做大规模 RL。

## GRPO 在改什么

GRPO 这类方法的重要动机之一，是想减少 PPO 一些重组件的依赖，同时尽量保持可用的优化信号。  
其中很关键的一个直觉是：

> 对同一个 prompt 采样多条回答，组内比较谁更好，再做中心化，就能在不显式训练 value model 的情况下获得一个相对 advantage 信号。

## 代码拆解：GRPO 的 group baseline 直觉

```python
import numpy as np

def group_advantages(rewards):
    rewards = np.array(rewards)
    return rewards - rewards.mean()
```

这段代码非常短，但很有启发性。  
它说明了 GRPO 里的一个核心想法：

- 同一个题目内部比；
- 用组均值做 baseline；
- 这样能减少不同 prompt 难度差异带来的方差。

也正因为如此，GRPO 在某些任务上能够在不引入完整 value model 的前提下，得到相对可用的优势估计。

## 一个更现实的选择框架：PPO、DPO、GRPO 怎么选

可以粗略这样理解：

### PPO

- 更完整；
- 更传统 RL；
- 控制手段丰富；
- 但工程最重。

### DPO

- 偏好数据驱动；
- 更像监督学习；
- 管线更轻；
- 适合已有高质量 preference pairs 的场景。

### GRPO / 类似 group-based 方法

- 试图在 RL 场景里减少一部分复杂组件；
- 更依赖组内相对比较；
- 对可采样、可打分任务尤其有吸引力。

所以不是谁永远更先进，而是：

> 你的奖励从哪里来、系统复杂度能承受多少、任务是否可验证，这些才决定方法选择。

## 面试里可以怎么讲

如果面试官问：**“为什么 PPO 在 LLM 对齐里实现成本高？”**

你可以答：

> 因为 PPO 不只是一个损失函数，它需要 rollout、reward model、value model、advantage estimation、KL 控制、ratio/clipping 等一整套 on-policy RL 管线，所以模块多、计算重、调试难，整体工程复杂度很高。

如果面试官问：**“为什么 DPO 更工程友好？”**

可以答：

> 因为 DPO 直接从 preference pairs 优化 policy，绕开了显式 reward model 和复杂的 on-policy RL 闭环，使训练更接近监督学习流程，因此更轻量、实现和调试成本也更低。

## 复习题

1. RLHF 真正难的地方为什么不只是目标函数本身？
2. PPO 相比 DPO 的主要复杂度来自哪些额外组件？
3. reward hacking 的根源是什么？
4. verifiable reward 为什么特别适合数学、代码这类任务？
5. GRPO 的 group baseline 在直觉上帮助解决了什么问题？
