# RL and Alignment

## 关键结论

DeepSeek-R1 真正重要的，不是“又换了一个 RL 算法名词”，而是它把 **reinforcement learning 从对齐尾声，提前到了 reasoning 能力塑形的主舞台**。

- 它先承认一个现实：只靠 SFT，很难把长链推理稳定放大。
- 它先用 `R1-Zero` 验证 pure RL 能否让 reasoning 自然长出来，再用 SFT 和第二阶段 RL 把行为收回到更可读、可交付的范围内 [DeepSeek-R1, Sections 1-3]。
- 它选择 `GRPO`，不是为了新颖，而是因为 verifier-rich 的长 CoT 任务里，value model 又贵又难学，组内相对比较更实用 [DeepSeek-R1, Section 2.1; Appendix A.3]。

如果用一句话概括本页：**DeepSeek 不是把 RL 当成“最后抛光”，而是把它当成“先把 reasoning 拉上去，再慢慢修成人类可用版本”的主引擎。**

## 背景：为什么要把 RL 提前到训练主线

### 传统做法为什么不够

主流路线通常是：

1. 先用 SFT 把模型训练成一个可用助手；
2. 再用 RLHF 或偏好优化做对齐收尾。

这套方法对“做一个能聊、能跟随指令的 assistant”很有效，但对 **长链 reasoning** 有两个天然问题：

- 人工 CoT 往往只是“人类会怎么写”，不一定是“模型最有效的推理路径”；
- 一旦先用大量 SFT 把推理风格压到某种模板上，后续 RL 的探索空间就会变窄 [DeepSeek-R1, Section 1; Appendix A.2]。

所以 DeepSeek 关注的不是“怎么让模型更像一个好助手”，而是更前面的问题：

> 如果 base model 已经很强，能不能直接用 RL 把 reasoning 行为放大出来？

### DeepSeek 这一页真正想解决什么

这一页要回答三件事：

1. 为什么 DeepSeek 不愿意把 RL 只当作对齐收尾；
2. 为什么 `R1-Zero -> R1` 要走“先放大 reasoning，再补可读性和对齐”的路线；
3. 为什么 `GRPO` 比传统 `PPO + value model` 更适合这类任务。

## DeepSeek 具体怎么做

### 第一步：先让 reasoning 自己长出来

DeepSeek-R1 的第一步不是大规模人工 CoT，而是让 `DeepSeek-V3-Base` 直接进入 pure RL，得到 `R1-Zero` [DeepSeek-R1, Sections 1-2]。

它背后的判断很直接：

- base model 已经有知识和语言底座；
- 数学、代码、逻辑等任务有相对可靠的 verifier；
- 既然结果可验证，就可以让模型通过 trial-and-error 学出更强的中间行为。

`R1-Zero` 因而承担的是“证明 reasoning 能不能被直接诱导出来”的任务，而不是“做一个马上可交付的聊天产品”。

### 第二步：再把 raw reasoning 修成可用模型

`R1-Zero` 虽然能把 reasoning 放大出来，但也带来明显副作用：

- 推理很长；
- 可读性差；
- 语言混杂；
- 对 general helpfulness 支撑不够 [DeepSeek-R1, Section 1; Section 2.3]。

所以正式版 `R1` 走的是分阶段修正路线：

| 阶段 | 作用 | 为什么需要它 |
| --- | --- | --- |
| DeepSeek-V3-Base | 提供知识和语言底座 | 没有强 base model，RL 很难直接放大 reasoning |
| R1-Zero | 用 pure RL 验证 reasoning emergence | 先确认模型能否自己长出长 CoT、反思与验证 |
| Cold-start SFT | 把 raw reasoning 整理成人类更易读的表达 | 只靠 RL，输出太野，不适合直接交付 |
| First RL Stage | 继续强化 reasoning，并加入语言一致性约束 | 保住 reasoning，同时减少语言混杂 |
| Second RL Stage | 引入 helpfulness / harmlessness 等对齐目标 | 让模型不止会解题，也更像一个可用助手 |

这条路线的核心不是“RL 和 SFT 二选一”，而是 **RL 负责发现强推理，SFT 负责把强推理收束成可读、可复用的模型习惯**。

### 第三步：用 GRPO 而不是 PPO

DeepSeek 采用 `GRPO`，关键原因不是新颖，而是它更贴合 reasoning 任务的约束 [DeepSeek-R1, Section 2.1]。

对一类题目采样多条答案后，GRPO 用组内相对好坏来计算 advantage，而不是再训练一个 value model 去预测“这段中间推理未来值多少钱”。

可以把它的直觉写成：

$$
A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

这件事在 reasoning 场景里有三个好处：

1. **省掉 value model**：显存和工程都更轻；
2. **适合 outcome reward**：数学、代码题常常只需要看最终结果是否对；
3. **更适合同题多样本比较**：同一道题里哪个解更好，往往比“全局分数该是多少”更容易判断 [DeepSeek-R1, Appendix A.3]。

### 这条路线带来的直接优点

相比“先大规模 SFT，再保守 RLHF”的传统思路，DeepSeek 这条线最大的优点有三类：

- **推理上限更高**：模型更容易长出反思、验证、回溯这类长链行为；
- **奖励更硬**：在 verifier-rich 任务里，优化目标更接近真实解题质量；
- **训练职责更清楚**：RL 负责探索，SFT 和第二阶段 RL 负责把探索结果变得可读、可对话、可对齐。

## 数据怎么说明这些优点

### 证据一：R1-Zero 证明了 pure RL 确实能放大 reasoning

论文给出的最重要证据不是某一个单项分数，而是 `R1-Zero` 的行为变化：

- thinking time 持续增长；
- 出现自我反思和 alternative exploration；
- 在数学、代码、STEM 等可验证任务上显著提升 [DeepSeek-R1, Section 2.3]。

这说明一件关键的事：**只要 base model 足够强、reward 足够可靠，reasoning 不一定非要先靠人工 CoT 喂出来。**

### 证据二：正式版 R1 不是只会“多想”，而是把行为收束回了更可用分布

R1 并不是停在 `R1-Zero` 那种“会想但不太好用”的状态。论文在 Section 4 给出的结果说明，正式版 R1 在 reasoning 指标之外，也把 instruction following、偏好表现和整体可交付性一起拉了上来 [DeepSeek-R1, Section 4]。

这组证据说明：

- pure RL 负责把 reasoning 上限打开；
- 后续 SFT 和 mixed RL 负责把这个上限收回到用户可用范围内。

换句话说，DeepSeek 不是在赌“pure RL 一步到位”，而是在证明“先放大，再修正”比“一开始就压得很保守”更有效。

### 证据三：为什么说 GRPO 比 PPO 更贴这类任务

论文没有把 `GRPO > PPO` 说成一种普适结论，但它明确给出了 reasoning 场景下的经验判断：

- 长 CoT 里，value estimation 本来就难；
- verifier-rich 任务里，最终 outcome reward 比中间 token 的密集监督更可靠；
- 同题多样本比较天然适合组内相对优势 [DeepSeek-R1, Section 2.1; Appendix A.3]。

因此，GRPO 的收益不是“理论上无敌”，而是 **它刚好更适合 DeepSeek 想打的这类仗**。

## 思考问题

- 如果你的任务没有可靠 verifier，DeepSeek 这种“先 pure RL 再收束”的路线还能成立吗？
- 在 reasoning 训练里，你更担心“模型不会想”，还是“模型会想但收不回来”？为什么？
- 如果把这条路线迁移到 tool use 或 structured output，第一步更该改 reward，还是先改训练环境？
