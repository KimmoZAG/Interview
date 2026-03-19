# DeepSeek-R1 Distillation：如何把长链推理能力迁移到更小模型

## 关键结论

R1 的 distillation 回答的是一个非常现实的问题：**当大模型已经通过 RL 学会更强 reasoning 之后，这种能力能不能被迁移成更小、更便宜、更易部署的模型资产。**

- DeepSeek 的回答是：可以，而且不该让每个小模型都重新支付一次昂贵 RL 成本 [DeepSeek-R1, Section 1; Section 6]。
- 被迁移的不是短答案本身，而是 reasoning style、problem decomposition、reflection 和更适合交付的表达方式 [DeepSeek-R1, Appendix B.3.2; Appendix B.3.3]。
- Distillation 在这里不是附送动作，而是 reasoning 能力产品化和开源化的关键出口。

所以本页最重要的结论是：**大模型负责昂贵探索，小模型负责廉价继承。**

## 背景：为什么强 reasoning 还需要蒸馏

### 旧做法为什么不够

如果一套强 reasoning 能力只能存在于超大模型上，那它的部署成本、推理成本和普及成本都会很高。更麻烦的是，如果每个中小模型都要重新跑一遍大规模 RL，那成本会再次爆炸。

所以 reasoning 训练天然分成两个阶段：

- **发现阶段**：用大模型 + RL 找到更强推理行为；
- **扩散阶段**：把这些行为迁移给更小、更便宜的模型。

旧式蒸馏常常更关注“答案像不像 teacher”，但 reasoning 场景更难：如果只蒸馏最终答案，真正有价值的中间行为很容易丢失。

### DeepSeek 这页想解决什么

这一页真正要回答三件事：

1. 为什么 R1 的 reasoning 不能只停留在大模型本体上；
2. DeepSeek 到底蒸馏了什么，不蒸馏什么；
3. 论文里的数据怎样说明 reasoning 能力确实可以被迁移。

## DeepSeek 具体怎么做

### 第一步：先让大模型去做昂贵探索

DeepSeek 没有试图让小模型直接承担 reasoning 发现任务，而是先让大模型完成这件事：

- `DeepSeek-V3-Base` 提供强底座；
- `R1-Zero` 与 `R1` 通过 RL 找到更强的 reasoning 轨迹；
- 然后再把这些高质量轨迹整理成可监督学习的数据 [DeepSeek-R1, Sections 1-4]。

这样做的核心好处是：最贵的探索过程只做一次。

### 第二步：蒸馏的不是终点答案，而是整条 reasoning 轨迹

DeepSeek 的蒸馏并不是“把最后答案抄给 student”。它真正迁移的是：

- problem decomposition；
- reflection / verification 行为；
- 更像解题过程的中间推理；
- 更适合人类阅读和交付的表达风格 [DeepSeek-R1, Appendix B.3.2; Appendix B.3.3]。

这就是为什么 distillation 数据在进入 student 之前，还要经过：

- rejection sampling；
- 风格清洗；
- 语言混杂过滤；
- human-friendly 改写。

换句话说，DeepSeek 蒸馏的是“可教的 reasoning 语料”，而不是“原始 RL 痕迹”。

### 第三步：把同一套 reasoning 迁移到不同模型族和参数规模

论文没有只给一个蒸馏样板，而是覆盖了多个模型族和参数带 [DeepSeek-R1, Appendix B.4.3]：

| 模型族 | 覆盖规模 |
| --- | --- |
| Qwen | 1.5B、7B、14B、32B |
| Llama | 8B、70B |

这说明 DeepSeek 想验证的不是“某一个 student 能不能学会”，而是：

- reasoning transfer 能否跨参数规模成立；
- reasoning transfer 能否跨 base family 成立；
- 强 reasoning 是否能变成一个可复用的模型资产，而不是一版一次性的实验结果。

### 这条路线带来的直接优点

Distillation 在这里有三类直接收益：

1. **部署门槛更低**：强 reasoning 不再只绑定超大模型；
2. **训练经济学更好**：昂贵 RL 被大模型集中支付，小模型只做监督继承；
3. **开源扩散更容易**：可以对外发布多个尺寸的 distilled checkpoints，让更多用户在不同硬件预算下使用。

## 数据怎么说明这些优点

### 证据一：蒸馏数据规模本身就说明这不是“顺手微调”

论文给出 distillation 使用约 `800k` supervised data，其中：

- reasoning-related data 约 `600k`；
- non-reasoning data 约 `200k` [DeepSeek-R1, Appendix B.3.3; Appendix B.4.3]。

这说明 distillation 在 DeepSeek 这里不是附带动作，而是一条明确的后续训练路线。

### 证据二：训练配置说明它在学习长 reasoning，而不是短答案模仿

论文给出的 distillation 配置包括 [DeepSeek-R1, Appendix B.4.3]：

- 最大 context length：`32,768`；
- batch size：`64`；
- 训练时长：`2-3 epochs`；
- scheduler：cosine decay。

这组数字说明 student 学的不是短输出风格，而是 **长上下文下的完整 reasoning 轨迹**。

### 证据三：论文明确把 distilled models 作为重要交付结果

在摘要、正文和附录里，DeepSeek 都反复强调 distilled models 的价值：小模型在数学、代码和 STEM reasoning 上明显强于各自原始 instruction-tuned 对应物 [DeepSeek-R1, Abstract; Section 4; Appendix D.1]。

这意味着 Distillation 证明的不是“teacher 很强”，而是另一件更关键的事：

> 强 reasoning 能力可以被迁移，而不必永远锁在 frontier model 身上。

## 思考问题

- 如果 student 足够小，蒸馏过去的更像“真实推理能力”，还是“像推理一样说话的风格”？
- 你会优先保留最终答案、原始长链推理，还是经过整理的 human-friendly reasoning？为什么？
- 对开源生态来说，真正的拐点是 frontier model 首次学会推理，还是强推理第一次被稳定蒸馏到小模型？
