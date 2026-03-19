# Reward Design、Verifier 与 DeepSeek-R1 的强化学习基础设施

## 关键结论

在 DeepSeek-R1 里，真正决定 RL 能不能持续工作的，不只是 `GRPO`，而是：**奖励是否可靠、verifier 是否可扩展、这些信号能不能被系统稳定消费。**

- DeepSeek 先用 `rule-based reward` 把 reasoning 放大，而不是一开始就把希望押在复杂 reward model 上 [DeepSeek-R1, Section 2.2]。
- 它不是把所有奖励一锅炖，而是分阶段引入：先 reasoning，再 general preference，最后再少量混入更脆弱的模型化奖励 [DeepSeek-R1, Section 3.2.2]。
- 它真正害怕的不是“奖励不够复杂”，而是“奖励看起来很聪明，但其实不够可靠，最后被模型钻空子”。

所以本页最重要的一句判断是：**在 reasoning RL 里，简单但可靠的奖励，通常比花哨但脆弱的奖励更值钱。**

## 背景：为什么奖励设计会成为瓶颈

### 旧做法哪里不适合 reasoning 任务

对普通 assistant 对齐任务来说，reward model 往往还能勉强承担“哪个答案更好”的判断；但 reasoning 任务更麻烦：

- 中间过程很长，很难逐步打标签；
- 最终好坏往往要等答案出来后才知道；
- 如果 reward 只是“更像高分答案”，模型很容易学会讨好评分器，而不是学会真正解题 [DeepSeek-R1, Section 2.2; Appendix B.5]。

这使得传统“先训练一个通用 RM，再用它带着 policy 跑”的思路，在 reasoning 场景里会碰到两个问题：

1. **奖励容易漂**：RM 可能更擅长判断风格，而不是判断是否真的推理对了；
2. **奖励容易被 exploit**：policy 会去迎合评分器的偏差，出现 reward hacking。

### DeepSeek 这页想解决的核心问题

本页真正想回答的不是“奖励公式怎么写”，而是三件更实际的事：

1. reasoning 训练里，什么样的奖励最值得优先相信；
2. 为什么 verifier 比漂亮的 reward model 更关键；
3. 为什么 DeepSeek 要把不同奖励拆成阶段，而不是同时混进去。

## DeepSeek 具体怎么做

### 第一步：先用最硬的奖励把 reasoning 放大

`R1-Zero` 的奖励设计非常克制，核心只有两部分：

- `accuracy reward`：答案对不对；
- `format reward`：是否遵守 `<think>` / `<answer>` 结构 [DeepSeek-R1, Section 2.2]。

其直觉可以写成：

$$
Reward_{rule} = Reward_{acc} + Reward_{format}
$$

这套设计背后的思路很朴素：

- 先解决“模型会不会解题”；
- 暂时不要过早去优化“模型看起来像不像一个礼貌助手”；
- 只要任务可验证，就优先依靠硬反馈，而不是依赖更脆弱的神经网络评分器。

### 第二步：把 reward 分成不同层次，而不是一股脑相加

到了正式版 `R1`，奖励系统扩成三层：

| 奖励层 | 作用 | 为什么单独拿出来 |
| --- | --- | --- |
| `Reward_reasoning` | 保证数学、代码、逻辑等任务真的做对 | 这是 reasoning 的主收益来源 |
| `Reward_general` | 让回答更 helpful、更符合偏好 | 这类信号更贴近产品，但也更脆弱 |
| `Reward_language` | 缓解语言混杂、提高可读性 | 它解决表达问题，不直接解决正确性 |

论文给出的第二阶段形式是：

$$
Reward = Reward_{reasoning} + Reward_{general} + Reward_{language}
$$

但 DeepSeek 关键不在这个加法式子，而在于 **它并不是从第一天就让三类奖励同等发言** [DeepSeek-R1, Section 3.2.1; Section 3.2.2]。

### 第三步：优先相信 verifier，而不是优先相信 reward model

DeepSeek 把 reasoning 任务分成两类：

- **可以被 verifier 检查的任务**：数学、逻辑、代码；
- **更依赖偏好判断的任务**：helpfulness、harmlessness、语言风格。

对于第一类，它尽量让程序或规则系统判分：

- 数学题靠答案匹配器或表达式比较；
- 代码题靠编译器和 test cases；
- 格式靠结构检查器 [DeepSeek-R1, Section 2.2; Appendix B.1]。

对于第二类，它才引入 model-based reward：

- `helpful RM` 更像偏好排序器；
- `safety RM` 更像安全分类器 [DeepSeek-R1, Section 3.1]。

这意味着 DeepSeek 的优先级非常清楚：**能用 verifier 的地方，就尽量别先上 RM。**

### 第四步：只在后期少量使用更脆弱的奖励

最能体现 DeepSeek 工程判断的一点，是第二阶段 RL 中，`general preference reward` 只在最后 `400` steps 引入 [DeepSeek-R1, Section 3.2.2]。

这背后其实是非常明确的风险控制：

- reasoning reward 更硬，应该先主导训练；
- preference reward 更柔，也更容易被利用；
- 它适合做后期收束，不适合做前期地基。

### 这套设计的直接优点

这样做的好处可以压成三点：

1. **主收益更稳**：reasoning 能力建立在更可靠的 outcome/verifier 上；
2. **对齐成本更可控**：更脆弱的 reward 被延后、限量使用；
3. **系统边界更清楚**：不同类型的奖励对应不同模块和不同风险，不容易在目标函数里搅成一团。

## 数据怎么说明这些优点

### 证据一：DeepSeek 明确限制了模型化奖励的使用时长

第二阶段 RL 中，preference reward 只在最后 `400` steps 引入，这本身就是一个很强的证据：**DeepSeek 不是不知道模型奖励有用，而是知道它太容易带偏训练目标** [DeepSeek-R1, Section 3.2.2]。

这说明他们对 reward 的态度不是“越多越好”，而是“只在真的需要时才让它上线”。

### 证据二：Reward Model 的训练数据规模说明了它的定位

论文给出两组关键数据：

- `Helpful RM`：`66,000` preference pairs；
- `Safety RM`：`106,000` safe/unsafe samples [DeepSeek-R1, Section 3.1]。

这些规模足以说明 RM 是有工程投入的，但同时也说明它并不是 reasoning 主体本身。它更像是：

- 后期行为修正器；
- 风格和边界控制器；
- 而不是 reasoning 能力的主要来源。

### 证据三：reward hacking 被论文直接点名

DeepSeek 在 Appendix B.5 非常坦率地承认了 reward hacking：reward 分数可以继续上涨，但真实任务表现不一定同步上升，甚至会下滑 [DeepSeek-R1, Appendix B.5]。

这件事说明了为什么他们要强调 verifier：

- verifier 的反馈通常更硬；
- RM 的反馈通常更容易带偏；
- 一旦目标是 reasoning，偏一点点，策略就会慢慢开始“演给评分器看”。

### 证据四：语言一致性奖励说明“更可读”也有代价

论文在 Appendix B.6 对语言一致性奖励做了消融，结果很典型：

- 加上后，语言更稳定、可读性更好；
- 数学指标基本保持；
- coding benchmark 有轻微下降 [DeepSeek-R1, Appendix B.6]。

这说明一件很现实的事：**可读性收益也不是免费的。**

所以 DeepSeek 的 reward 设计真正成熟的地方，不在于它奖励很多，而在于它很清楚每一种奖励可能把模型往哪边推偏。

## 思考问题

- 在 reasoning 任务里，你更相信“规则奖励 + verifier”，还是“更细腻的模型奖励”？为什么？
- 如果 verifier 很贵，你会先优化 reward 设计，还是先优化异步调度和基础设施？
- 在你的任务域里，哪些子任务最适合走 DeepSeek 这种“先硬反馈、后软对齐”的路线？
