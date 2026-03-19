# DeepSeek-R1 的 RL Infrastructure：如何把长 CoT 强化学习真正跑起来

## 关键结论

如果只看算法，DeepSeek-R1 似乎只是“用了 GRPO”；但如果真正去想它怎么训练，会发现更关键的问题是：**长 CoT、组采样、verifier、reward model 和参数更新，怎么在同一套系统里不互相拖死。**

- DeepSeek 把 RL 明确拆成 `Rollout / Inference / Rule-based Reward / Training` 四个模块 [DeepSeek-R1, Appendix B.1]。
- 它真正优化的不是单模块速度，而是整条闭环吞吐：谁会拖后腿，就想办法把谁异步隐藏起来。
- 它把显存也当成调度问题，而不是默认让所有模型一直常驻 GPU。

所以这页最值得记住的一句话是：**DeepSeek 不是把 RL 当一个 loss function 在跑，而是把它当成一个跨生成、判分、验证、训练和显存调度的全栈系统问题。**

## 背景：为什么普通后训练系统不适合长 CoT RL

### 旧有系统为什么会卡住

常规 SFT 或短输出 RLHF 的系统，通常假设：

- 样本长度还算可控；
- 奖励计算主要靠模型前向；
- verifier 不会特别慢；
- actor、reference、reward model 的切换开销还能忍。

但到了 DeepSeek-R1 这种长 CoT reasoning 训练，这些假设都开始失效：

1. **样本不是现成的，要在线生成**：一题不只一条答案，而是一组答案；
2. **奖励不只是一层前向**：还要跑格式检查、答案匹配、编译器、test cases；
3. **长输出把所有问题放大**：padding 更多、长尾更长、显存更紧、切换更频繁 [DeepSeek-R1, Section 2.1; Appendix B.1]。

### DeepSeek 这页想解决什么

这一页不讲“为什么要做 RL”，而是只讲一个更现实的问题：

> 当你已经决定要做长 CoT RL 之后，怎样才能把它真的跑到足够大、足够稳、足够快？

## DeepSeek 具体怎么做

### 第一步：把 RL 闭环拆成四个模块

DeepSeek 在附录里把 RL framework 明确拆成四块 [DeepSeek-R1, Appendix B.1]：

| 模块 | 主要职责 | 为什么必须拆开 |
| --- | --- | --- |
| Rollout Module | 用 actor 采样多条输出 | 生成侧要追求高吞吐，不该被训练逻辑拖住 |
| Inference Module | 跑 reward model / reference model 前向 | 这是 GPU 型打分任务，和 rollout 的负载不同 |
| Rule-based Reward Module | 跑编译器、答案匹配器、格式检查器等 verifier | 这类工作更异构、延迟长尾更重 |
| Training Module | 计算 loss 并更新策略 | 它关心的是 batch 组织、反向传播和负载均衡 |

这不是“软件工程上的优雅”，而是因为四类工作负载本来就不是一回事：

- 有的偏生成；
- 有的偏 GPU 前向；
- 有的偏 CPU / 外部工具；
- 有的偏训练更新。

不拆开，最后一定是大家一起堵车。

### 第二步：优先解决长尾 verifier，而不是只盯 GPU

对 reasoning RL 来说，最容易被低估的一环恰恰是 `Rule-based Reward Module`：

- 数学题要答案解析；
- 代码题要编译和 test cases；
- 格式检查虽然快，但只是其中一小部分 [DeepSeek-R1, Appendix B.1]。

这类 verifier 的共同问题是：

- 延迟分布长尾；
- 很难统一 batch 化；
- 最慢样本特别容易拖住整批训练。

所以 DeepSeek 采取的核心动作是 **异步重叠**：

- rollout 在继续采样；
- inference 在继续打分；
- verifier 在后台慢慢跑；
- training 不必等所有慢检查都串行结束后才开始下一轮。

这一步的意义不是“再多压 10% 性能”，而是防止整个系统退化成“GPU 等 CPU，训练等 verifier”。

### 第三步：把 rollout 当成专用高吞吐服务来做

DeepSeek 的 rollout 不是普通线上服务，而是“为 RL 采样服务”的生成系统：

- 用 `vLLM workers` 提高多样本吞吐；
- 在 MoE 架构下继续复用 expert parallel；
- 为热点专家做冗余部署；
- 用 `MTP` 做 self-speculative decoding，缩短长尾样本完成时间 [DeepSeek-R1, Appendix B.1]。

这说明一个很重要的工程判断：

**RL rollout 的目标不是让单个用户尽快拿到一个答案，而是让训练系统尽快拿到一大批可比较的候选答案。**

### 第四步：把训练瓶颈从“反向传播”扩展到“样本组织”

长 CoT 训练时，很多时候最浪费的不是反向传播本身，而是 padding。

DeepSeek 的解决办法非常直接：

1. 全局 batch 先按长度排序；
2. 在 data parallel group 内分发；
3. 每个进程内部再用 `Best-Fit` 把样本打包进固定长度 chunks；
4. 最后对齐 chunk 数，避免有些设备一直拿到更长样本 [DeepSeek-R1, Appendix B.1]。

换句话说，它不是只想着“怎么训练”，而是先想“这些超长样本怎样装车最不浪费”。

### 第五步：把显存也当成调度系统的一部分

DeepSeek 没有让 actor、reference、reward model 永远一起常驻 GPU，而是做了模块级 `offload / reload` [DeepSeek-R1, Appendix B.1]。

它的思路很像轮班：

- rollout 时，actor 优先占显存；
- inference 时，reference / reward model 上场；
- training 时，再把训练所需实例调回来。

这背后的收益非常实际：

- 峰值显存更低；
- 同一批 GPU 能跑更长序列；
- RL 不必因为“模型都想常驻”而直接炸显存。

### 这套设计带来的直接优点

DeepSeek 这套基础设施的收益，可以压缩成四点：

1. **吞吐更稳**：最慢 verifier 不再轻易拖死整条闭环；
2. **资源更省**：显存从“大家同时占”变成“谁干活谁上”；
3. **长序列更可训**：长度排序和 packing 直接减少了 padding 浪费；
4. **系统资产可复用**：MoE、DualPipe、MTP 等预训练期能力没有被浪费，而是继续服务 RL。

## 数据怎么说明这些优点

### 证据一：采样规模已经远超“普通后训练”

论文给出的第一阶段 RL 配置非常能说明问题 [DeepSeek-R1, Section 2.1]：

- 每题采样 `16` 个输出；
- 每步 `32` 个问题；
- 所以单步 batch size 是 `512`；
- 每次 rollout 会产生 `8,192` 个 outputs；
- 再拆成 `16` 个 minibatches，只做 `1` 个 inner epoch。

这组数字说明 DeepSeek 走的是很典型的：

- **大规模采样**；
- **相对克制更新**；
- **先把高质量比较样本收上来，再消化**。

如果基础设施扛不住，这套配置根本跑不起来。

### 证据二：长 CoT 规模直接把系统问题推到前台

论文还给出 rollout 最大长度：

- 前期最大 `32,768` tokens；
- 后期提升到 `65,536` tokens [DeepSeek-R1, Section 2.1]。

这几个数字的含义非常直接：

- 一旦序列这么长，padding、显存、长尾延迟都会立刻成为主问题；
- 这时如果还沿用“普通微调脚本 + 顺序判分”的思路，系统会很快卡死。

所以这些数字本身就是证据：**DeepSeek 必须把 RL 变成独立基础设施问题，而不是在原有后训练管线后面随便加一层。**

### 证据三：reference model 的更新节奏也体现了系统权衡

论文提到 reference model 每 `400` steps 用最新 policy 替换一次 [DeepSeek-R1, Section 2.1]。这说明 DeepSeek 对系统节奏也有明确判断：

- 不能每一步都激进同步，否则代价太高；
- 也不能长期不动，否则 KL 约束会逐渐失真。

这类细节看起来小，但恰好说明它不是“只会堆硬件”，而是在认真做闭环节奏控制。

## 思考问题

- 在这套 RL 基础设施里，你觉得真正的第一瓶颈更像 rollout、verifier，还是显存切换？为什么？
- 如果没有模块级 offload / reload，DeepSeek 这种长 CoT RL 还剩多大现实可行性？
- 如果把这套系统迁移到 tool use 或 software engineering RL，你最先想改的是哪一块？
