# 张量、shape 与内存布局

## 要点

- 推理性能的很多问题，本质是 **shape（维度）+ layout（stride）+ dtype（精度）** 的组合问题
- 任何算子讨论都应先明确：输入/输出 shape、是否 contiguous、是否需要 transpose/reshape
- layout 变化通常意味着额外 copy 或低效访问（coalescing 变差）
- 很多看似“kernel 不够快”的问题，真正根因其实是 shape 或 layout 在上游就已经不对劲了。

## 通用知识

### 它是什么

张量是模型计算的基本对象，但在系统里它不只是“一个多维数组”。

一个张量至少包含几层信息：

- shape：每个维度多大
- dtype：每个元素是什么精度
- layout / stride：这些元素在内存里怎么排布
- 是否 contiguous：是否可被视作连续块

这几件事共同决定了一个算子在硬件上会不会跑得顺。

### 它解决什么问题

理解 shape、layout 和 dtype，主要是为了回答：

- 一个算子到底在处理什么规模的数据
- 访存是否连续
- 是否发生了隐式 copy、cast 或 transpose
- 为什么“数学上一样”的两个实现，真实性能会差很多

### 为什么在 AI 系统里重要

因为很多性能问题都发生在真正进入热点 kernel 之前：

- 维度组织不合理
- transpose / view 之后 stride 奇怪
- 后续算子需要 contiguous，触发隐式 copy
- mixed precision 流程里发生了额外 cast

这些问题如果不先看 shape 和 layout，后面做 profiler 往往只会看到“某个 kernel 很慢”，却看不到它为什么慢。

### 它的收益与代价

收益：

- 能更快判断热点是不是“算子本身问题”
- 能更快发现上游数据组织导致的性能浪费
- 能把模型结构翻译成系统执行对象

代价：

- 需要养成对 shape、stride、contiguous 的敏感度
- 很多问题看起来细碎，但对吞吐和 tail latency 的影响可能非常实在

## 先把“张量”说清楚

- shape：例如 `[B, S, H]`（batch、序列长度、hidden）
- dtype：FP32/FP16/BF16/INT8/INT4
- contiguous：是否连续；stride 是多少
- view vs copy：reshape/transpose 可能只是 view，也可能触发 copy（取决于框架与后续算子需求）

一句很值得记住的话：

- “shape 决定你算多少，layout 决定你怎么读，dtype 决定你每次读多少字节。”

## 推理里常见的关键维度

- LLM：
  - Prefill：`B x S` 大、并行度高
  - Decode：`S≈1`（逐 token），更容易被 launch/同步/访存开销主导

再补一个非常常见的形状对象：

- Q/K/V 张量往往会在 `[B, S, H]` 和 `[B, S, Nh, Dh]` 等形式之间切换

这类切换本身不一定贵，但一旦和 transpose、pack、kernel 输入要求叠在一起，往往就会引出性能差异。

## layout 在系统里到底影响什么

最核心的影响有 3 类：

1. 访存是否连续，是否利于 coalescing
2. 后续算子是否可以直接消费当前张量
3. 是否需要额外做 materialize / copy / cast

也就是说，layout 不是“实现细节”，它常常决定了：

- 一次算子是不是一次真计算
- 还是“先搬一遍内存，再开始计算”

## 最小例子

假设有一个张量逻辑上都是 `[B, S, H]`，但两种情况：

1. 连续存储，后续 GEMM 可以直接读
2. 做过 transpose 之后只是 view，stride 不再连续，而后续 kernel 又要求 contiguous

第二种情况下，系统很可能会在你没注意到的时候多做一次 copy。

从数学上看，两者结果一样；从系统上看，第二种可能多出：

- 额外内存流量
- 额外 kernel
- 更差的 cache / coalescing 行为

这就是为什么很多“明明公式没变”的实现，性能却差得很真实。

## 工程落地：你要在日志里打印什么

- 输入/输出的 shape、dtype、是否 contiguous
- 关键中间张量（例如 Q/K/V、KV cache）的 shape
- 动态 shape 变化点（batching、padding、不同输入长度）
- 如果可能，也记录 stride 或至少记录是否发生了 layout 转换

## 工程例子

一个常见场景：

- 模型离线单测正常
- 线上吞吐却莫名其妙比预期低

后来发现根因不是核心 GEMM 太慢，而是：

- batching 后引入了额外 padding
- 某个中间张量在 reshape + transpose 后不再连续
- 后续 kernel 为了满足输入要求做了隐式 contiguous copy

这类问题如果只盯最重的算子，往往会漏掉真正的起点。

## 推理优化工程师视角

对推理优化工程师来说，这篇最重要的价值是建立 4 个本能：

1. 性能问题先问 shape 是否合理
2. 然后问 layout 是否合理
3. 再问 dtype 是否合理
4. 最后才去问 kernel 本身够不够快

因为很多时候，热点 kernel 只是替上游的 shape/layout 问题在背锅。

## 常见面试问题

### 初级

1. shape、dtype、stride 分别描述什么？
2. 为什么 contiguous 对性能很重要？

### 中级

1. 为什么 transpose / reshape 之后可能触发隐式 copy？
2. 为什么同样逻辑 shape 的张量，性能可能差很多？

### 高级

1. 如果某个 kernel 很慢，你如何判断是 kernel 本身问题还是输入 layout 问题？
2. 在 LLM 推理里，shape 和 layout 常在哪些中间张量上最值得盯？

## 易错点

- “看似没有 copy”但因为后续算子需要 contiguous 导致隐式 copy
- transpose 后的 stride 导致访存不连续，吞吐骤降
- 只看逻辑 shape，不看真实 layout 和 dtype

## 排查 checklist

- [ ] 能否固定 shape 复现？
- [ ] 是否出现了隐式的 layout 转换或 dtype cast？
- [ ] 关键张量是否 contiguous？stride 是否合理？
- [ ] padding、batching 或 transpose 是否改变了后续算子的输入条件？

## 参考

- 框架张量布局文档与 profiler 输出
- 建议串读：`02-kernel-execution-model.md`、`03-memory-hierarchy-and-roofline.md`
