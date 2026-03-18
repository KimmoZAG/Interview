# 张量、shape 与内存布局

## 一句话先讲清

推理性能里很多“看起来像 kernel 慢”的问题，根因其实更早：**shape 不合理、layout 不友好、dtype 不合适，导致后面的 kernel 只能替上游背锅。**

一句最值得记住的话是：**shape 决定你算多少，layout 决定你怎么读，dtype 决定你每次读多少字节。**

## 关联知识网络

- 执行模型：[`Kernel 与并行基础（SIMD/SIMT）`](02-kernel-execution-model.md)
- 带宽与 Roofline：[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)
- 图融合：[`计算图、融合与调度`](04-graph-fusion-scheduling.md)
- 量化：[`量化基础（INT8/INT4）与误差`](05-quantization-basics.md)
- 模型侧热点：[`Transformer 推理所需的最小知识`](../03-llm-architecture/01-transformer-minimum.md)

## 为什么值得先学这一页

- 很多性能浪费发生在真正进入热点 kernel 之前。
- 你如果不先看 shape、stride、contiguous，profiling 里只会看到“某个 kernel 很慢”，却看不到它为什么慢。
- 这页是把模型结构翻译成系统执行对象的第一步。

## 一个张量，系统里至少要看哪几层信息

| 属性 | 它回答什么 |
|---|---|
| Shape | 处理的数据规模有多大 |
| Dtype | 每个元素多大、精度和带宽压力如何 |
| Layout / Stride | 元素在内存里怎么排布 |
| Contiguous | 能不能被当成连续块高效消费 |

理解这几件事，主要是为了回答：

- 一个算子到底在处理什么规模的数据？
- 访存是否连续？
- 是否发生了隐式 copy、cast 或 transpose？
- 为什么“数学上一样”的两个实现，真实性能会差很多？

## 为什么 layout 会直接影响性能

layout 最核心地影响三件事：

1. 访存是否连续，是否利于 coalescing
2. 后续算子能否直接消费当前张量
3. 是否需要额外做 materialize / copy / cast

也就是说，layout 不是“实现细节”，它常常决定：

- 一次算子是不是一次真计算
- 还是“先搬一遍内存，再开始计算”

## 在 LLM 里最常见的几个形状直觉

- Prefill：`B × S` 大，并行度更高
- Decode：通常 `S ≈ 1`，更容易被 launch、同步和访存开销主导
- Q / K / V 张量经常在 `[B, S, H]` 与 `[B, S, N_h, D_h]` 之间切换

这些切换本身不一定贵，但一旦和 transpose、pack、kernel 输入要求叠在一起，就会引出很真实的性能差异。

## 最小例子：为什么“结果一样”不等于“成本一样”

假设一个张量逻辑 shape 都是 `[B, S, H]`，但有两种情况：

1. 连续存储，后续 GEMM 可以直接读
2. 做过 transpose 之后只是 view，stride 不再连续，而后续 kernel 又要求 contiguous

第二种情况下，系统很可能会在你没注意到的时候多做一次 copy。

从数学上看，结果一样；从系统上看，第二种可能多出：

- 额外内存流量
- 额外 kernel
- 更差的 cache / coalescing 行为

这就是为什么很多“公式没变”的实现，性能却能差得非常真诚。

## 工程落地：日志里最该打印什么

- 输入 / 输出的 shape、dtype、是否 contiguous
- 关键中间张量（例如 Q/K/V、KV cache）的 shape
- 动态 shape 变化点（batching、padding、不同输入长度）
- 如果可能，也记录 stride，至少记录是否发生了 layout 转换

## Troubleshooting：为什么核心 GEMM 不慢，系统还是吞吐低

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| profiler 上大 GEMM 不慢，但整体慢 | 上游 layout 不友好 | 查关键张量是否 contiguous、是否有隐式 copy |
| 同样模型离线正常，线上慢很多 | batching / padding 改了 shape | 对比线上线下 shape 分布 |
| 某些请求特别慢 | 动态 shape 导致额外 transpose / cast | 记录不同请求的 shape / dtype / stride |
| kernel 次数变多 | materialize / layout 转换增多 | 看 trace 中是否出现额外 copy / cast kernel |

### 一个排障顺序

1. 先问 shape 是否合理。
2. 再问 layout / stride 是否合理。
3. 再看 dtype 是否引入了额外 cast 或带宽负担。
4. 最后才去问 kernel 本身够不够快。

## 推理优化工程师视角

这页最重要的价值，是帮你建立 4 个本能：

1. 性能问题先问 shape 是否合理。
2. 然后问 layout 是否合理。
3. 再问 dtype 是否合理。
4. 最后才去问 kernel 本身够不够快。

很多时候，热点 kernel 只是替上游的 shape / layout 问题在背锅。

## 面试高频问法

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

- “看似没有 copy”，但后续算子要求 contiguous，结果触发隐式 copy
- transpose 后 stride 不友好，导致访存不连续、吞吐骤降
- 只看逻辑 shape，不看真实 layout 和 dtype

## 排查 checklist

- [ ] 能否固定 shape 复现？
- [ ] 是否出现了隐式 layout 转换或 dtype cast？
- [ ] 关键张量是否 contiguous？stride 是否合理？
- [ ] padding、batching 或 transpose 是否改变了后续算子的输入条件？

## 参考资料

- 框架张量布局文档与 profiler 输出
- 建议串读：[`Kernel 与并行基础（SIMD/SIMT）`](02-kernel-execution-model.md)、[`内存层级与性能模型（Roofline）`](03-memory-hierarchy-and-roofline.md)
