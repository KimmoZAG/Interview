# 06｜Kernel、Profiling 与 Triton

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec06/>

课程导航：上一讲 [05 GPU 基础](05-gpus.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [07 并行训练 1](07-parallelism.md)

工程桥接：[`AI Infra / 计算图融合与调度`](../ai-infra/01-operator-optimization/04-graph-fusion-scheduling.md)｜[`AI Infra / 推理优化 Playbook`](../ai-infra/02-inference-engine/05-optimization-playbook.md)｜[`AI Infra / 可观测性与调试`](../ai-infra/02-inference-engine/06-observability-and-debugging.md)

## 核心定义（What & Why）

> **一句话总结**：这一讲讲的不是“怎么炫技手写 kernel”，而是如何先通过 benchmark 和 profiling 找到真实热点，再决定是交给编译器、Triton，还是最后才上 CUDA 手写优化。

## 关联知识网络

- 前置：[`05 GPU 基础`](05-gpus.md)
- 延伸：[`07 并行训练（一）`](07-parallelism.md)
- 延伸：[`10 推理优化`](10-inference.md)
- 平行：[`AI Infra / 计算图融合与调度`](../ai-infra/01-operator-optimization/04-graph-fusion-scheduling.md)
- 平行：[`AI Infra / 推理优化 Playbook`](../ai-infra/02-inference-engine/05-optimization-playbook.md)
- 排障：[`AI Infra / 可观测性与调试`](../ai-infra/02-inference-engine/06-observability-and-debugging.md)

## 先抓住这讲要点

- 不要盲目手写 kernel，先 **benchmark + profile**。
- 很多性能问题来自 elementwise 路径的多次 launch 和多次访存。
- 对大量中小型算子来说，`torch.compile` / Triton 往往已经很强；真要手搓 CUDA，也应该是在 profiling 证明值得之后。
- “我觉得这里慢”和“profile 证明这里慢”之间，往往差着一个数量级的误判。

## 代表图

![lec06](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec06/01-05-40-1400.webp)

## 这一讲在回答什么

如果上一讲是在讲 GPU 为什么会快/慢，那么这一讲就是在回答：

- 你怎么找到真正的瓶颈？
- 什么时候该相信编译器？
- 什么时候该自己写 kernel？
- Triton 这种 DSL 到底解决了什么开发问题？

## 为什么这页特别适合用“小样本”方式学

这页最容易学歪的地方，是把它读成：

- 记几个 profiling 工具名；
- 记 Triton 语法长什么样；
- 记“fusion 会更快”这种正确但没牙齿的结论。

真正更像工程师的读法是：

> 先把热点量出来，再把 launch、访存、融合收益算出来，最后才决定值不值得写 Triton / CUDA。

也就是说，这页不是“会不会写 kernel”的问题，而是“会不会用证据决定要不要写 kernel”的问题。

## 中文解读

### 1. 为什么 profiling 是第一步

很多看起来“复杂”的算子，拆开以后主要时间都花在 GEMM 上；  
也有很多看起来“小”的算子，其实被 kernel launch overhead 和 memory movement 吃掉了。

这就是为什么性能优化常见的第一条经验是：

> 不要对热点拍脑袋，先量出来。

如果不 profile，你会很容易掉进两种错觉：

1. 以为某段 Python 代码慢，实际上慢的是底层 CUDA kernel；
2. 以为某个复杂算子是瓶颈，实际上大头只是几个 elementwise op 的重复访存。

### 2. benchmark 和 profile 不是一回事

- **benchmark**：回答“整体花了多久”；
- **profile**：回答“时间花在哪儿”。

它们必须一起用：

- 只有 benchmark，没有 profile：你知道慢，但不知道为什么慢；
- 只有 profile，没有 benchmark：你知道每个 kernel 长什么样，但不知道整体收益值不值得。

### 3. 为什么 naive elementwise 实现经常很慢

像 GELU 这种函数，数学上看就是一串加减乘除和非线性。  
如果你直接用多行 PyTorch 表达：

- 可能会拆成多个 kernel；
- 每个 kernel 都读写完整 tensor；
- 中间结果多次落回 global memory。

结果就是：

- FLOPs 没多少；
- IO 巨大；
- launch 次数很多；
- 实际速度很差。

这就是为什么 fused kernel 往往能明显提速。

### 一个能现场说清的量化例子：为什么 5 个小 kernel 会把自己拖死

假设一个 `GELU` 路径被拆成 5 个小 kernel：

- 每个 kernel 计算本体只要 `8 µs`
- 每次 launch + 调度额外要 `6 µs`

那么总时间大约就是：

$$
5 \times (8 + 6) = 70\,\mu s
$$

如果把它 fuse 成 1 个 kernel，即便计算本体稍微复杂一点，变成 `25 µs`，总时间也可能只是：

$$
25 + 6 = 31\,\mu s
$$

这时收益并不来自“数学突然变少”，而是来自：

- launch 次数减少；
- 中间张量少写回几次；
- DRAM 往返减少。

这就是为什么面试里如果你只说“fusion 更快”，通常还不够；更好的说法是：

> 对中小型 pointwise 链，launch overhead 和多次访存常常比 FLOPs 本身更致命，所以 fusion 先省的是调度和 IO，再省的是时间。

### 4. Triton 的价值是什么

Triton 把 GPU 编程从 thread 级别提升到 block/vector 级别：

- 更容易写 fused kernel；
- 更容易表达 masked load/store；
- 更容易做 block 级并行；
- 开发效率通常比 CUDA C++ 高很多。

你不需要手工管理大量 thread index 的细节，而是更多地在想：

- 每个 block 处理什么；
- 数据如何按块读取；
- 如何在寄存器里完成局部计算；
- 如何避免无意义的读写。

这让 Triton 特别适合：

- fused elementwise kernels；
- row-wise softmax；
- attention/MLP 周边的小热点。

### 5. `torch.compile` 为什么值得优先试

现代编译器已经能自动做很多事情：

- op fusion；
- kernel selection；
- 形状特化；
- 生成 Triton kernel。

所以工程上一个非常实用的策略是：

1. 先写清晰正确的 PyTorch；
2. 先试编译器/JIT；
3. profile 后再决定是否手写 Triton/CUDA。

因为手写 kernel 的成本不只在“写出来”，更在：

- debug；
- 维护；
- 适配新 shape；
- 兼容新硬件；
- 防回归。

### 6. 什么时候值得手写 CUDA

当且仅当下面三件事同时成立时：

1. 你确认这是热点；
2. JIT / Triton 没打到目标；
3. 你能接受更高维护成本。

换句话说：

> 手写 CUDA 不应该是“默认冲动”，而应该是“最后一公里手段”。

## 对比表：benchmark / profile / compile / Triton / CUDA 分别在干什么

| 手段 | 主要回答的问题 | 优点 | 主要风险 / 代价 |
|---|---|---|---|
| Benchmark | 整体到底慢不慢 | 简单直观，能看端到端收益 | 不告诉你慢在哪 |
| Profile | 时间到底花在哪 | 能定位热点和调用路径 | 只看 profile 容易忽略整体收益 |
| `torch.compile` | 编译器能不能自动融合 / 生成更优 kernel | 成本低，常有惊喜收益 | 某些 shape / graph 不稳定 |
| Triton | 值不值得写块级 fused kernel | 开发效率高，适合中小热点 | 仍有维护与调优成本 |
| CUDA C++ | 最后一公里极致优化 | 自由度最高 | 开发、调试、适配成本最高 |

## 代码拆解：正确 benchmark 的骨架

```python
import time
import torch

def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters
```

关键不是 `time.time()`，而是前后的 `torch.cuda.synchronize()`：  
GPU 是异步执行的，不同步的话，你测到的常常是“CPU 把任务提交出去花了多久”，而不是 kernel 真跑了多久。

### 为什么 warmup 重要

第一次运行时，常常会发生：

- CUDA context 初始化；
- kernel cache 建立；
- JIT 编译；
- lazy allocation。

如果不 warmup，你测到的很可能不是 steady-state 性能，而是“开场动画时间”。

### 一个最容易在面试里踩坑的点：benchmark 不是拿秒表一摁就完了

如果你省掉 `torch.cuda.synchronize()`，测到的往往只是：

- CPU 把任务丢给 GPU 花了多久；
- 而不是 GPU 真的执行完花了多久。

所以更像工程师的表述应该是：

1. 先 warmup，排掉首次编译 / lazy init；
2. timing 边界前后显式同步；
3. 固定 shape、dtype、stream 和输入布局；
4. 同时报 microbenchmark 和端到端 benchmark。

## 代码拆解：Triton 风格思维

```python
# 伪代码：一个 block 处理一段连续数据
offsets = block_start + arange(0, BLOCK_SIZE)
x = load(ptr + offsets, mask=offsets < n)
y = gelu(x)
store(out_ptr + offsets, y, mask=offsets < n)
```

这里最关键的不是语法，而是模式：

- 用向量化 offsets 一次处理一块；
- masked load/store 处理尾部；
- 中间变量尽量留在寄存器。

这和 CUDA 里显式写线程索引不同，Triton 更像让你直接描述：

> 这一块数据怎么被读进来、怎么算、怎么写回去。

## softmax 为什么适合拿来理解 Triton

softmax 是一个非常典型的 GPU 教学案例，因为它既有：

- elementwise 操作；
- reduction（求 max / sum）；
- 数值稳定性要求；
- 很强的 IO 压力。

一个好的 softmax kernel通常会：

1. 一次读入一行；
2. 先做 max reduction；
3. 再做 exp；
4. 再做 sum reduction；
5. 最后一次性写回归一化结果。

也就是说，好的实现会尽量减少“多遍扫描整行”的次数。

## 工程上的推荐工作流

一个非常实用的优化流程是：

1. 先写对；
2. benchmark；
3. profile 找热点；
4. 先试 `torch.compile`；
5. 再试 Triton；
6. 只有真的必要时，才上 CUDA C++。

这个顺序的意义是：

- 降低维护成本；
- 避免过早优化；
- 把工程复杂度花在最值得的地方。

## AI Infra / 性能优化视角

如果把这页和 `05 GPU`、`10 推理优化` 连起来看，会发现它其实在教同一件事：

> 不是“哪种底层技术最酷”，而是“哪个瓶颈最值得你花工程复杂度去打”。

所以一个成熟的优化顺序通常是：

- 先看端到端收益；
- 再看热点是否真在关键路径；
- 然后优先用 `torch.compile` / 现成 fused op；
- 只有在收益足够大且热点稳定时，才考虑手写 Triton / CUDA。

这比“会写 kernel”更重要，因为它决定你是性能工程师，还是只是一个愿意多写几百行底层代码的人。

## 💥 实战踩坑记录（Troubleshooting）

> 现象：自己写了一个 fused kernel，局部 microbenchmark 很漂亮，但模型端到端只快了 1% 左右。

- **误判**：以为只要某个 kernel 提速很多，整体就一定会明显变快。
- **根因**：真实系统的关键路径可能主要还在 GEMM、数据搬运、调度等待，或者这个热点在总时长里占比本来就不大。
- **解决动作**：
    - 先看端到端 benchmark，再看 profile 占比；
    - 确认这个 kernel 是否真的在关键路径上；
    - 比较优化收益和维护成本，不要为了 1% 收益背 30% 复杂度债务。
- **复盘**：性能优化最怕“局部胜利，系统失败”。

> 常见异常：benchmark 数字忽高忽低，重复跑结果不稳定。

- 先检查是否 warmup；
- 再检查 timing 边界是否有 `torch.cuda.synchronize()`；
- 最后确认输入 shape、stream 和编译状态是否一致。

## 面试里怎么讲这一讲

如果面试官问：**“为什么不能一上来就手写 CUDA kernel？”**

你可以答：

> 因为性能优化首先要确认瓶颈，而不是凭感觉写底层代码。很多热点最终集中在 GEMM 或少数高频 elementwise 路径上，现代编译器和 Triton 已经能自动完成很多 fusion 和 kernel 生成。只有在 profile 明确证明热点存在、自动优化不足、并且收益足够覆盖维护成本时，手写 CUDA 才划算。

如果被问：**“Triton 相比 CUDA 的主要优势是什么？”**

可以答：

> Triton 提供的是 block-centric、vectorized 的 GPU 编程抽象，更容易表达 masked load/store、融合的 elementwise 计算和 row-wise reduction。它显著降低了编写高性能 kernel 的门槛，同时在很多场景下能接近手写 CUDA 的性能。

## 本讲小结

这一讲想建立的是“优化纪律”而不是“优化冲动”：

- 先测；
- 再查；
- 再改；
- 优先自动化工具；
- 最后才是手写底层实现。

这比单纯会写 kernel 更重要，因为它能避免你把时间花在错误热点上。

## 复习题

1. 为什么 naive GELU 会慢？
2. `torch.cuda.synchronize()` 在 benchmark 里为什么必要？
3. Triton 相比 CUDA 的主要抽象优势是什么？
4. benchmark 和 profile 的区别是什么？
5. 为什么推荐先试 `torch.compile` 再决定是否手写 kernel？

## 面试常见题目

1. 你会如何定位一个模型里的性能热点？
2. 什么时候适合写 Triton kernel，什么时候不适合？
3. kernel fusion 为什么经常能带来大收益？
4. benchmark 做得不严谨，最容易导致什么错结论？
5. 为什么“会写 kernel”不等于“会做性能优化”？

## 面试题答题提示

### 1. 优化顺序比技巧更重要

先 benchmark，再 profile，再改热点。这个顺序本身就是面试里很重要的工程信号。

### 2. 手写 kernel 不是默认答案

如果 `torch.compile`、现成 fused op、框架优化已经够用，优先用更便宜的方案。手写底层实现应该是后手。

### 3. 讲 Triton 时要落在抽象边界

它的优势是更适合写块级并行、tile 级操作和快速实验，而不是“比 CUDA 高级所以一定更快”。
