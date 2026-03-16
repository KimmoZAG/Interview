# 10｜推理优化

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec10/>

## 先抓住这讲要点

- 推理优化关心的不是训练 loss，而是服务指标：**TTFT、latency、throughput、并发能力、成本**。
- Prefill 和 decode 是两个不同世界：前者更像大矩阵批处理，后者更像反复读取 KV cache 的带宽战。
- 现代 LLM 推理系统的很多关键创新，本质都围绕三个字：**KV cache**。

## 这一讲在整门课里的位置

训练讲的是“怎么把模型做出来”，推理讲的是“怎么把模型用起来”。  
一个模型哪怕训练得再漂亮，如果上线后：

- 首 token 出得慢；
- 吞吐差；
- 长上下文爆显存；
- 并发一高就抖；

那它在产品上就不够好用。

所以这讲真正回答的是：

> 当模型已经训练完成，如何把它变成一个响应快、成本可控、能支撑真实流量的推理系统？

## 这讲想训练你什么能力

这一讲训练的不是“会调 API 参数”，而是：

- 区分 prefill 和 decode 的性能本质；
- 理解为什么 KV cache 是推理系统设计中心；
- 知道现代推理优化都在减少什么瓶颈；
- 能把模型结构改动（GQA/MLA/CLA）和系统收益连起来看。

## 代表图

![lec10](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec10/00-16-07-1400.webp)

## 推理系统首先关注什么指标

训练阶段最常问的是 loss、MFU、tokens/sec。  
但到了推理阶段，用户和产品最关心的是另一套指标。

### 1. TTFT（Time To First Token）

用户问完问题后，多久看到第一个 token。  
这几乎决定了“体感快不快”。

### 2. Latency

一次请求从进入到完成的总时延。  
对于交互式系统，这决定了可用性；对于链式 agent，这还会层层叠加。

### 3. Throughput

单位时间系统能处理多少 token 或多少请求。  
这是成本和规模化部署的关键。

### 4. 并发与稳定性

现实系统不是只服务一个用户。  
并发一上来，调度、cache 管理、batching 策略都会开始左右整体表现。

所以推理优化不是“让一个请求更快”这么简单，而是：

> 在真实负载下，把交互体验、机器利用率和单位成本同时尽量做好。

## Prefill 和 decode 为什么是两个世界

很多刚接触推理优化的人，会把整个生成过程当成一段连续计算。  
但其实 prefill 和 decode 的性能特征差别很大。

### 1. Prefill：更像大批量并行计算

当用户输入一大段 prompt 时，模型要先把这段前缀全部过一遍。  
这一步通常：

- 序列较长；
- 可以并行处理整段上下文；
- 更容易形成大矩阵乘；
- 因而常常更接近 compute-bound。

### 2. Decode：一次一个 token 的增量生成

真正“慢”的部分通常在这里。  
因为 decode 阶段往往每步只生成一个新 token：

- 批量小；
- 矩阵乘容易退化；
- 计算强度下降；
- 但每一步都要读取已有 KV cache。

所以 decode 很容易从“算不动”转成“搬不动”。

这就是为什么你常听到一句非常重要的话：

> Prefill 更可能偏 compute-bound，decode 更可能偏 memory-bound。

## 为什么 decode 难以吃满 GPU

GPU 喜欢的是：

- 大 batch；
- 大矩阵；
- 高 arithmetic intensity；
- 连续、规则的数据访问。

而 decode 给它的往往是：

- 一次一个 token；
- 需要反复访问越来越长的历史 KV；
- 内存读写占比越来越高；
- 计算量和带宽压力不成比例。

所以很多时候，不是 GPU FLOPs 不够，而是根本没有足够“密”的工作让它持续饱和。

## KV cache 为什么是推理系统的中心

如果没有 KV cache，那么每生成一个新 token，你都得把整个历史前缀重新过一遍注意力。  
这显然太浪费了。

于是我们缓存历史 token 的 Key 和 Value：

- 老的前缀不用反复重算；
- 每次只新增最新 token 的 K/V；
- 大幅节省重复计算。

这就是 KV cache 的根本意义。

但它并不是免费午餐。  
缓存之后，新的问题会立刻出现：

1. **显存占用很大**；
2. **每步都要大量读 cache**；
3. **长上下文下内存分配和碎片变复杂**；
4. **并发请求会让 cache 管理更难**。

所以现代推理系统的大量优化，本质都在回答一个问题：

> 我怎样保留 KV cache 的收益，同时尽量压低它带来的存储和带宽成本？

## 代码拆解：KV cache 更新示意

```python
def append_kv(cache_k, cache_v, new_k, new_v):
    cache_k.append(new_k)
    cache_v.append(new_v)
    return cache_k, cache_v
```

这虽然是伪代码，但很传神，因为它揭示了 decode 阶段的核心行为：

- 读旧的 cache；
- 写新的 cache；
- 再继续下一步。

如果这个过程的数据布局、分配方式、读取方式不够高效，系统很容易出现：

- 带宽被打满；
- allocator 成为瓶颈；
- 长上下文下抖动明显；
- 并发上升时吞吐恶化。

## GQA / MLA / CLA 到底在减少什么

很多结构创新表面上看是“模型设计”，但它们在推理里往往指向同一个目标：**减小 KV footprint**。

### 1. GQA（Grouped Query Attention）

它减少的是 K/V head 数量，让多组 query 共享更少的 K/V heads。  
结果是：

- KV cache 更小；
- 读取压力更低；
- 推理更省显存。

### 2. MLA（Multi-Head Latent Attention / 相关变体思路）

核心直觉是：不直接用高维原始 KV，而是用更紧凑的 latent 表达。  
本质还是在压缩需要长期保存和读取的状态。

### 3. CLA（Cross-Layer / related shared-cache ideas）

如果不同层之间可以共享一部分表示或投影，那就有机会进一步减少 cache 规模。  
这类方法同样是在围绕“历史状态太大”这个问题做文章。

一句话总结：

> 这些方法虽然形式不同，但共同目标都是让每个已生成 token 留下的历史负担更小。

## speculative decoding 为什么能提速

这是推理优化里非常漂亮的一类想法。  
直觉上它像这样：

1. 用一个小模型先猜几个 token；
2. 用大模型并行验证这些猜测；
3. 如果猜对了，就等于一次前进了好几步。

## 代码拆解：speculative decoding 直觉

```python
def speculative_step(draft_model, target_model, prompt):
    draft_tokens = draft_model.sample_many(prompt, k=4)
    verified = target_model.verify(prompt, draft_tokens)
    return verified
```

它真正的收益来自：

- 小模型生成便宜；
- 大模型验证可以更并行；
- 如果接受率足够高，就减少了目标模型逐 token 串行前进的次数。

最关键的一点是：设计得当时，最终采样分布仍可与目标模型直接采样保持一致。  
这也是它比“粗暴近似”更有吸引力的地方。

## 推理优化常见方向，其实都在做三件事

把各种推理系统论文和工程方案放在一起看，会发现大多都在做这三类事：

### 1. 少存

- 压缩 KV；
- 减少 head 数；
- 更紧凑的数据布局；
- 更好的内存分页与复用。

### 2. 少搬

- 减少不必要的数据移动；
- 改进 cache 布局；
- 用更高效的 attention / paged cache 组织方式；
- 降低 decode 阶段的带宽负担。

### 3. 少等

- 更好的 batching；
- request scheduling；
- speculative decoding；
- prefill/decode 分离优化。

所以推理优化不是单点绝招，而是一套围绕**存储、带宽、调度**展开的系统工程。

## 面试里可以怎么讲

如果面试官问：**“为什么 decode 往往比 prefill 更难优化？”**

你可以回答：

> 因为 prefill 通常可以并行处理整段 prompt，更容易形成大矩阵计算，所以偏 compute-bound；而 decode 往往是逐 token 生成，batch 更小，算子更稀疏，同时每步还要读取历史 KV cache，因此更容易受显存带宽和 cache 访问限制，属于更典型的 memory-bound 场景。

如果面试官问：**“现代 LLM 推理优化主要围绕什么展开？”**

可以答：

> 很多优化都围绕 KV cache 展开，因为它同时决定了显存占用、历史状态读取带宽和长上下文下的可扩展性。像 GQA、MLA、CLA、paged KV、speculative decoding，本质上都在减少 KV 相关的存储、搬运或等待成本。

## 复习题

1. TTFT、latency、throughput 在推理系统里分别对应什么体验或成本问题？
2. 为什么 prefill 常偏 compute-bound，而 decode 常偏 memory-bound？
3. KV cache 带来了哪些收益，又引入了哪些新问题？
4. GQA / MLA / CLA 在系统收益上共同指向什么目标？
5. speculative decoding 的速度提升，为什么本质上是在减少目标模型的串行推进次数？
