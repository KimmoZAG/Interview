# 训练资源核算：参数、激活、梯度、优化器状态

## 要点

- 从 CS336 的视角看，训练大模型前必须先算资源账：**显存是否装得下、吞吐是否合理、总 FLOPs 是否可承受**。
- 训练显存的主要构成通常不是只有参数，还包括：**激活、梯度、优化器状态、临时工作区、通信 buffer**。
- 对 decoder-only Transformer，一个足够常用的训练 FLOPs 粗估是：总训练 FLOPs 约为 $6 N T$。
  其中 $N$ 是非 embedding 参数量，$T$ 是训练 token 数。这个估算适合中短上下文的 dense Transformer 做预算级判断。
- 混合精度、activation checkpointing、ZeRO/FSDP、并行切分，本质上都在改这本资源账，而不是“免费加速”。

## 先分三本账

在工程上，至少要把下面三本账分开：

1. 显存账：单卡能不能装下
2. 吞吐账：每秒能处理多少 token / step
3. 总预算账：整个训练 run 需要多少 FLOPs、多少时间、多少卡时

很多讨论会把这三件事混在一起，结果就是：

- 显存够，但吞吐太差
- 吞吐够，但总训练预算完全不可接受
- 预算够，但单卡根本放不下模型与激活

## 1. 参数与权重内存

设模型参数量为 $P$，参数 dtype 字节数为 $b_w$。

则权重本身占用约为：

$$
M_{weights} \approx P \times b_w
$$

常见情况：

- FP32：$b_w = 4$
- BF16 / FP16：$b_w = 2$

如果采用混合精度训练，还经常会额外保留 FP32 master weights，因此真实权重内存可能高于表面 dtype。

## 2. 梯度内存

梯度通常与参数同形状，因此粗估为：

$$
M_{grads} \approx P \times b_g
$$

很多实现里梯度会保存在 FP32 或至少更高精度格式中，所以不要默认它和权重占用完全一致。

## 3. 优化器状态

以 AdamW 为例，常见要保存：

- 一阶矩 $m$
- 二阶矩 $v$

若状态以 FP32 保存，则优化器状态约为：

$$
M_{opt} \approx 2P \times 4
$$

如果还有 master weights，则还可能再加：

$$
M_{master} \approx P \times 4
$$

这也是为什么 AdamW 的训练显存，常常比“模型权重本身”大很多。

一个常见经验：

- 只看权重大小，会严重低估训练显存
- 真正大模型训练里，优化器状态和激活往往才是决定性部分

## 4. 激活内存

激活是训练资源核算里最容易被低估的部分。

对于 decoder-only Transformer，激活内存通常与下列量相关：

- batch size $B$
- sequence length $S$
- hidden size $H$
- layers $L$
- 是否保存 attention 中间结果
- 是否启用 activation checkpointing

只做数量级判断时，可以记住：

- 激活通常与 $B \times S \times H \times L$ 同阶
- 上下文长度和层数一高，激活内存会迅速变大
- attention 中间张量在长上下文下尤其危险

## 5. 临时工作区与通信 buffer

除了上述三大项，真实训练还会消耗：

- GEMM / attention kernel 的 workspace
- NCCL / collectives 的通信 buffer
- dataloader / host-to-device staging buffer
- allocator 碎片带来的额外开销

所以做预算时，不要把单卡显存算到 100%。应保留安全余量，否则 very close to OOM 的训练通常会抖得很难看。

## 一个最小显存心智模型

对单卡训练，最粗略的训练显存可以写成：

$$
M_{train} \approx M_{weights} + M_{grads} + M_{opt} + M_{acts} + M_{workspace}
$$

这不是精确公式，但足够你快速判断：

- 应该先减 batch
- 还是先减 context
- 还是必须上 checkpointing / sharding / 多卡并行

## 训练 FLOPs 的最小估算

CS336 的 scaling 视角里，一个极常用的粗估是：

$$
F_{train} \approx 6NT
$$

其中：

- $N$：模型参数量（通常指非 embedding 参数）
- $T$：训练 token 总数

这个公式适合：

- dense decoder-only Transformer
- 中短上下文
- 做 budget 级规划而不是精确 profiler 计费

不适合直接用于：

- 极长上下文 attention 主导的场景
- MoE 等稀疏结构
- 需要精确到 kernel 级别的实际账单预测

## Activation Checkpointing 在资源账里做了什么

它的本质不是“免费省显存”，而是：

- 少存一部分激活
- 反向传播时重新做部分前向计算

也就是：

- 显存下降
- 计算量上升
- 吞吐通常下降

所以是否使用 checkpointing，不是纯显存问题，而是显存与训练时间之间的 trade-off。

## 混合精度在资源账里做了什么

混合精度最常见的收益：

- 权重、激活、通信可以更省内存
- Tensor Core 利用率更高时吞吐也可能更高

但要注意：

- 优化器状态未必降到同样精度
- 数值稳定性和 loss scaling 仍需关注

## 你至少要会算的三个例子

### 例 1：为什么 context length 翻倍后显存飙升

因为激活和 attention 相关中间量会随 $S$，甚至在某些部分随 $S^2$ 放大。

### 例 2：为什么把优化器从 SGD 换成 AdamW 后单卡突然装不下

因为 AdamW 至少多带了两份与参数同阶的状态。

### 例 3：为什么模型参数明明不大，但训练还是 OOM

因为训练不是只装权重，还要装激活、梯度、优化器状态、workspace 和 buffer。

## 易错点

- 只按模型权重大小估算训练显存
- 只会看“参数量”，不会看 token 数与 context length 对总预算的影响
- 把 checkpointing 当成纯收益，不去计它的重算成本
- 用 $6NT$ 做了预算，却忘了它只是粗估，不是最终账单

## 排查 checklist

- [ ] 你能分别给出 weights / grads / optimizer state / activations 的数量级吗？
- [ ] 当前 OOM 更像是参数装不下，还是激活太大？
- [ ] 如果减 batch、减 context、开 checkpointing，哪个动作最直接？
- [ ] 你的 FLOPs 预算是粗估、理论值，还是 profiler/框架统计值？

## CS336 对照

- 官方 lecture 对应：Lecture 2（PyTorch, resource accounting）、Lecture 9/11（scaling laws）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://rd.me/cs336/lec2
  - https://www.rajdeepmondal.com/blog/cs336-lecture-2
  - https://bearbearyu1223.github.io/posts/cs336-training-a-transformer-lm-part-1/