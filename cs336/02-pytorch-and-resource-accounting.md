# 02｜PyTorch 与资源核算

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec02/>

课程导航：上一讲 [01 课程总览与 Tokenization](01-overview-and-tokenization.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [03 架构与超参数](03-architectures-and-hyperparameters.md)

## 先抓住这讲要点

- 真正的大模型工程，从来不是“能跑就行”，而是先问：**要多少 FLOPs、多少显存、多久能训完**。
- 一个很常用的训练估算规则是：

$$
	ext{Training FLOPs} \approx 6 \times N_{params} \times N_{tokens}
$$

- 显存预算必须同时算：参数、梯度、优化器状态、激活。
- PyTorch 不只是“写模型”的工具，它也是你理解 storage、device、view、copy 和 memory layout 的窗口。

## 代表图

![lec02](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec02/00-53-22-1400.webp)

## 这一讲想训练什么能力

这讲其实在训练一种非常工程化的直觉：

> 在改模型之前，先学会算账。

你需要能快速回答这些问题：

- 这个模型大概多少参数？
- 如果训练 300B tokens，大概多少 FLOPs？
- 80GB H100 上单卡大概能放多大模型？
- 为什么参数明明算着能装下，运行时还是 OOM？
- 为什么 `.view()` 有时不拷贝，`.contiguous()` 却会突然吃内存？

这些问题，全部都属于“资源核算能力”。而这恰恰是很多初学者最缺、工程里最值钱的能力之一。

## 中文解读

### 1. 为什么 CS336 一开始就讲资源核算

因为在大模型场景里，资源约束不是次要条件，而是设计边界。

很多看似“模型问题”的结论，本质上先被资源约束裁剪过：

- 你的 hidden size 能不能再增大？
- batch size 能不能再加？
- Adam 能不能顶得住？
- activation checkpointing 值不值？
- 该不该用混合精度？

所以这讲的核心不是 PyTorch 语法，而是：

> 学会把模型转成 FLOPs、bytes、时间和吞吐。

### 2. 训练时间为什么经常先看 6 倍规则

对一个常见的 dense 线性层，前向大致需要：

$$
2mnp
$$

这里可以理解为：

- 一个乘法；
- 一个加法；
- 对所有位置和参数展开。

反向时通常还要计算：

- 对输入的梯度；
- 对权重的梯度。

这两部分加起来，经常又接近前向的两倍，所以一步训练大致是：

$$
	ext{forward} + \text{backward} \approx 3 \times \text{forward}
$$

如果把 forward 进一步粗略写成“每个 token 大约扫过一遍参数”，就能得到大模型工程里非常经典的经验式：

$$
	ext{Training FLOPs} \approx 6PN
$$

其中：

- $P$ 是参数量；
- $N$ 是训练 token 数。

### 3. 为什么这个公式有用

它最大的价值不是“精确”，而是“快”。

比如你有：

- 7B 参数模型；
- 300B tokens；
- 有效算力 $= 5 \times 10^{14}$ FLOPs/s。

那你可以立刻估算：

$$
6 \times 7 \times 10^9 \times 3 \times 10^{11}
= 1.26 \times 10^{22} \text{ FLOPs}
$$

训练时间大约：

$$
\frac{1.26 \times 10^{22}}{5 \times 10^{14}} = 2.52 \times 10^7 \text{ s}
$$

也就是数百天量级。  
然后你就知道：哦，这不是“我晚上跑一下”的实验，这是要上集群、要并行、要重新估预算的实验。

## 代码拆解：参数与训练时间估算

```python
def estimate_training_time(num_params, num_tokens, peak_flops, mfu=0.5):
    total_flops = 6 * num_params * num_tokens
    effective_flops = peak_flops * mfu
    return total_flops / effective_flops

def estimate_adam_memory(num_params, bytes_per_param=2, master_bytes=4):
    param = num_params * bytes_per_param
    grad = num_params * bytes_per_param
    m = num_params * master_bytes
    v = num_params * master_bytes
    master = num_params * master_bytes
    return param + grad + m + v + master
```

### 这段代码在表达什么

这段代码表达的是一种**资源先行**思维：

- 你还没开始调模型，先知道它是不是训得起；
- 你还没启动集群，先知道大概需要几张卡、几天时间；
- 你还没开 optimizer，先知道显存是不是已经不够了。

### 这里为什么 `bytes_per_param=2`

因为实际训练里，参数经常以 BF16/FP16 参与前向与部分反向。  
但优化器状态常常仍保留 FP32，所以 `master_bytes=4` 比较常见。

这也是混合精度训练的典型模式：

- **计算时尽量低精度**；
- **关键状态尽量高精度**。

## 显存为什么总比你想象中更紧张

很多人第一次估算显存时，只会算：

$$
	ext{params} \times \text{bytes per param}
$$

但真实训练里，通常至少还要加上：

- 参数本体
- 梯度
- optimizer states
- activation
- 临时 buffer / kernel workspace

以 Adam 为例，除了参数本身，还要存：

- gradient
- 一阶矩 $m$
- 二阶矩 $v$
- 有时还会有 master weight

所以很多时候，真正的总显存占用更像：

$$
	ext{Memory} = \text{Params} + \text{Grads} + \text{Opt States} + \text{Activations}
$$

而且 activation 还会随着：

- batch size
- sequence length
- hidden size
- layer 数

一起膨胀。

换句话说：

> “参数能装下”从来不等于“训练能跑起来”。

## PyTorch 视角：tensor 到底是什么

PyTorch tensor 不只是一个数组，它更准确地说是：

- 一块 storage；
- 加上一组 shape / stride / dtype / device 元数据。

这带来一个非常重要的结论：

- 有些操作只改视图，不拷贝数据；
- 有些操作会真的申请新内存。

### 最常见的坑

- `transpose()` 通常只是改 stride；
- `view()` 依赖底层是否连续；
- `.contiguous()` 可能触发真实复制；
- CPU/GPU 之间移动 tensor 往往很贵。

## 代码拆解：view / contiguous 的直觉

```python
import torch

x = torch.arange(12).reshape(3, 4)
y = x.t()                 # 通常是 view，stride 变了
z = y.contiguous()        # 这里可能真实复制一份新内存
```

### 为什么这件事重要

如果你在大张量上频繁触发不必要的 copy：

- 显存会涨；
- 带宽会被吃掉；
- kernel 的输入 layout 也可能变差。

这就是为什么真正做性能优化时，layout 和 contiguity 不是小事。

## Device placement：为什么“在哪儿算”比“怎么算”还关键

在 PyTorch 里，默认 tensor 常在 CPU 上创建。  
如果你之后再把它搬到 GPU，就发生了一次 host-to-device copy。

这在小实验里问题不大，但在训练 loop 里反复做，就是明显开销。

### 一个简单原则

- 能在目标设备直接创建，就直接创建；
- 能减少 CPU/GPU 来回搬运，就尽量减少；
- 不要把 data movement 当成“顺便发生的小事”。

因为很多时候，你以为在优化算力，结果真正卡住你的其实是 PCIe / HBM 带宽。

## Matmul 为什么是核算中心

深度学习大部分重活，本质上都能还原成矩阵乘法。  
对 $(m \times n) \cdot (n \times p)$ 的 dense matmul：

$$
	ext{FLOPs} = 2mnp
$$

这个公式非常值得熟。因为：

- FFN 是 matmul；
- attention 里的 Q/K/V projection 是 matmul；
- output projection 也是 matmul。

所以只要你会算 matmul，你就已经能粗估 Transformer 大头成本。

## MFU 是什么，为什么大家喜欢报它

MFU = Model FLOPs Utilization，大致是：

$$
	ext{MFU} = \frac{\text{measured FLOPs/s}}{\text{theoretical peak FLOPs/s}}
$$

它的意义是：

- 不是问“卡有多强”；
- 而是问“你的模型把卡用得怎么样”。

同一张 H100：

- 如果你的 kernel 形状烂、访存糟、同步多，MFU 就低；
- 如果实现、并行和 shape 调得好，MFU 才会上来。

## 大规模数据加载：为什么会提 memmap

语言模型训练数据往往是超长 token 序列。  
如果你每次都把整个数据文件读进内存，基本不现实。

所以工程里常见方案是：

- 把 tokenized 数据存成大数组；
- 用 `numpy.memmap` 做按需读取；
- 训练时抽取窗口组成 batch。

这背后的本质是：

> 数据系统也要流式化、分页化，不能什么都一次性“全加载”。

## 面试里怎么讲这一讲

如果被问：**“为什么大模型训练要先做资源核算？”**

你可以答：

> 因为大模型实验成本高，很多设计在资源层面就已经不可行了。通常需要先估算训练 FLOPs、有效吞吐、显存占用，再决定模型大小、batch size、精度策略和并行方案。像 6PN 的 FLOPs 规则、Adam 的状态内存和 activation 占用，都是做预算时最先看的量。

如果被问：**“为什么参数装得下，训练还是会 OOM？”**

可以答：

> 因为训练显存不只包含参数，还包括梯度、优化器状态、激活和临时 buffer。尤其 Adam 会额外带来一阶矩、二阶矩和可能的 master weights，而长序列和大 batch 下 activation 也会迅速膨胀，所以只按参数量估算显存通常严重低估真实开销。

## 本讲小结

这一讲想建立的核心能力是：

- 把模型看成 tensor 与 storage 的组合；
- 把训练看成 FLOPs 与 bytes 的消耗过程；
- 把实现细节看成吞吐、显存、时间预算的一部分。

后面讲 Transformer、GPU、并行训练时，你会反复用到这里建立的“算账直觉”。

## 复习题

1. 训练 FLOPs 的 6 倍规则来自哪里？
2. 为什么 Adam 的显存占用比 SGD 大很多？
3. view 和 copy 在 PyTorch 中的区别是什么？
4. 为什么参数大小不足以决定模型能否训练？
5. MFU 在工程上反映的是什么？

## 面试常见题目

1. 如果给你一个 7B 模型和固定 token budget，你会怎么估训练时间？
2. 为什么混合精度能省显存，但不代表所有状态都能低精度保存？
3. activation checkpointing 本质上在拿什么换什么？
4. 为什么 `.contiguous()` 有时会突然导致显存尖峰？
5. 如果单卡总是 OOM，你会按什么顺序排查？

## 面试题答题提示

### 1. 不要只背公式，要说用途

回答资源核算问题时，最好把公式和决策连起来：它帮助你决定模型规模、batch size、并行方案和实验预算，而不只是“知道一个数字”。

### 2. 显存问题一定要分桶讲

尽量拆成参数、梯度、优化器状态、激活、临时 buffer 五类。这样回答会比笼统说“显存不够”更像做过训练系统。

### 3. PyTorch 细节要落到 storage 和 copy

像 `view`、`transpose`、`contiguous` 这些问题，核心不是 API 名字，而是谁会共享底层 storage，谁会触发真实复制。
