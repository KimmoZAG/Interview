# 03｜架构与超参数

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec03/>

课程导航：上一讲 [02 PyTorch 与资源核算](02-pytorch-and-resource-accounting.md)｜课程索引 [00-index](00-index.md)｜学习路线 [study-roadmap](study-roadmap.md)｜面试指南 [interview-prep-guide](interview-prep-guide.md)｜下一讲 [04 Mixture of Experts](04-mixture-of-experts.md)

## 先抓住这讲要点

- 现代 LLM 架构逐渐收敛到一些经验默认值：`Pre-Norm + RMSNorm + RoPE + SwiGLU + bias-free linear`。
- 很多改动不是为了“理论更优美”，而是为了**更稳、更快、更适合大规模训练**。
- 架构选择从来不只影响训练 loss，也会反过来影响推理吞吐、KV cache、大规模并行和稳定性。

## 代表图

![lec03](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec03/00-44-10-1400.webp)

## 这一讲在回答什么

这一讲其实是在回答：

- 为什么今天的大模型结构看起来越来越像？
- 为什么很多小改动会变成“行业默认配置”？
- 架构选择为什么不能只看表达能力，还要看稳定性和系统代价？

一句话概括：

> 现代 LLM 架构不是从理论课本里长出来的，而是从大量训练失败、数值不稳和工程权衡里筛出来的。

## 中文解读

### 1. 为什么现代 LLM 架构越来越收敛

如果你把近几年的主流开源/闭源模型摆在一起看，会发现它们有一种很明显的“收敛感”：

- norm 位置越来越统一；
- 激活函数越来越统一；
- 位置编码越来越统一；
- 线性层里 bias 越来越少；
- attention 结构越来越受推理约束影响。

这不是巧合，而是因为这些设计同时满足了三件事：

1. **训练稳定**
2. **系统友好**
3. **在大规模上反复被验证过**

### 2. Pre-Norm 为什么几乎取代 Post-Norm

Post-Norm 是原始 Transformer 的经典写法，但在深层网络、大 batch、大学习率下更容易训练不稳。  
Pre-Norm 把归一化放在子层前面，保留了更接近 identity 的残差路径，因此：

- 梯度传递更顺；
- loss spike 更少；
- 深模型更容易训起来。

可以把它理解成：

> 残差分支越像“高速公路”，深层网络越不容易堵车。

### 3. RMSNorm 为什么常见

LayerNorm 要做：

1. 减均值
2. 除标准差
3. 再乘可学习参数

RMSNorm 则只保留“按均方根缩放”的部分，不做均值中心化。  
从表达式上可以写成：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

它的好处不是参数省一点，而是：

- 少一些计算；
- 少一些访存；
- kernel 更简单；
- 大规模时更容易吃到稳定吞吐。

在 GPU 上，这类“少一次 memory movement”往往比“少几次 FLOPs”更值钱。

### 4. 为什么很多线性层会去掉 bias

很多现代实现选择 bias-free linear，不是因为 bias 完全没用，而是因为：

- norm 已经在调节统计量；
- bias 的收益常常不大；
- 去掉它可以减少参数、访存和一些数值扰动；
- 对 kernel fusion 也更友好。

它属于那种“单项收益不夸张，但长期看几乎没有负担”的改法。

### 5. RoPE 为什么几乎成标配

RoPE（Rotary Position Embedding）把 query / key 的各个二维子空间做旋转，旋转角度由位置决定。  
这样做以后，注意力内积会天然携带**相对位置信息**。

这比绝对位置编码更受欢迎，原因通常有三点：

- 结构简单；
- 不需要在 embedding 层额外加一份大位置表；
- 对长上下文外推更友好。

你可以把 RoPE 理解成：

> 不直接把“位置编号”塞给模型，而是让向量几何关系本身反映位置差。

### 6. SwiGLU 为什么会替代传统 FFN 激活

传统 FFN 大致是：

$$
\text{FFN}(x) = W_2 \sigma(W_1 x)
$$

而 SwiGLU 这类门控 MLP 更像：

$$
\text{SwiGLU}(x) = W_3(\text{Swish}(W_1x) \odot W_2x)
$$

这里多了一条“门控路径”，让模型能更细粒度地控制哪些通道该放大、哪些该抑制。  
经验上，它经常比普通 GeLU / ReLU MLP 更强。

但要注意：

- 门控会多出一条投影；
- 所以中间维度通常要从 $4D$ 下调，常见近似是 $\frac{8}{3}D$，以保持参数量大致可比。

### 7. 超参数为什么也属于“架构”的一部分

课程里把这些一起讲，是因为很多结构选择和超参数根本分不开：

- 头数怎么分配；
- 每头维度多大；
- FFN hidden size 设成多少；
- vocab 多大；
- 深宽比怎么选。

这些不是纯粹的“调参”，它们直接决定：

- 表达能力；
- kernel 形状；
- 并行效率；
- KV cache 大小；
- 推理 latency。

### 8. 为什么 GQA / MQA 会影响训练时架构选择

以前大家更偏向从“训练表达力”选 attention 结构；现在很多模型会从“推理能不能扛住”倒推架构。

因为自回归推理时，KV cache 是长期驻留显存的大头。  
如果把所有 query head 都各自带一份 K/V，推理代价会很高。

所以：

- MQA：多个 query head 共享一组 K/V；
- GQA：多个 query head 共享少量 K/V 组。

它们做的都是：

> 用一点 attention 表达力上的折中，换大幅推理内存与带宽收益。

## 代码拆解：最小 RMSNorm

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

这里没有减均值，只有尺度归一化。  
工程上它的意义是：**更少的读写，更简单的 kernel，更好的吞吐。**

### 为什么这一点在大模型里重要

单层看似只少了一点点操作；  
但把它乘上：

- 几十层到上百层；
- 每 step 数十亿 token；
- 大规模训练多卡同步。

最后就可能变成非常现实的吞吐差异。

## 代码拆解：SwiGLU 的形状直觉

```python
def swiglu(x, w1, w2, w3):
    a = x @ w1
    b = x @ w2
    return (a * torch.sigmoid(a) * b) @ w3
```

SwiGLU 的关键是“门控”：不是单一路径通过激活函数，而是用另一路信号去控制信息流。  
它通常比普通 FFN 更强，但要配合调整中间维度，比如从 $4D$ 改成大约 $\frac{8}{3}D$。

## 面试里怎么讲这一讲

如果面试官问：**“为什么现代 LLM 常用 RMSNorm + SwiGLU + RoPE？”**

你可以答：

> 因为这些组件在大规模训练里形成了比较稳定的经验最优点。RMSNorm 更省访存、实现更简单；SwiGLU 提升 MLP 表达能力；RoPE 提供相对位置建模且长上下文外推更自然。它们共同特点是，不只是提升效果，也兼顾训练稳定性和系统效率。

如果面试官问：**“为什么 attention 架构会被推理约束反向塑形？”**

可以答：

> 因为推理时 decode 是 memory-bound，KV cache 是长期成本。像 GQA/MQA 这样的设计，核心不是改善训练 loss，而是减少 KV cache 大小和带宽消耗，从而提高部署吞吐和可服务性。

## 本讲小结

这一讲的核心不是背一串术语，而是看懂现代 LLM 结构背后的筛选逻辑：

- 谁更稳；
- 谁更省；
- 谁在大规模上更可维护；
- 谁更适合最终上线推理。

所以，架构选择从来不是“纯模型问题”，它一直都是**模型 + 数值稳定 + 系统成本**的联合优化问题。

## 复习题

1. 为什么现代 LLM 更偏好 Pre-Norm 而不是 Post-Norm？
2. RMSNorm 相比 LayerNorm 省在哪里？
3. 为什么 GQA 是一个推理友好的改动？
4. 为什么 SwiGLU 往往要搭配调整 FFN hidden size？
5. 为什么说现代 LLM 架构是一种经验收敛，而不是单一理论推导？

## 面试常见题目

1. 如果你从头配一个 decoder-only Transformer，哪些默认选项最可能先定下来？
2. RoPE 为什么比绝对位置编码更适合现代长上下文模型？
3. 为什么说很多架构改动其实是在被推理系统反向塑形？
4. GQA、MQA、MHA 的主要 trade-off 分别是什么？
5. 什么时候你会怀疑某个架构改动只是小规模有效？

## 面试题答题提示

### 1. 回答架构问题时，别只讲效果

最好同时讲训练稳定性、实现复杂度和推理代价。现代 LLM 架构的筛选标准从来不是纯 accuracy。

### 2. 讲组件时要讲它解决的旧问题

比如 Pre-Norm 是为了解决深层训练稳定性，GQA 是在解决推理时 KV cache 过大。这样回答会更有因果链。

### 3. “默认配置”本质是经验最优点

别把 RMSNorm、RoPE、SwiGLU 讲成数学真理，更准确的说法是：它们是当前规模、硬件和训练配方下的经验收敛结果。
