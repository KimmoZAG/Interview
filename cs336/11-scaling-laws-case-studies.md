# 11｜Scaling Law 案例：MUP、WSD 与工程外推

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec11/>

## 这讲的核心结论

- 真正的 scaling workflow 不只是“画条线”，而是：**稳定参数化 + 合理 LR schedule + 小模型代理实验 + 大模型验证**。
- MUP 的目标是让超参数在不同宽度下更可迁移。
- WSD（Warmup-Stable-Decay）让数据 scaling 和 checkpoint rewinding 更实用。

## 代表图

![lec11](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec11/00-21-16-1400.webp)

## 中文解读

### 1. MUP 在解决什么问题

普通参数化下，模型变宽后最优学习率经常漂移。  
MUP 试图通过初始化和每层 LR 缩放，让“在小模型上调好的超参”更能迁移到大模型。

### 2. WSD 为什么适合 scaling 实验

WSD 把训练分成：

1. warmup
2. stable plateau
3. rapid decay

这样做的好处是：你可以在同一条长训练曲线上，通过 checkpoint rewind 近似不同 token budget 的效果，节省大量重复实验成本。

### 3. 为什么不同论文的 tokens/params 比例差别这么大

因为它高度依赖：

- 架构
- 数据质量
- LR schedule
- 拟合方法
- 评测口径

所以 `20:1` 不是圣经，更像一个时代性的经验值。

## 代码拆解：WSD schedule 示意

```python
def wsd(step, warmup, stable_end, total, lr):
    if step < warmup:
        return lr * step / warmup
    if step < stable_end:
        return lr
    ratio = (step - stable_end) / (total - stable_end)
    return lr * (1 - ratio)
```

这个 schedule 的核心不是优雅，而是工程可控：  
plateau 阶段给你稳定观察区，decay 阶段给你收敛收口。

## 复习题

1. MUP 试图改善哪类超参迁移问题？
2. WSD 相比传统 cosine schedule 的工程优势是什么？
3. 为什么不同模型的最优 token/param 比例会不同？
