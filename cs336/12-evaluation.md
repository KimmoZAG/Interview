# 12｜评测：指标、协议与陷阱

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec12/>

## 这讲的核心结论

- 评测没有“绝对分数”，只有**带上下文的信号**。
- 先问“谁在用这个结果”，再决定测什么。
- 输入分布、prompt 协议、评分方式、解释框架，这四层都会改变最终结论。

## 代表图

![lec12](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec12/00-10-40-1400.webp)

## 中文解读

### 1. 为什么 benchmark 分高，不代表“更强”

因为你不知道它是不是：

- prompt 工程更强；
- 在该 benchmark 上污染更重；
- 用了工具、检索、CoT；
- 牺牲了成本、延迟、稳定性。

所以评测要说明“测的是模型、方法，还是完整系统”。

### 2. perplexity 仍然重要，但不够

perplexity 是细粒度、连续型信号，非常适合：

- scaling law 分析；
- 训练调参；
- 数据混配比较。

但它不能完全代表 instruction following、agent 能力和安全性。

### 3. 开放式评测为什么难

因为很多任务没有唯一标准答案。  
这就需要：

- human preference；
- model-as-a-judge；
- rule-based constraints；
- task-specific verifier。

每一种都有偏差来源。

## 代码拆解：一个评测配置至少该包含什么

```python
eval_config = {
    "input_distribution": "mmlu_zero_shot",
    "invocation": "direct_answer",
    "tool_use": False,
    "metric": "accuracy",
    "decontamination": True,
}
```

这段配置表达的是一个很重要的观念：  
**没有协议说明的分数，基本不值得信。**

## 复习题

1. 为什么说 benchmark score 是 context-dependent signal？
2. perplexity 适合什么，不适合什么？
3. 评测时为什么必须写清楚 invocation protocol？
