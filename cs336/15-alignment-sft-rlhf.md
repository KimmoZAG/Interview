# 15｜对齐（一）：SFT 与 RLHF 全景

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec15/>

## 这讲的核心结论

- 预训练模型不等于可用助手，真正变成 ChatGPT 风格系统，往往要靠 post-training。
- SFT 负责把“会语言”变成“会按指令说话”；RLHF 则继续把“像人喜欢的样子”推得更远。
- 高质量 SFT 数据并不总是越“华丽”越好，过度展示 citation / style 可能反而教出表面模仿。

## 代表图

![lec15](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec15/00-47-54-1400.webp)

## 中文解读

### 1. InstructGPT 三步走

1. SFT：给示范答案
2. Reward Model：学偏好
3. RL：优化策略

这是后训练时代最经典的一条主线。

### 2. 为什么 instruction data 不是越多越好

因为小而精的数据对行为有很强杠杆。  
如果数据里满是“看起来高质量、实际 base model 并不具备支撑能力”的答案，就可能把模型教成会模仿格式、不会保证真实性。

### 3. 为什么 RLHF 吸引人

因为让人类“写标准答案”太贵；  
让人类“二选一判断哪个更好”往往更便宜，也更稳定。

## 代码拆解：SFT 的最小损失

```python
def sft_loss(logits, labels):
    # 常规 next-token cross entropy
    return cross_entropy(logits[:, :-1], labels[:, 1:])
```

SFT 本质上还是监督学习，只不过数据从普通文本变成了 prompt-response 格式。

## 代码拆解：pairwise preference 的数据形式

```python
sample = {
    "prompt": "Explain attention",
    "chosen": "Good answer...",
    "rejected": "Bad answer...",
}
```

这就是 RLHF / DPO 最常见的数据基元：同一个 prompt，对比一个更好答案和一个更差答案。

## 复习题

1. 为什么 post-training 对产品模型几乎是必须的？
2. 为什么 rich SFT data 可能反而教会 hallucination？
3. 为什么 pairwise preference 比 full demonstration 更便宜？
