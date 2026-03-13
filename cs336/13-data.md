# 13｜数据工程（一）：数据来源、过滤与版权

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec13/>

## 这讲的核心结论

- 对 LLM 来说，数据不是配角，而是**行为上限的第一决定因素**。
- 训练通常分成 pre-training、mid-training、post-training 三阶段，数据质量逐级提升、规模逐级缩小。
- 数据工程不是纯理论问题，而是强工程、强法律、强启发式的问题。

## 代表图

![lec13](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec13/00-15-05-1400.webp)

## 中文解读

### 1. 为什么说数据决定模型行为

模型会什么、不会什么、偏向什么、会不会说代码、会不会说多语言，很大程度上取决于训练集中出现了什么。

### 2. 三阶段数据的目标不同

- **Pre-training**：追求覆盖与规模；
- **Mid-training**：追求能力定向，比如数学、代码、长上下文；
- **Post-training**：追求交互格式、安全与人类偏好。

### 3. Common Crawl 为什么又香又难吃

它量很大，但原始网页里充满：

- boilerplate
- spam
- near-duplicate
- 有毒内容
- 版权风险

所以真正的数据价值，常常来自后处理而不是原始 crawl 本身。

## 代码拆解：数据 pipeline 最小骨架

```python
def data_pipeline(raw_docs):
    docs = extract_text(raw_docs)
    docs = language_filter(docs)
    docs = deduplicate(docs)
    docs = quality_filter(docs)
    docs = license_filter(docs)
    return docs
```

这条流水线说明：一个好数据集不是“下载下来”，而是“加工出来”。

## 版权与工程现实

- 是否可爬，不只看能不能访问，还要看 ToS；
- 是否可训练，不只看技术上能不能用，还要看 license 与法务风险；
- 是否可发布，又是另一层问题。

## 复习题

1. 为什么 mid-training 在现代模型里越来越重要？
2. Common Crawl 的主要问题有哪些？
3. 为什么 license filtering 是数据工程不可省的一步？
