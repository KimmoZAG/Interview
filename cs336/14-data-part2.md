# 14｜数据工程（二）：过滤、去重与分布匹配

原始来源：<https://tuananhbui89.github.io/blog/2025/cs336-lec14/>

## 这讲的核心结论

- 数据过滤的本质是：从大集合 $R$ 中选出更接近目标集合 $T$ 的子集 $T'$。
- 快速过滤常用 n-gram / FastText / 轻量分类器；高质量过滤往往要 teacher-student 蒸馏。
- 去重是 web-scale 数据工程的必修课，exact dedup 和 near dedup 都要做。

## 代表图

![lec14](https://tuananhbui89.github.io/assets/img/cs336-2025/frames/lec14/01-08-41-1400.webp)

## 中文解读

### 1. 过滤的统一视角

无论你做语言识别、质量过滤、数学数据挑选，流程都很像：

1. 建模或打分；
2. 得到每篇文档分数；
3. threshold 或采样保留。

### 2. 为什么 n-gram 仍然有用

因为它很快。  
在十亿级网页上，先用便宜模型过滤掉明显垃圾，往往比一开始就上大模型更现实。

### 3. 为什么 near dedup 很难

精确重复可以直接 hash；  
但轻微改写、模板页、重复 license 文本、伪原创内容，都需要近似相似度方法，如 MinHash / LSH / embedding ANN。

## 代码拆解：Bloom Filter 直觉

```python
class TinyBloom:
    def __init__(self, m=1024):
        self.bits = [0] * m
        self.m = m

    def add(self, x):
        self.bits[hash(x) % self.m] = 1

    def contains(self, x):
        return self.bits[hash(x) % self.m] == 1
```

真实 Bloom Filter 会用多个 hash，但直觉很简单：  
**用极省空间的位图做“可能见过”的快速判定。**

## 代码拆解：MinHash / LSH 的一句话理解

```python
# 文档 -> shingles -> sketch -> buckets -> verify
```

意思是：

- 先把文档切成 token shingles；
- 用 sketch 压缩；
- 用 LSH 把相似文档送进同一个桶；
- 最后再做更精确验证。

这能把原本 $O(N^2)$ 的相似度比对问题压到可用范围。

## 复习题

1. 质量过滤和语言识别在流程上有什么共性？
2. 为什么 exact dedup 不足以处理 web 数据？
3. MinHash + LSH 为什么适合大规模 near dedup？
