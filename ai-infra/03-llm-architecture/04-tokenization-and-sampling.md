# Tokenizer 与采样

## 一句话先讲清

这一页回答的是：**输入是怎么被切成 token 的，输出又是怎么从概率分布里被“挑”出来的；而这两个看似边角的配置，为什么会直接影响成本、延迟、质量和可复现性。**

对于推理系统来说，Tokenizer 决定 prefill 有多贵，采样策略决定输出有多稳。它们不是“小配置”，而是系统行为的一部分。

## 为什么值得单独学

- 同一段文本，换个 tokenizer，token 数可能差很多，prefill 成本也会跟着变。
- 线上“风格漂移”或“难以复现”的生成问题，常常不是权重变了，而是 sampling 配置变了。
- benchmark 如果没把 tokenizer 版本和 sampling 参数固定住，结论经常会变得不太老实。

## 关联知识网络

- 模型侧基础：[`Transformer 推理所需的最小知识`](01-transformer-minimum.md)
- 成本来源：[`Attention、KV cache 与吞吐/延迟`](02-attention-kv-cache.md)
- 数值与稳定性：[`常见层：Norm/激活/残差 与数值稳定性`](03-norm-activation-stability.md)
- 评测口径：[`评测与基准：accuracy/latency/throughput`](05-evaluation-and-benchmarking.md)
- 系统落地：[`推理栈全景`](../02-inference-engine/01-inference-stack-overview.md)、[`LLM Serving`](../02-inference-engine/04-llm-serving.md)
- 长上下文影响：[`长上下文 Serving`](../02-inference-engine/08-long-context-serving.md)

## 它们分别解决什么问题

### Tokenizer

Tokenizer 负责把原始文本变成模型可处理的 token id 序列。

它决定了：

- 同一段文本会被切成多少个 token
- 上下文窗口会被消耗得多快
- prefill 阶段要处理多少序列长度

### Sampling

Sampling 负责把模型输出的概率分布转成最终生成 token。

它决定了：

- 输出是更保守还是更多样
- 结果是否容易复现
- 用户感觉到的“风格”“发散度”“稳定性”

## 为什么它们会影响系统成本

| 配置项 | 直接影响 | 典型工程后果 |
|---|---|---|
| Tokenizer | token 数、上下文利用率 | prefill 成本、KV cache 压力、长上下文预算 |
| Temperature | 分布尖锐程度 | 输出稳定性 vs 多样性 |
| Top-k | 固定候选集合截断 | 输出更可控，但可能偏保守 |
| Top-p | 累积概率自适应截断 | 输出更自然，但更依赖分布形态 |
| Seed | 随机性复现 | 线上问题是否能回放 |

一句工程直觉：**如果 tokenizer 让输入 token 数下降 20%，那你往往不是只省了 20% 预处理，而是连 prefill、attention、KV cache 压力都一起降。**

## Tokenizer：别把它当纯预处理

常见 tokenizer 方案包括 BPE、SentencePiece 及其变体。工程上更值得关注的是：

- 批量 tokenize 的并行化能力
- 输入长度限制与截断策略
- special tokens 与 prompt 模板的一致性
- 相同文本在不同 tokenizer 下的 token 长度差异

一个非常实用的判断标准是：**你到底是在比较模型，还是在比较 tokenizer 把文本切得多短。**

## Sampling：别把它当“随便调一下”

### 三个最常见参数

- `temperature`：整体调节分布是更尖锐还是更平坦
- `top-k`：只在概率最高的前 $k$ 个 token 中采样
- `top-p`：在累计概率达到 $p$ 的集合中采样

可以先记一个够用的直觉：

- `temperature` 更像全局平滑度旋钮
- `top-k` 更像固定候选集合裁剪
- `top-p` 更像按概率质量自适应截断

### 常见组合的行为倾向

| 组合 | 倾向 | 风险 |
|---|---|---|
| `temperature = 0` | 更稳定、更接近贪心 | 多样性低，容易模板化 |
| `temperature = 0.7 ~ 1.0` + `top-p` | 更自然、常见于对话生成 | 结果波动更大，复现更难 |
| 较小 `top-k` | 更可控、更保守 | 容易错过长尾但合理的候选 |
| 激进采样参数 | 输出更开放 | 幻觉、漂移、风格不稳更明显 |

## 最小例子

假设同一句输入：

- tokenizer A 切成 1200 个 token
- tokenizer B 切成 900 个 token

在同样模型和硬件下，B 往往更省 prefill 成本，因为 attention 和 KV cache 只面对更短序列。

再看采样：

- `temperature = 0` 更接近稳定、保守输出
- `temperature = 1.0 + top-p = 0.95` 通常更开放、多样，但更难复现

这就是为什么“同一个模型体验变了”，不能只盯着权重参数看。

## Troubleshooting：用户说“昨天正常，今天风格很飘”时怎么查

| 现象 | 第一怀疑点 | 如何验证 |
|---|---|---|
| 同题多次回答差异很大 | sampling 参数变了 | 对比 temperature / top-k / top-p / seed |
| 吞吐突然变好或变差 | tokenizer 或 prompt 模板改了 | 对比同样输入的 token 数变化 |
| 明明没换模型，体验却变了 | tokenizer 版本、special tokens 变化 | 固定 prompt 做回放对比 |
| 长上下文场景成本飙升 | token 压缩率不理想 | 统计输入 token 长度分布 |
| benchmark 无法复现线上输出 | seed / 采样配置未记录 | 查实验配置与线上请求日志 |

### 一个排障顺序

1. 先固定 tokenizer 版本、special tokens、prompt 模板。
2. 再固定 sampling 参数：temperature、top-k、top-p、seed、repetition penalty。
3. 对比同一段文本的 token 长度是否发生变化。
4. 最后再讨论是不是模型质量或权重本身出了问题。

很多生成问题其实不是模型坏了，而是**输入被切得不一样，或者输出被采样得更激进了**。

## 推理优化工程师视角

从推理优化工程师视角看，这一页最重要的不是 NLP 术语，而是两条账：

1. **成本账**：Tokenizer 决定序列有多长，从而决定 prefill、KV cache、长上下文压力有多大。
2. **复现账**：Sampling 决定线上输出能不能稳定回放，能不能做公平 benchmark。

因此一个成熟的系统通常会把下面这些内容一起版本化：

- tokenizer 版本
- special tokens
- prompt 模板
- temperature / top-k / top-p / seed

把这层配置管好，能省掉非常多“以为是模型坏了”的误判。

## 面试高频问法

### 初级

1. tokenizer 为什么会影响推理成本？
2. top-k 和 top-p 的区别是什么？

### 中级

1. 为什么同一段文本换 tokenizer 后，系统吞吐可能明显变化？
2. 为什么线上生成问题必须记录 sampling 参数？

### 高级

1. 如果 benchmark 没变，但线上体验变差，你如何判断是不是 sampling 配置导致？
2. 为什么 tokenizer 差异会让不同模型间的训练/推理成本比较不再直接可比？

## 易错点

- 没记录采样参数，导致线上问题无法复现
- 只看“文本长度”，不看真实 token 长度
- 把 tokenizer 当静态资产，不量化它对成本和上下文利用率的影响
- 把配置变化误判成模型质量变化

## 排查 checklist

- [ ] 当前 tokenizer 版本是否固定？special tokens 是否一致？
- [ ] 是否记录了 temperature、top-k、top-p、seed、repetition penalty？
- [ ] 相同文本在不同 tokenizer 下 token 长度差多少？
- [ ] prompt 模板是否也被一起固定？
- [ ] 当前问题更像模型质量问题，还是 sampling / 配置问题？

## 参考资料

- tokenizer 相关课程与文档
- SentencePiece / BPE 资料
- 生成采样策略相关资料
