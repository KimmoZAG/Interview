# Tokenizer 与采样（top-k/top-p/temperature）

## 要点

- Tokenizer 影响：吞吐（预处理开销）、上下文长度（token 数）、以及最终输出质量
- 采样策略影响：生成质量与可复现性；线上排查需要记录随机种子与策略参数
- CS336 对 tokenizer 的强调，不只是 NLP 预处理，而是它直接决定后续的训练和推理成本

## 通用知识

### 它是什么

Tokenizer 负责把原始文本映射成 token id 序列；采样策略负责把模型输出的概率分布转成最终生成 token。

一个在输入端，一个在输出端，但两者都会直接影响：

- 成本
- 延迟
- 质量
- 可复现性

### 它解决什么问题

Tokenizer 解决的是：

- 文本如何变成模型可处理的离散单元

采样策略解决的是：

- 模型在多个候选 token 之间如何选择

### 为什么在 AI 系统里重要

因为：

- tokenization 会直接改变序列长度，进而改变 prefill 成本
- sampling 会直接影响输出稳定性、随机性和线上问题复现难度

### 它的收益与代价

Tokenizer 更高压缩率的收益：

- token 更少
- context 利用率更高
- prefill 成本更低

代价可能是：

- 词表更复杂
- 某些分词边界不符合目标任务

采样更激进的收益：

- 输出更多样

代价是：

- 可复现性更差
- 质量和稳定性更容易漂移

## Tokenizer

- 常见：BPE / SentencePiece 变体
- 工程关注点：
  - 批量 tokenize 的并行化
  - 输入长度限制与截断策略
  - token 数统计（影响 prefill 成本）

补一个工程直觉：

- 同一段文本，如果 tokenizer 让 token 数下降 20%，那 prefill 成本通常也会随之明显下降
- 所以 tokenizer 不是“预处理小问题”，而是系统成本问题

## 采样

- temperature：调分布“尖锐/平坦”
- top-k：只在前 k 个 token 中采样
- top-p（nucleus）：累积概率到 p 的集合中采样

一个够用的生成直觉：

- temperature 更像全局平滑度调节
- top-k 更像固定候选集合截断
- top-p 更像按累计概率自适应截断

## 最小例子

假设同一句输入：

- tokenizer A 切成 1200 个 token
- tokenizer B 切成 900 个 token

在同样模型和硬件下，B 往往更省 prefill 成本，因为 attention 和 cache 都只面对更短序列。

再看采样：

- `temperature=0` 更接近稳定、保守输出
- `temperature=1.0 + top-p=0.95` 通常更开放、多样，但也更难复现

## 工程例子

一个典型线上问题是：

- 用户反馈“同一个问题昨天答得正常，今天风格很飘”

排查时经常不是模型权重变了，而是：

- sampling 参数改了
- tokenizer 版本变了
- special token 或 prompt 模板配置变了

所以生成问题排查时，tokenizer 版本和 sampling 配置必须跟模型版本一起记录。

## 推理优化工程师视角

推理优化工程师读这一章，不应该只把它当成“模型输入输出的边角料”。更准确地说：

- tokenizer 决定了序列有多长，从而决定 prefill 成本和上下文利用率
- sampling 决定了生成行为是否稳定、问题是否可复现
- 这两者都会直接影响线上体验，却很容易被误归因成“模型质量变化”

因此在系统排查里，常见的好习惯是：

1. 把 tokenizer 版本、special tokens、prompt 模板一起固化
2. 把 temperature / top-k / top-p / seed 视作 benchmark 配置的一部分
3. 对比“token 数变化”和“模型速度变化”，避免把输入变短误当成模型更快

很多线上生成问题其实不是模型坏了，而是输入被切得不一样、输出被采样得更激进了。把这层配置管理好，能省掉大量无效排障。

## 常见面试问题

### 初级

1. tokenizer 为什么会影响推理成本？
2. top-k 和 top-p 的区别是什么？

### 中级

1. 为什么同一段文本换 tokenizer 后，系统吞吐可能明显变化？
2. 为什么线上生成问题必须记录 sampling 参数？

### 高级

1. 如果某模型 benchmark 没变，但线上体验变差，你如何判断是否是 sampling 配置导致？
2. 为什么 tokenizer 差异会让不同模型之间的训练/推理成本比较变得不直接可比？

## 可复现与排查

- 记录：采样参数、随机种子、是否启用 repetition penalty 等
- 注意：同样参数在不同实现/不同精度下也可能存在细微差异

## 易错点

- 没有记录采样参数导致线上问题无法复现
- tokenization 的差异导致“同一段文本”token 数差很多，进而性能不同
- 把 tokenizer 当成静态资产，而不去量化它对成本和上下文利用率的影响

## 排查 checklist

- [ ] 当前 tokenizer 版本是否固定？special tokens 是否一致？
- [ ] 是否记录了 temperature、top-k、top-p、seed、repetition penalty？
- [ ] 相同文本在不同 tokenizer 下 token 长度差多少？
- [ ] 当前问题更像模型质量问题，还是 sampling/配置问题？

## 参考资料

- tokenizer 相关课程/文档
- SentencePiece / BPE 资料
- 生成采样策略相关资料
