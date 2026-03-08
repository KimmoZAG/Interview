# Tokenizer 与采样（top-k/top-p/temperature）

## 要点

- Tokenizer 影响：吞吐（预处理开销）、上下文长度（token 数）、以及最终输出质量
- 采样策略影响：生成质量与可复现性；线上排查需要记录随机种子与策略参数

## Tokenizer

- 常见：BPE / SentencePiece 变体
- 工程关注点：
  - 批量 tokenize 的并行化
  - 输入长度限制与截断策略
  - token 数统计（影响 prefill 成本）

## 采样

- temperature：调分布“尖锐/平坦”
- top-k：只在前 k 个 token 中采样
- top-p（nucleus）：累积概率到 p 的集合中采样

## 可复现与排查

- 记录：采样参数、随机种子、是否启用 repetition penalty 等
- 注意：同样参数在不同实现/不同精度下也可能存在细微差异

## 易错点

- 没有记录采样参数导致线上问题无法复现
- tokenization 的差异导致“同一段文本”token 数差很多，进而性能不同
