# Tokenizer 与采样（top-k/top-p/temperature）

## 要点

- Tokenizer 影响：吞吐（预处理开销）、上下文长度（token 数）、以及最终输出质量
- 采样策略影响：生成质量与可复现性；线上排查需要记录随机种子与策略参数
- CS336 对 tokenizer 的强调，不只是 NLP 预处理，而是它直接决定后续的训练和推理成本

## Tokenizer

- 常见：BPE / SentencePiece 变体
- 工程关注点：
  - 批量 tokenize 的并行化
  - 输入长度限制与截断策略
  - token 数统计（影响 prefill 成本）

## 从 CS336 看 BPE 的最小工程流程

如果把 BPE 当成系统组件而不是算法题，它大致分 4 步：

1. 把原始文本转成字节序列
2. 做 pre-tokenization，把文本切成更稳定的片段
3. 统计相邻 byte/token pair 频次，迭代合并
4. 在新文本上按训练时得到的 merge 顺序重放

真正重要的工程点：

- 训练 tokenizer 和使用 tokenizer 必须共享完全一致的 merge 规则
- special tokens 必须在 encode/decode 两侧严格对齐
- 预分词规则会直接影响最终压缩率和词表利用率

## 为什么 compression ratio 很重要

CS336 Lecture 1 里很强调这一点：tokenizer 的好坏不只是语言学问题，更是成本问题。

- token 数更少：同样文本对应更短序列
- 序列更短：prefill 成本更低
- 对训练而言：同样 token budget 能看到的原始文本内容更多或更少

所以 tokenizer 评估至少要看两件事：

- 压缩率/序列长度
- 下游模型质量

## 采样

- temperature：调分布“尖锐/平坦”
- top-k：只在前 k 个 token 中采样
- top-p（nucleus）：累积概率到 p 的集合中采样

补一个足够用的生成直觉：

- temperature 更像全局调节分布平滑度
- top-k 更像硬截断
- top-p 更像按累计概率自适应截断

线上排查时，采样问题经常会伪装成“模型退化”，但其实只是参数配置变了。

## 可复现与排查

- 记录：采样参数、随机种子、是否启用 repetition penalty 等
- 注意：同样参数在不同实现/不同精度下也可能存在细微差异

## 你应该能回答的最小问题

- 为什么同一段文本换 tokenizer 后，性能会显著变化
- 为什么训练时看似细小的 tokenization 差异，会导致 scaling 和 benchmark 结果不可直接比较
- 为什么线上生成问题必须把采样参数和 tokenizer 版本一起记录

## 易错点

- 没有记录采样参数导致线上问题无法复现
- tokenization 的差异导致“同一段文本”token 数差很多，进而性能不同
- 把 tokenizer 当成静态资产，而不去量化它对成本和上下文利用率的影响

## CS336 对照

- 官方 lecture 对应：Lecture 1（overview, tokenization）
- 推荐官方入口：https://github.com/stanford-cs336/spring2025-lectures
- 推荐外部笔记：
  - https://rd.me/cs336/lec1
  - https://bearbearyu1223.github.io/posts/cs336-note-simple-bpe/
  - https://www.rajdeepmondal.com/blog/cs336-lecture-1
