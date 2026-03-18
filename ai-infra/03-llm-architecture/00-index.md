# 大模型架构索引

定位：回答 **推理优化工程师需要懂到什么程度的模型知识，才能做出正确的系统判断**。

## 建议学习顺序

1. Transformer 最小知识（推理视角）
2. Attention 与 KV Cache（模型语义层）
3. Norm / 激活 / 残差 与数值稳定性
4. Tokenizer 与 Sampling
5. 评测与 Benchmark：accuracy / latency / throughput
6. MoE：路由、容量、系统代价
7. 长上下文的模型侧问题
8. 推理工程师需要懂的训练资源账（选修）

## 存量内容映射

- [Transformer 推理所需的最小知识](01-transformer-minimum.md)
- [Attention、KV cache 与吞吐/延迟](02-attention-kv-cache.md)
- [常见层：Norm/激活/残差 与数值稳定性](03-norm-activation-stability.md)
- [Tokenizer 与采样](04-tokenization-and-sampling.md)
- [评测与基准：accuracy/latency/throughput](05-evaluation-and-benchmarking.md)
- [MoE 最小导读](06-moe-minimum.md)
- [训练资源核算（选修）](07-training-resource-accounting.md)

## 已完成“实战型重构”的核心页

- [Transformer 推理所需的最小知识](01-transformer-minimum.md)
- [Attention、KV cache 与吞吐/延迟](02-attention-kv-cache.md)
- [常见层：Norm/激活/残差 与数值稳定性](03-norm-activation-stability.md)
- [Tokenizer 与采样](04-tokenization-and-sampling.md)
- [评测与基准：accuracy/latency/throughput](05-evaluation-and-benchmarking.md)
- [MoE 最小导读](06-moe-minimum.md)
- [训练资源核算（选修）](07-training-resource-accounting.md)

## 后续补强建议

- 重写 `Attention 与 KV Cache` 的边界：只讲模型语义和成本，不展开 allocator 细节
- 为 `MoE` 单独补“dense vs sparse 的面试问法”
- 给本板块每篇都增加一个“推理优化工程师视角”小节
