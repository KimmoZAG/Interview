# 张量、shape 与内存布局

## 要点

- 推理性能的很多问题，本质是 **shape（维度）+ layout（stride）+ dtype（精度）** 的组合问题
- 任何算子讨论都应先明确：输入/输出 shape、是否 contiguous、是否需要 transpose/reshape
- layout 变化通常意味着额外 copy 或低效访问（coalescing 变差）

## 先把“张量”说清楚

- shape：例如 `[B, S, H]`（batch、序列长度、hidden）
- dtype：FP32/FP16/BF16/INT8/INT4
- contiguous：是否连续；stride 是多少
- view vs copy：reshape/transpose 可能只是 view，也可能触发 copy（取决于框架与后续算子需求）

## 推理里常见的关键维度

- LLM：
  - Prefill：`B x S` 大、并行度高
  - Decode：`S≈1`（逐 token），更容易被 launch/同步/访存开销主导

## 工程落地：你要在日志里打印什么

- 输入/输出的 shape、dtype、是否 contiguous
- 关键中间张量（例如 Q/K/V、KV cache）的 shape
- 动态 shape 变化点（batching、padding、不同输入长度）

## 易错点

- “看似没有 copy”但因为后续算子需要 contiguous 导致隐式 copy
- transpose 后的 stride 导致访存不连续，吞吐骤降

## 排查 checklist

- [ ] 能否固定 shape 复现？
- [ ] 是否出现了隐式的 layout 转换或 dtype cast？
- [ ] 关键张量是否 contiguous？stride 是否合理？

## 参考

- cppreference/框架文档 + 你实际用的 profiler 输出（建议贴图或记录关键数值）
