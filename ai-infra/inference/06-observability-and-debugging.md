# 可观测性与调试（profiling、tracing、metrics）

## 要点

- 没有可观测性就很难做性能/稳定性闭环
- 线上最有用的不是“平均值”，而是分布与分桶（长度、并发、模型版本）

## Metrics（建议你后续按项目补）

- 请求：QPS、排队时间、超时率、错误码
- 延迟：TTFT、TPOT、p50/p95/p99
- 资源：GPU utilization、SM occupancy（如果可得）、显存占用、CPU 利用率
- 质量：回归样例通过率、关键业务指标

## Tracing

- 关键 span：tokenize、prefill、decode 循环、postprocess
- 关联：请求 ID、模型版本、采样参数、输入长度

## Profiling

- CPU：热点函数、线程争用
- GPU：kernel 时间分布、kernel 次数、带宽/占用

## 易错点

- 指标没有按输入长度/输出长度分桶，导致看不出问题
- 只看 GPU 利用率，不看 kernel 数/带宽/同步点

## 排查 checklist

- [ ] 能否用一次 trace 把单请求的全链路还原？
- [ ] 是否能定位到具体 kernel/算子热点？
- [ ] 是否能把性能回归关联到一次版本变更？
