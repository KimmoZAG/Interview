# 评测与基准：accuracy/latency/throughput

## 要点

- 评测需要同时覆盖：正确性（或质量）与性能（吞吐、延迟、尾延迟）
- 基准要可复现：固定硬件/软件版本、固定输入集、明确 warmup 与统计方法

## 指标

- 质量：任务集得分、人工评审、回归样例
- 性能：
  - throughput：tokens/s、req/s
  - latency：TTFT（首 token 时间）、TPOT（每 token 时间）、p50/p95/p99
  - 资源：显存占用、GPU 利用率、CPU 利用率

## 基准设计

- 固定输入长度分桶：短/中/长
- 区分 prefill 与 decode
- 并发水平：1、N（接近线上）

## 易错点

- 只做单条请求 benchmark，和线上差距巨大
- 忽略 warmup/缓存（CUDA graph、kernel cache、编译 cache）

## 排查 checklist

- [ ] benchmark 脚本是否固定了版本与参数？
- [ ] 是否按长度分桶报告 p95/p99？
- [ ] 是否记录了运行时的 GPU/CPU/显存曲线？
