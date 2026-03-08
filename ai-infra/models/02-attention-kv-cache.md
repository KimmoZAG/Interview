# Attention、KV cache 与吞吐/延迟

## 要点

- Attention 在 decode 阶段常见瓶颈：读取历史 KV（带宽）+ 小 shape 计算（launch/调度）
- KV cache 的组织方式（连续/分页/分块）会显著影响显存占用与吞吐

## Attention 的推理关注点

- Prefill：典型是“全量 attention”，计算更重
- Decode：单 token query 与历史 KV 做 attention，主要成本在读 KV 与 softmax/加权求和

## KV cache

- 存什么：每层的 K/V
- 为什么需要：自回归生成避免重复计算历史 token 的 K/V
- 关注项：
  - dtype（FP16/BF16/INT8 等）
  - layout（便于连续读取）
  - 分配策略（避免频繁扩容/拷贝）

## 吞吐/延迟的常见矛盾

- 吞吐：更依赖 batching（合并请求）
- 尾延迟：更敏感于排队、同步点、显存抖动、cache miss

## 易错点

- 只看平均延迟不看 p95/p99
- 不同长度请求混在一起导致 padding 浪费或调度不稳定

## 排查 checklist

- [ ] 统计 prefill 与 decode 的时间占比
- [ ] KV cache 的显存占用随并发增长曲线
- [ ] 是否存在频繁的 KV reallocation 或 copy
