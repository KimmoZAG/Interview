# 术语表

- FLOPs：浮点运算次数（粗略衡量算力需求）
- Roofline：以算力峰值与带宽峰值作为上界的性能模型
- Arithmetic Intensity：算术强度（FLOPs/Byte），用于判断算力/带宽瓶颈
- Layout：张量内存布局（例如 NCHW/NHWC；或连续/stride）
- Fusion：算子融合（减少访存/launch 开销）
- KV cache：自回归生成时缓存的 Key/Value
- Prefill / Decode：LLM 推理两阶段（上下文填充 / 逐 token 解码）
- Batching：把多个请求合并执行以提升吞吐
