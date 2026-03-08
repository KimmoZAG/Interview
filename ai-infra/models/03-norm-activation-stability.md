# 常见层：Norm/激活/残差 与数值稳定性

## 要点

- 推理时常见数值问题：低精度下的溢出/下溢、softmax 稳定性、norm 的归一化误差
- 性能上：norm/激活多为 memory-bound 的 pointwise 链，适合融合

## Norm

- LayerNorm / RMSNorm：推理常见
- 关注点：
  - reduction（求均值/方差）引入同步与带宽压力
  - 低精度累计：是否使用 FP32 accumulate

## 激活与门控

- GELU / SiLU / SwiGLU：注意实现差异与近似
- 门控结构通常引入额外的 pointwise 与 reshape

## 残差

- `y = x + f(x)`：典型可与后续 norm/激活融合

## 易错点

- 低精度下 softmax 或 norm 的数值不稳导致输出漂移
- 融合后改变了计算顺序（浮点非结合律）引入细微差异

## 排查 checklist

- [ ] 使用固定输入对比融合前后最大/均方误差
- [ ] 看异常值：是否出现 inf/nan（尤其低精度）
- [ ] norm 是否 FP32 accumulate？
