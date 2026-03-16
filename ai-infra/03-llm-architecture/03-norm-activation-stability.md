# 常见层：Norm/激活/残差 与数值稳定性

## 要点

- 推理时常见数值问题：低精度下的溢出/下溢、softmax 稳定性、norm 的归一化误差
- 性能上：norm/激活多为 memory-bound 的 pointwise 链，适合融合

## 通用知识

### 它是什么

这一类层包括：

- Norm：LayerNorm、RMSNorm
- 激活：GELU、SiLU、SwiGLU
- 残差：`x + f(x)` 形式的主干连接

它们单个看起来都不如 attention 或 GEMM 显眼，但在模型稳定性和执行效率上非常关键。

### 它解决什么问题

- Norm：控制数值尺度，提高训练和推理稳定性
- 激活：提升模型非线性表达能力
- 残差：让深层网络更容易优化、更稳定传递信息

### 为什么在 AI 系统里重要

因为它们往往：

- 数值上很敏感，尤其在低精度下
- 性能上偏 memory-bound，容易成为 pointwise 碎片热点
- 经常和融合、kernel 顺序、accumulate 精度一起出问题

### 它的收益与代价

- 更稳的 norm 和激活设计能让深层模型更可靠
- 但在推理实现里，pointwise 链条一长，就容易拖慢执行或引入细微数值差异

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

## 最小例子

一个常见场景：

- 权重和激活都用低精度
- 但 norm 的 reduction 若也用低精度累计

就更容易出现：

- 均值/方差估计不稳
- 输出漂移
- 极端情况下出现 inf / nan

所以很多实现会在关键 reduction 上保留 FP32 accumulate。

## 工程例子

一个典型问题是：

- 融合了 `residual + norm + activation` 后性能提升了
- 但某些回归样例输出发生轻微漂移

常见原因包括：

- 浮点计算顺序改变
- reduction 精度变化
- 激活近似实现不同

这就是为什么 pointwise 融合要同时做性能验证和误差验证。

## 推理优化工程师视角

从推理优化角度看，这一章的核心价值是提醒你：不是所有热点都长得像大 GEMM。

- norm / activation / residual 往往是 memory-bound
- 单个 kernel 看起来不大，但串起来会带来大量 launch 和访存开销
- 一旦做融合，就要同时承担数值漂移验证成本

所以在真实工程里，这类层通常意味着两条线要一起抓：

1. 性能线：减少 pointwise 碎片、减少读写往返、提升融合度
2. 正确性线：控制误差、检查 accumulate 精度、固定回归样例

如果一个优化只带来一点吞吐提升，却让误差边界不清楚、回归成本暴涨，那它未必是好优化。对推理系统来说，稳定可验证，往往比“理论上更快一点”更重要。

## 常见面试问题

### 初级

1. LayerNorm 和 RMSNorm 在直觉上有什么差别？
2. 为什么激活和 norm 往往更像 memory-bound？

### 中级

1. 为什么低精度下 norm 常需要更高精度 accumulate？
2. 为什么 pointwise 融合虽然单次收益不大，却常常值得做？

### 高级

1. 如果融合后性能更好但输出漂移，你会先排查哪些数值因素？
2. 为什么某些大模型更偏爱 RMSNorm 这类设计？

## 易错点

- 低精度下 softmax 或 norm 的数值不稳导致输出漂移
- 融合后改变了计算顺序（浮点非结合律）引入细微差异
- 只关注大 GEMM，不关注 pointwise 链对尾延迟和 kernel 次数的影响

## 排查 checklist

- [ ] 使用固定输入对比融合前后最大/均方误差
- [ ] 看异常值：是否出现 inf/nan（尤其低精度）
- [ ] norm 是否 FP32 accumulate？

## 参考资料

- LayerNorm / RMSNorm 相关资料
- 激活函数与数值稳定性实践资料
- kernel 融合与低精度实现相关文档
