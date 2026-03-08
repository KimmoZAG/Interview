# Runtime：ONNX Runtime / TensorRT（要点清单）

## 要点

- ONNX Runtime（ORT）：偏通用 runtime，生态好、易集成
- TensorRT（TRT）：偏 NVIDIA GPU 极致优化，常见于静态/半静态图与 FP16/INT8 场景

## 选型维度（工程视角）

- 支持的算子覆盖与 fallback 路径
- 动态 shape 支持与性能波动
- 量化与校准工具链
- 部署形态：C++/Python、跨平台要求、版本锁定成本

## 常见性能手段

- 图级：常量折叠、融合、消除冗余 reshape/transpose
- runtime：内存池、I/O binding、异步拷贝与 stream
- 编译缓存：engine/cache 的构建与复用

## 易错点

- 动态 shape 导致生成多个 engine/cache，显存与构建时间暴涨
- fallback 到非预期 provider（CPU/GPU 混跑）

## 排查 checklist

- [ ] 实际跑的是哪个 execution provider / engine？
- [ ] engine/cache 命中率是多少？
- [ ] kernel 数量是否异常偏多（融合没生效）？
