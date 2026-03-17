# AI Infra 学习笔记（重构中）

目标：围绕 **推理优化方向**，用“工程师视角 + 面试视角”把 AI Infra 的核心知识串成一条清晰主线。

## 新组织方式（主线）

本目录已开始从旧的 `operators / models / inference / appendix` 四类结构，重构为更适合学习、复习与面试的五大板块：

1. **算子优化**：单个算子 / kernel / 子图为什么快、为什么慢、如何定位
2. **推理引擎**：runtime / compiler / serving / KV cache / allocator / observability
3. **大模型架构**：推理优化工程师需要掌握的 Transformer / attention / sampling / MoE 等核心知识
4. **通信技术**：collectives、互联拓扑、并行策略、MoE 通信、多卡推理扩展性
5. **附录**：术语、参考、CS336 映射，以及对主线有帮助但不属于第一优先级的扩展内容

## 当前状态说明

- 新结构已经启用，后续内容维护统一在五大板块目录下进行
- 旧结构已完成迁移，不再作为主入口使用

## 建议阅读顺序

如果你的目标是 **推理优化 / LLM Infra / Serving / 性能工程**，推荐按下面顺序读：

1. `03-llm-architecture/`
2. `01-operator-optimization/`
3. `02-inference-engine/`
4. `04-communication/`
5. `05-appendix/`

## 写作约定

主线章节尽量统一成：

1. **一页结论 / 要点**
2. **通用知识**
3. **最小例子 / 工程例子**
4. **常见面试问题**
5. **易错点**
6. **排查 checklist**
7. **参考资料**

模板见：[`05-appendix/chapter-template.md`](05-appendix/chapter-template.md)

## 入口

- 总索引：[`00-index.md`](00-index.md)
- 重构路线图：[`restructure-roadmap.md`](restructure-roadmap.md)
- 算子优化：[`01-operator-optimization/00-index.md`](01-operator-optimization/00-index.md)
- 推理引擎：[`02-inference-engine/00-index.md`](02-inference-engine/00-index.md)
- 大模型架构：[`03-llm-architecture/00-index.md`](03-llm-architecture/00-index.md)
- 通信技术：[`04-communication/00-index.md`](04-communication/00-index.md)
- 附录：[`05-appendix/00-index.md`](05-appendix/00-index.md)
