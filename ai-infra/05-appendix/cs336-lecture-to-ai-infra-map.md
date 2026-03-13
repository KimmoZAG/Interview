# CS336 逐讲映射：从课程到 AI Infra 笔记

> 说明：这篇不是逐字复述 lecture，而是把课程主题翻译成这套仓库里的学习路径。

## 最小映射

- Lecture 1-3 → `03-llm-architecture/`
- Lecture 5-6 → `01-operator-optimization/`
- Lecture 7-8 → `04-communication/`
- Lecture 9-11 → `02-inference-engine/` + `05-appendix/`（scaling 拓展）
- Lecture 12-17 → `03-llm-architecture/` + `05-appendix/`

## 扩展映射（按主题理解）

### 模型基础与训练直觉

- Lecture 1-3
- 对应仓库：
	- `../03-llm-architecture/01-transformer-minimum.md`
	- `../03-llm-architecture/02-attention-kv-cache.md`
	- `../03-llm-architecture/04-tokenization-and-sampling.md`

适合补的问题：

- Transformer 最小工作原理是什么？
- 为什么 attention 会牵出 KV cache 和长上下文问题？

### 算子与性能模型

- Lecture 5-6
- 对应仓库：
	- `../01-operator-optimization/01-tensors-shapes-layout.md`
	- `../01-operator-optimization/03-memory-hierarchy-and-roofline.md`
	- `../01-operator-optimization/06-flashattention-io-aware.md`

适合补的问题：

- 为什么某些优化看起来像“数学改写”，本质却是 IO 优化？
- 怎么从 roofline 角度判断一个 kernel 值不值得继续抠？

### 通信与并行

- Lecture 7-8
- 对应仓库：
	- `../04-communication/01-training-parallelism.md`
	- `../04-communication/02-communication-foundations.md`
	- `../04-communication/05-parallelism-to-communication.md`

适合补的问题：

- 为什么并行策略其实是在选择通信形态？
- 为什么总通信量相同，扩展效率却可以差很多？

### 编译、runtime 与 serving

- Lecture 9-11
- 对应仓库：
	- `../02-inference-engine/02-runtime-onnxruntime-tensorrt.md`
	- `../02-inference-engine/03-graph-compiler-tvm-mlir-xla.md`
	- `../02-inference-engine/04-llm-serving.md`

适合补的问题：

- 为什么离线 benchmark 的收益到线上未必成立？
- runtime / compiler / serving 分别在解决哪一层问题？

### scaling、数据与后训练

- Lecture 12-17
- 对应仓库：
	- `scaling-laws-and-budgeting.md`
	- `pretraining-data-engineering.md`
	- `data-mixing-and-curriculum.md`
	- `post-training-and-alignment.md`
	- `reward-and-verifier-design.md`

适合补的问题：

- 为什么 loss 变好不一定代表产品更好？
- 后训练到底在优化什么，reward 和 verifier 分别扮演什么角色？

## 建议顺序

1. `03-llm-architecture/`
2. `01-operator-optimization/`
3. `02-inference-engine/`
4. `04-communication/`

## 一种适合面试复习的用法

如果你正在准备面试，可以把这篇当成“课程内容 → 面试主线”的跳板：

- 先用 lecture 建立背景
- 再回到对应仓库文章，把回答重写成“概念 + 例子 + 工程判断 + 常见坑”

这样能避免回答里只有课程名词，没有系统味道。
