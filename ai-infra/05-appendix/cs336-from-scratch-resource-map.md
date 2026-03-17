# CS336：Language Modeling from Scratch 资源导读

> 说明：这篇不是课程翻译，而是 **如何把 CS336 当成这套 AI Infra 笔记的外部增强材料来使用**。

## 这份导读解决什么问题

如果你直接刷 CS336，常见痛点是：

- 课程内容很系统，但和“推理优化工程师复习主线”之间距离有点远
- 很多 lecture 很重要，却不知道该映射到仓库里的哪条主线去消化

这份导读的目标就是：

- 帮你先找入口
- 帮你知道每类资料该补哪块短板
- 帮你避免“看了很多课程，落不到当前笔记结构里”

## 官方入口

- 课程主页：https://cs336.stanford.edu/
- 2025 存档：https://cs336.stanford.edu/spring2025/
- 讲义仓库：https://github.com/stanford-cs336/spring2025-lectures
- YouTube 播放列表：https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

## 推荐使用方式

### 方式一：先读仓库主线，再回看课程

适合：

- 已经在准备面试
- 想优先建立“推理优化工程师”视角

建议顺序：

1. `../03-llm-architecture/`
2. `../01-operator-optimization/`
3. `../02-inference-engine/`
4. `../04-communication/`
5. 再用 CS336 补训练与理论背景

### 方式二：跟着课程看，再用仓库做工程化重写

适合：

- 想把课程内容转成更贴近系统/面试的理解

做法是：

- 每看完一讲，回到仓库找对应主线文章
- 优先补“这个课程点在工程里意味着什么、常见瓶颈是什么、面试会怎么问”

## 课程材料如何搭配这套笔记

| 你想补什么 | 优先看哪里 |
|---|---|
| Transformer / attention / tokenization 基础 | `../03-llm-architecture/` |
| kernel、roofline、fusion、FlashAttention | `../01-operator-optimization/` |
| runtime、编译器、serving、KV cache | `../02-inference-engine/` |
| 多卡通信、collective、拓扑 | `../04-communication/` |
| scaling、数据、对齐、奖励设计 | `../05-appendix/` |

## 什么时候最值得回看 CS336

- 当你主线概念已经通了，但觉得训练背景还不牢
- 当你能回答“是什么”，但还不够会回答“为什么这样设计”
- 当你想把面试回答从背定义，提升到带一点课程深度和推导味道

## 建议搭配阅读

- [cs336-lecture-to-ai-infra-map.md](cs336-lecture-to-ai-infra-map.md)
- [../03-llm-architecture/01-transformer-minimum.md](../03-llm-architecture/01-transformer-minimum.md)
- [../02-inference-engine/05-optimization-playbook.md](../02-inference-engine/05-optimization-playbook.md)

## 一个实用提醒

CS336 很适合补“从头理解大模型系统”，但这套仓库更偏“推理优化工程师如何组织知识”。

最好的使用方式通常不是二选一，而是：

- 用仓库建立主线
- 用课程补推导、背景和更完整的训练语境
