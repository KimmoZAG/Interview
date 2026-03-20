---
tags:
  - AI Infra
  - LMCache
  - KV Cache
  - LLM Serving
description: 基于 LMCache 源码的面试导向学习笔记，聚焦 KV Cache 复用、分层卸载、P2P 共享与 Disaggregated Prefill。
---

# LMCache 学习笔记

这套笔记只做一件事：把 **LMCache 为什么存在、核心链路怎么跑、源码里关键抽象如何配合、面试时该怎么讲到位** 讲清楚。

阅读目标不是“知道 LMCache 支持哪些 feature”，而是回答下面这些更硬的问题：

1. **为什么 KV Cache 会从模型内部细节，变成独立的系统层？**
2. **LMCache 如何把 GPU 上的 KV，变成可复用、可搬运、可跨实例共享的对象？**
3. **分层存储、异步加载、P2P 查找、PD 分离式预填充，这几条链路各自解决什么瓶颈？**
4. **如果你来设计类似系统，哪些地方最容易在并发、IO、显存、序列化、调度上踩坑？**

## 建议阅读方式

如果你的目标是 **AI Infra / 推理优化 / Serving / 系统架构面试**，推荐顺序如下：

1. 先看 [00-index.md](00-index.md)，建立整套系统地图。
2. 面试前冲刺时，先过一遍 [interview-review-outline.md](interview-review-outline.md)。
3. 再按章节顺序阅读，每章都先抓住“痛点 -> 机制 -> 代码 -> 面试问法”。
4. 最后回看每章最后的“面试可能问到的问题”，训练自己的系统表达。

## 本系列的固定写法

每一章都严格按同一套结构展开：

1. **技术背景**：没有 LMCache 时，Serving 框架到底卡在哪里。
2. **技术核心（结合代码）**：关键类、关键函数、关键数据流，配少量伪代码和核心结构解释。
3. **面试可能问到的问题**：只问有区分度的底层问题，并给出满分回答思路。

## 当前内容

当前已经提供完整内容：

1. 主线 8 章正文。
2. 附录 A：Native Fast Path 与 C++/CUDA 扩展。
3. 附录 B：Operator 与生产级多节点部署。
4. [interview-review-outline.md](interview-review-outline.md) 面试速记版总复习提纲。