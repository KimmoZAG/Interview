# CS336：Language Modeling from Scratch 资源导读

## 要点

- CS336 的核心不是“会调一个大模型 API”，而是从 **tokenizer、Transformer、训练、并行、数据、对齐** 一路下钻到可运行实现。
- 这门课非常偏工程，适合把“模型知识”和“系统知识”一起打通；如果只站在高层框架视角，很容易卡在显存、通信、kernel 和数据处理这些底层问题上。
- 官方资源最值得优先跟：**课程主页、可执行讲义仓库、五次作业仓库、公开视频**。
- 社区资源的价值主要体现在两点：**补齐官方省略的踩坑细节**，以及展示“别人是如何把 baseline 做成现代 LLM 实现”的。
- 如果你的目标是建立底层直觉，推荐顺序是：**官方讲义/作业说明 → 自己实现最小版本 → 再看社区代码和博客对照**。
- 如果你要把这门课真正吸收到当前知识库里，建议配合阅读：[cs336-lecture-to-ai-infra-map.md](cs336-lecture-to-ai-infra-map.md)。它把 17 讲逐讲映射到了现有目录和待补空白。

## 课程定位

官方对这门课的定义非常明确：它不是一门只讲论文综述的 NLP 课，而是一门“像操作系统课那样，从零把语言模型做出来”的课程。课程主线覆盖：

1. tokenization 与基础资源核算
2. Transformer 架构与训练超参数
3. GPU、kernel、Triton 与并行
4. scaling laws、推理、评测
5. 数据清洗/去重
6. SFT、RLHF/RL、对齐

从官方 prerequisites 也能看出门槛确实不低：

- Python 与软件工程能力要非常熟
- 需要深度学习基础，以及一定的 GPU/系统优化直觉
- 需要线性代数、概率统计、机器学习基础

结论：这门课更像“**大模型底层系统课**”，而不是“模型应用课”。

## 已映射到主干笔记

为了避免 CS336 只停留在“附录资源汇总”，当前已经明确落到主干章节的笔记包括：

### Models

- [../models/01-transformer-minimum.md](../models/01-transformer-minimum.md)
- [../models/02-attention-kv-cache.md](../models/02-attention-kv-cache.md)
- [../models/04-tokenization-and-sampling.md](../models/04-tokenization-and-sampling.md)
- [../models/05-evaluation-and-benchmarking.md](../models/05-evaluation-and-benchmarking.md)
- [../models/06-training-resource-accounting.md](../models/06-training-resource-accounting.md)
- [../models/07-post-training-and-alignment.md](../models/07-post-training-and-alignment.md)
- [../models/08-moe-minimum.md](../models/08-moe-minimum.md)
- [../models/09-reward-and-verifier-design.md](../models/09-reward-and-verifier-design.md)

### Operators

- [../operators/02-kernels-and-parallelism.md](../operators/02-kernels-and-parallelism.md)
- [../operators/03-memory-hierarchy-and-roofline.md](../operators/03-memory-hierarchy-and-roofline.md)
- [../operators/04-graph-fusion-scheduling.md](../operators/04-graph-fusion-scheduling.md)
- [../operators/06-training-parallelism.md](../operators/06-training-parallelism.md)
- [../operators/07-flashattention-io-aware.md](../operators/07-flashattention-io-aware.md)

### Inference

- [../inference/01-inference-stack-overview.md](../inference/01-inference-stack-overview.md)
- [../inference/05-optimization-playbook.md](../inference/05-optimization-playbook.md)
- [../inference/07-scaling-laws-and-budgeting.md](../inference/07-scaling-laws-and-budgeting.md)
- [../inference/08-pretraining-data-engineering.md](../inference/08-pretraining-data-engineering.md)
- [../inference/09-training-metrics-vs-product-metrics.md](../inference/09-training-metrics-vs-product-metrics.md)
- [../inference/10-data-mixing-and-curriculum.md](../inference/10-data-mixing-and-curriculum.md)
- [../inference/11-paged-kv-and-allocator.md](../inference/11-paged-kv-and-allocator.md)
- [../inference/12-long-context-training-and-serving.md](../inference/12-long-context-training-and-serving.md)

这些笔记现在都已经补入了：

- 对应的 CS336 lecture 主线
- 更贴近课程语境的工程结论
- 官方 lecture 与外部高质量笔记的延伸阅读入口

到这一步，CS336 的主干主题已经基本都能在知识库里找到对应落点。后续如果继续扩展，更适合做“实验模板、对照表、案例图谱”这类深化材料，而不是继续补基础条目。

## 官方资源

### 1. 课程主页

- 主页：https://cs336.stanford.edu/
- 2025 存档：https://cs336.stanford.edu/spring2025/

建议用途：先用主页确认课程范围、先修要求、作业结构和时间线，不要一上来就埋头看社区二手总结。

### 2. 公开视频

- YouTube 播放列表：https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

建议用途：把 lecture 当成“体系化主线”，博客和开源仓库只做补充。部分地区或未登录状态下页面可能跳转登录，但播放列表本身是官方公开入口。

### 3. 可执行讲义仓库

- 2025 讲义：https://github.com/stanford-cs336/spring2025-lectures

这个仓库最有辨识度的点，不是“有讲义”，而是：

- 讲义以 `lecture_*.py` 形式存在，可执行
- 支持生成 trace 文件
- 仓库内带有 `trace-viewer` 前端，可本地可视化运行结果
- 还包含远程执行/Slurm 相关脚本
- 部分内容直接暴露了 CUDA / Triton / 系统实验素材

这说明官方教学思路不是把代码当作附录，而是把“**可运行观测**”作为讲义的一部分。

### 4. 官方作业仓库

- GitHub 组织页：https://github.com/stanford-cs336
- Assignment 1 Basics：https://github.com/stanford-cs336/assignment1-basics
- Assignment 2 Systems：https://github.com/stanford-cs336/assignment2-systems
- Assignment 3 Scaling：https://github.com/stanford-cs336/assignment3-scaling
- Assignment 4 Data：https://github.com/stanford-cs336/assignment4-data
- Assignment 5 Alignment：https://github.com/stanford-cs336/assignment5-alignment

建议把这五个作业理解为一条连续能力链，而不是五个散点题目：

| 作业 | 核心目标 | 你真正练到的能力 |
|---|---|---|
| A1 Basics | tokenizer、Transformer、loss、AdamW、训练循环 | 从零搭建最小 LM |
| A2 Systems | profiling、FlashAttention2 Triton、DDP、state sharding | 系统性能与分布式训练 |
| A3 Scaling | scaling laws、预算推断、超参数迁移 | 用小实验预测大模型 |
| A4 Data | Common Crawl 处理、过滤、去重 | 预训练数据工程 |
| A5 Alignment | SFT、reasoning RL、DPO/GRPO 等 | 后训练与对齐 |

## 社区开源实现：看什么最值

社区仓库很多，但并不是都值得细看。优先挑这三类：

1. 严格贴近官方作业结构的实现
2. 在 baseline 上引入现代 LLM 组件的实现
3. 附带实验记录、踩坑总结、性能分析的实现

代表性仓库：

- Yangliu20：https://github.com/Yangliu20/language_modeling-basics
  - 适合看一个相对干净的 from-scratch baseline。
  - 重点价值：BPE、解码器 Transformer、AdamW、训练管线的最小闭环。

- anenbergb：https://github.com/anenbergb/LLM-from-scratch
  - 明确写出自己是跟着 CS336 做的现代 LLM 实现。
  - 重点价值：RMSNorm、SwiGLU、RoPE、Triton/FlashAttention2、GPU benchmarking、实验日志。

- Melody-Zhou：https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments
  - 结构上更接近“按作业拆开的完整实现集合”。
  - 重点价值：五次作业分目录组织，便于按主题跳读；README 里还串了中文博客文章。

如果你要“看代码学架构”，建议优先关注下面这些问题，而不是只看能不能跑通：

- tokenizer 的 special token 与 merge 规则怎么组织
- attention/MLP 的实现是按教学最小版写，还是已经替换成现代变体
- 训练循环里 checkpoint、resume、eval、日志是否完整
- Triton kernel 的目标到底是“能跑”，还是“能测、能比、能解释瓶颈”

## 独立博客与中文资料

先给一个结论：如果你的目标是“尽量把 CS336 外部课程笔记搜全”，目前能稳定核验到的来源，大致可以分成 5 类：

1. 官方讲义与 handout
2. 个人站点型 lecture notes
3. GitHub 仓库型 lecture notes / study notebook
4. 中文/韩文等二语整理资料
5. 单讲文章型笔记（Medium、博客单篇）

更实用的做法不是追求“全网所有链接”，而是先把 **覆盖完整、可回溯、持续更新** 的主干资源抓住，再把零散文章作为补充。

### 外部课程笔记总表

下表是截至这次检索中，比较值得保留进知识库的“课程笔记型”资源。筛选标准是：

- 能明确看出与 CS336 强相关
- 不是只有作业代码
- 至少在“lecture notes / study notes / summary”层面有稳定内容
- 链接当前仍可访问

| 来源 | 形式 | 覆盖度 | 特点 | 建议用途 |
|---|---|---|---|---|
| 官方 lectures repo | 可执行讲义 | 高 | 原始材料、trace、front-end viewer、最权威 | 一切二手笔记的校对基准 |
| Ran Ding | 独立站点 | 高 | 17 篇 lecture summary + exercise write-up | 快速回顾整门课 |
| Rajdeep Mondal | 独立站点 | 高 | 18 篇系列，按 lecture 顺序展开 | 作为系统化复习主线 |
| Dhyey Mavani | 独立站点 | 中 | 导航式课程笔记，Lecture 01 较完整 | 课程总览和入门 |
| bearbearyu1223 | 独立站点 | 中高 | 按主题连续写，偏工程细节 | 补训练/资源核算/实现细节 |
| 吴敬 | 中文博客 | 中高 | 覆盖推理、评测、数据、scaling 等后半段 | 中文深读与二次整理 |
| HtmMhmd | GitHub + PDF | 中高 | LaTeX/PDF 增强版 lecture summaries | 做打印/归档型材料 |
| feljost | GitHub notes | 中 | 已有多讲 markdown 笔记 | 当作精简 lecture 备忘录 |
| YYZhang2025 | GitHub + 网站 | 中高 | 作业实现强，附 lecture 网站与实验记录 | 看工程实现与实验参数 |
| JhTsin | GitHub 复刻型仓库 | 中高 | 复刻 lecture repo 并补 assignment 进度 | 对照官方材料与个人实现 |
| bigohofone | 韩文整理 | 中 | 韩语整理资料，含 materials 目录 | 多语言交叉参考 |
| Youknowwzh | 中文 notebook | 中 | 中文 study notebook，偏主题化 | 中文入门补充 |

### 1. 覆盖最完整的个人站点笔记

这类资源优先级最高，因为它们通常有更稳定的目录结构和 lecture 序列。

- Ran Ding：https://rd.me/cs336
   - 明确写明是 CS336 Spring 2025 的 summary notes 和 exercise write-ups。
   - 已核验到站点列出了 17 篇 lecture notes，覆盖从 Lecture 1 到后续课程主线。
   - 优势：短、密、回顾效率高。

- Rajdeep Mondal：https://www.rajdeepmondal.com/blog/cs336-overview
   - 已核验到这是一个 18 篇的系列页，从 course overview 到 Lecture 17。
   - 优势：目录完整、lecture 顺序清晰，适合拿来对照课程周进度。
   - 特别适合：你想做“每讲一页总结”的时候当参考骨架。

- bearbearyu1223：https://bearbearyu1223.github.io/posts/cs336-note-get-started/
   - 搜索结果已能稳定识别到多个关联页面，包括 BPE、Transformer 架构、训练 loop、TinyStories 实验、计算成本等主题。
   - 优势：不是只抄 lecture，而是把 lecture 内容转成“工程实现问题”。
   - 注意：该站点个别页面抓取不稳定，适合手工浏览，不适合把它作为唯一来源。

### 2. GitHub 上的 lecture notes / study notes 仓库

这类资源的价值在于“结构稳定、方便收藏、容易二次 fork”。缺点是很多仓库会混入 assignment 代码，质量参差不齐。

- HtmMhmd：https://github.com/HtmMhmd/CS336-Language-Modeling-from-Scratch-notes
   - 已核验到仓库按 `latex-lec1` 到 `latex-lec15` 组织，提供增强版 PDF 总结。
   - 特点：强调 LaTeX 排版、公式、mind map。
   - 适用场景：做归档资料、打印、快速浏览某一讲的结构化总结。

- feljost：https://github.com/feljost/stanford-cs336-notes
   - 已核验到存在 `lecture01.md`、`lecture02.md`、`lecture03.md`、`lecture04.md`、`lecture05.md`、`lecture09.md`、`lecture10.md`。
   - 特点：纯 lecture notes 风格，比较轻量。
   - 适用场景：对照单讲做速读。

- YYZhang2025：https://github.com/YYZhang2025/Stanford-CS336
   - 已核验到 README 同时包含作业实现、训练配置、学习曲线、样例输出，并指向个人 lecture 网站。
   - 特点：更偏“notes + solutions + experiments”三合一。
   - 适用场景：从 lecture 理解落到具体训练配置和实验记录。

- JhTsin：https://github.com/JhTsin/Stanford-CS336-notes-and-assignments-2025
   - 已核验到它包含 `lecture_*.py`、`nonexecutable/`、`trace-viewer/` 和 assignments 目录，整体接近官方 lecture repo 的复刻扩展版。
   - 特点：可以一边看官方式讲义结构，一边看作者自己的 assignment 实现进度与备注。
   - 适用场景：把 lecture materials 和作业实现放在同一仓库里对照阅读。

- marcostx：https://github.com/marcostx/cs336-lecture-notes
   - 已核验到目前覆盖至少 lecture1、lecture2 两个目录。
   - 特点：覆盖暂时不全，但属于真正的 lecture notes，不只是代码仓库。

- 9Skies9：https://github.com/9Skies9/CS336
   - 已核验到这是持续更新中的 notes + homework 仓库，但当前公开信息还不足以确认覆盖深度。
   - 建议：先观察，不作为主干来源。

### 3. 中文与多语言资料

除了英文资料，CS336 的多语言整理已经开始成形，这对后续做中文知识图谱很有帮助。

- 中文：吴敬的系列入口：https://realwujing.github.io/page/3/
   - 已核验到包含 inference、evaluation、data、scaling 等多讲内容。

- 中文：Datawhale 汇总：https://github.com/datawhalechina/cs336-tutorial
   - 更像协作入口，而不是完整 lecture notes。

- 中文：Youknowwzh study notebook：https://github.com/Youknowwzh/StanfordCS336-study-notebook
   - 已核验到包含“大模型概述”“Tokenization”“利用 PyTorch 构建大模型”等中文笔记。

- 韩文：bigohofone：https://github.com/bigohofone/cs336-spring-2025
   - 已核验到仓库包含 `materials` 与 `materials(ko)`，明确是韩语整理资料。

### 4. 单讲文章型笔记

这类资源通常不会覆盖整门课，但在某一讲上可能比仓库型笔记写得更透。

- Medium 单讲笔记样例：https://medium.com/@zgpeace/cs336-lecture-1-what-transfers-to-frontier-models-notes-on-overview-tokenization-68406c5f8846
   - 已核验到这是 Lecture 1 的详细笔记，覆盖课程定位、transferable knowledge、efficiency、BPE 等主题。
   - 特点：适合“先看一讲，试试看作者讲解风格”。

- 个人博客单讲页样例：Rajdeep 的 `cs336-lecture-1` 到 `cs336-lecture-17`
   - 这类属于“系列化单讲文章”，比单仓库 markdown 更适合顺序阅读。

### 5. 不建议纳入主干的条目

这轮检索里也发现了一些不适合进入主干索引的来源：

- 空仓库或未实质更新的仓库
- 只有 assignment code、没有笔记与总结的 repo
- 搜索结果能看到但页面抓取持续失败，且没有稳定目录页可回溯的站点

例如：

- `pengchengneo/CS336_Learn_Notes` 当前为空仓库
- 部分搜索命中的仓库虽然标题写了 notes，但公开元数据不足以证明其内容质量

### 英文导读/课程笔记

- Dhyey Mavani：https://dhyeymavani.com/teaching/stanford-cs336-language-modeling-from-scratch/

特点：更像“课程导航页 + lecture 提要”。适合快速扫一遍课程主线，尤其是 Lecture 01 对课程目标、历史脉络、scaling laws 的概括。

### 中文深度笔记

- 吴敬的 CS336 系列入口：https://realwujing.github.io/page/3/

特点：覆盖了 scaling、inference、evaluation、data 等后半段内容，对中文读者很友好。适合作为“课程听完之后的二次梳理”，尤其是数据处理与推理相关内容。

### 中文协作资料

- Datawhale 教程汇总：https://github.com/datawhalechina/cs336-tutorial

特点：更像一个社区入口，汇总了课程链接、视频入口、中文协作信息。它的价值不在“原创技术深度”，而在“降低中文读者的资源检索成本”。

### 关于个人博客资源的使用方式

你调研中提到的一些个人博客笔记非常有价值，但这类页面可能会遇到静态站点抓取不稳定、链接迁移、内容更新较快的问题。处理方式建议是：

- 把它们视作“踩坑与经验补充”，不要代替官方 handout
- 优先提炼其中的工程问题与解决思路
- 对关键事实仍然回到官方仓库/讲义核对

## 笔记使用策略

面对这么多 CS336 笔记，最常见的问题不是“资料太少”，而是“资料太散”。更稳妥的使用方式是：

### 用官方资源做事实基线

- 课程结构、作业要求、lecture 原意，以官方主页和官方 lectures repo 为准。
- 凡是涉及作业要求、算法定义、实验设定的细节，都要优先对照官方材料。

### 用系列型笔记做主线复习

如果你只想选 2 到 3 套最值得长期跟的外部笔记，我建议：

1. Rajdeep Mondal：适合按 lecture 顺序系统复习
2. Ran Ding：适合高频回顾和速查
3. bearbearyu1223 或吴敬：适合补工程细节与中文理解

### 用 GitHub 仓库做代码联动

- 看 lecture notes 时，旁边最好配一个 assignment 实现仓库。
- 对你这套 AI Infra 笔记而言，最有价值的不是“复制别人的结论”，而是把 lecture 里的概念落到：
   - shape/layout
   - kernel/带宽
   - prefill/decode
   - scaling/data/alignment

## 面向这套 AI 笔记的落地建议

如果目标是继续完善本仓库，而不是单纯收集链接，建议把后续工作拆成 3 层：

1. 资源层：维护这篇 CS336 导读，持续追加高价值外部笔记
2. 结构层：把 CS336 lecture 映射到现有 `operators/`、`models/`、`inference/` 目录
3. 内容层：把 lecture 中真正值得沉淀的知识点转写成你自己的工程笔记

一个简单映射例子：

- Lecture 1-3：补强 tokenizer、Transformer、资源核算
- Lecture 5-8：补强 GPU、kernel、parallelism、roofline
- Lecture 9-11：补强 scaling laws 与训练预算
- Lecture 10：反哺 [../inference/04-llm-serving.md](../inference/04-llm-serving.md)
- Lecture 12-14：补强评测体系与数据工程
- Lecture 15-17：补强对齐与 post-training

## 社区讨论：它解决什么问题

社区讨论最有价值的，不是“总结课程内容”，而是回答下面这些官方资料不一定展开的问题：

- 这门课到底有多硬核，适合什么背景的人上手
- 没有大规模 GPU 资源时，哪些部分能本地做，哪些只能读懂思路
- 某个作业在不同硬件/环境下有哪些现实约束
- 大家常见的训练崩溃、数值稳定、Triton 编译、数据处理瓶颈怎么排

可跟踪入口：

- Reddit LearnMachineLearning：https://www.reddit.com/r/learnmachinelearning/comments/1lxgabn/stanfords_cs336_2025_language_modeling_from/
- Reddit LocalLLaMA：https://www.reddit.com/r/LocalLLaMA/comments/1lxgb9q/stanfords_cs336_2025_language_modeling_from/
- 跨课程/组队讨论样例：https://www.reddit.com/r/LLMDevs/comments/1kw5gh7/looking_for_2_people_to_study_kaists_diffusion/

阅读姿势：把它们当作“工程反馈流”，不要把其中的主观判断直接当结论。

## 推荐学习路径

如果你的目标是服务于“AI Infra / 底层原理 / 面试复习”，推荐按下面顺序读：

1. 先打牢这套笔记里的基础：
   - [../models/01-transformer-minimum.md](../models/01-transformer-minimum.md)
   - [../models/02-attention-kv-cache.md](../models/02-attention-kv-cache.md)
   - [../operators/02-kernels-and-parallelism.md](../operators/02-kernels-and-parallelism.md)
   - [../operators/03-memory-hierarchy-and-roofline.md](../operators/03-memory-hierarchy-and-roofline.md)
   - [../inference/01-inference-stack-overview.md](../inference/01-inference-stack-overview.md)

2. 再对照 CS336 的课程顺序补齐“训练侧”视角：
   - tokenization
   - architecture/hyperparameters
   - GPU / Triton / parallelism
   - scaling laws
   - evaluation / data / alignment

3. 然后只选一个开源实现做深读：
   - 如果你要最小闭环，看 Yangliu20
   - 如果你要现代工程实现，看 anenbergb
   - 如果你要按作业逐个对照，看 Melody-Zhou

4. 最后把社区博客当“查漏补缺”：
   - 查环境坑
   - 查数值稳定推导
   - 查数据处理策略
   - 查训练/实验经验

## 易错点

- 把 CS336 当成“再学一遍 Transformer 原理”，忽略它其实更强调系统与工程实现。
- 一开始就钻 Triton/分布式实现，结果 tokenizer、训练循环、资源核算这些前置直觉没建立起来。
- 直接抄社区代码，导致失去“自己搭最小闭环”的训练价值。
- 只看课程前半段模型部分，不看 data、evaluation、alignment，最后对真实 LLM pipeline 的理解仍然是断裂的。
- 把社区里的经验贴当作标准答案，没有回到官方 handout 和仓库核对。

## 学习 checklist

- [ ] 我能说清楚这门课五次作业分别在练什么。
- [ ] 我能从零画出一个最小 LM 训练流程：文本 → tokenizer → ids → Transformer → loss → optimizer。
- [ ] 我知道 prefill/decode、带宽/算力、kernel/通信、训练/推理 这些视角为什么会在 CS336 中反复出现。
- [ ] 我至少完整读过一个官方 assignment README/handout，而不是只看二手总结。
- [ ] 我至少选过一个社区实现，对照过模型结构、训练脚本或系统优化部分。

## 参考资料

- Stanford CS336 主页：https://cs336.stanford.edu/
- Stanford CS336 Spring 2025 存档：https://cs336.stanford.edu/spring2025/
- Stanford CS336 lectures：https://github.com/stanford-cs336/spring2025-lectures
- Stanford CS336 GitHub 组织页：https://github.com/stanford-cs336
- CS336 YouTube 播放列表：https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_
- Ran Ding CS336 notes：https://rd.me/cs336
- Rajdeep Mondal CS336 notes：https://www.rajdeepmondal.com/blog/cs336-overview
- Jacek Dwojak CS336 study notes：https://jacekdwojakpl.github.io/cs336-study/about/
- Yangliu20 language_modeling-basics：https://github.com/Yangliu20/language_modeling-basics
- anenbergb LLM-from-scratch：https://github.com/anenbergb/LLM-from-scratch
- Melody-Zhou stanford-cs336-spring2025-assignments：https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments
- YYZhang2025 Stanford-CS336：https://github.com/YYZhang2025/Stanford-CS336
- HtmMhmd enhanced lecture summaries：https://github.com/HtmMhmd/CS336-Language-Modeling-from-Scratch-notes
- feljost stanford-cs336-notes：https://github.com/feljost/stanford-cs336-notes
- JhTsin Stanford-CS336-notes-and-assignments-2025：https://github.com/JhTsin/Stanford-CS336-notes-and-assignments-2025
- bigohofone cs336-spring-2025：https://github.com/bigohofone/cs336-spring-2025
- Youknowwzh StanfordCS336-study-notebook：https://github.com/Youknowwzh/StanfordCS336-study-notebook
- Dhyey Mavani CS336 notes：https://dhyeymavani.com/teaching/stanford-cs336-language-modeling-from-scratch/
- bearbearyu1223 CS336 notes entry：https://bearbearyu1223.github.io/posts/cs336-note-get-started/
- Rajdeep Mondal Lecture 1 entry：https://www.rajdeepmondal.com/blog/cs336-lecture-1
- Medium lecture note sample：https://medium.com/@zgpeace/cs336-lecture-1-what-transfers-to-frontier-models-notes-on-overview-tokenization-68406c5f8846
- 吴敬的 CS336 相关文章入口：https://realwujing.github.io/page/3/
- Datawhale CS336 tutorial：https://github.com/datawhalechina/cs336-tutorial

## 备注

如果你是按“课程作业”而不是按“公开资料学习”来推进，官方 honor code 明确不鼓励直接参考现成实现。因此更合理的顺序是：**先自己做，再把社区实现作为复盘材料**。