---
tags:
  - C++
  - 系统编程
  - STL
  - 并发
description: 现代 C++ 工程实践知识体系，覆盖 C++11 至 C++23。
---

# C++

面向现代 C++ 工程实践的知识体系，覆盖 **C++11 至 C++23**，以工程师视角精简提炼。

---

<div class="grid cards" markdown>

- :material-wrench: **通用知识**

    ---

    工具链、对象生命周期、RAII、模板、STL、并发基础、运行时多态。

    [:octicons-arrow-right-24: 进入](general/01-toolchain-and-build.md)

- :material-timeline: **版本特性**

    ---

    C++11 到 C++23 各版本引入的核心语言与库特性速查。

    [:octicons-arrow-right-24: C++11](versions/cpp11.md) · [:octicons-arrow-right-24: C++17](versions/cpp17.md) · [:octicons-arrow-right-24: C++20](versions/cpp20.md)

- :material-bookshelf: **附录**

    ---

    术语表、常见坑位清单、参考资料。

    [:octicons-arrow-right-24: 术语表](appendix/glossary.md)

</div>

---

## 通用知识

| 篇目 | 核心内容 |
|---|---|
| [工具链与构建](general/01-toolchain-and-build.md) | 编译器、链接器、CMake、预编译头 |
| [对象生命周期与值语义](general/02-object-lifetime-and-value-semantics.md) | 构造/析构顺序、移动语义、copy-elision |
| [RAII 与智能指针](general/03-raii-and-smart-pointers.md) | `unique_ptr`、`shared_ptr`、`weak_ptr`、自定义 deleter |
| [模板与类型](general/04-templates-and-types.md) | template、concept、SFINAE、`type_traits` |
| [STL 容器/迭代器/算法](general/05-stl-containers-iterators-algorithms.md) | 容器选择指南、范围算法、自定义比较器 |
| [并发基础](general/06-concurrency-basics.md) | `thread`、`mutex`、`atomic`、memory order |
| [运行时多态](general/07-runtime-polymorphism.md) | vtable、虚函数、多态设计与开销 |

## 推荐按“能力链”复习

如果你是为了面试或工程表达，建议不要把 C++ 读成语法清单，而是按下面几条链来过。

### 资源管理主线

`[对象生命周期与值语义](general/02-object-lifetime-and-value-semantics.md)`
→ `[RAII 与智能指针](general/03-raii-and-smart-pointers.md)`
→ `[并发基础](general/06-concurrency-basics.md)`

这条链适合回答：

- 对象什么时候活、什么时候死；
- 资源谁拥有、谁释放；
- 多线程下为什么生命周期和所有权问题会更危险。

### 泛型与抽象主线

`[模板与类型](general/04-templates-and-types.md)`
→ `[运行时多态](general/07-runtime-polymorphism.md)`

这条链适合回答：

- 编译期多态和运行时多态分别适合什么场景；
- 为什么模板代码快但复杂，虚函数接口稳但有运行时边界；
- 什么时候该用 Concepts / CRTP，什么时候该保留抽象基类。

### 工程面试高频 5 篇

如果时间有限，优先过这 5 篇：

1. [对象生命周期与值语义](general/02-object-lifetime-and-value-semantics.md)
2. [RAII 与智能指针](general/03-raii-and-smart-pointers.md)
3. [模板与类型](general/04-templates-and-types.md)
4. [并发基础](general/06-concurrency-basics.md)
5. [运行时多态](general/07-runtime-polymorphism.md)

这 5 篇已经基本覆盖：**生命周期、所有权、泛型、同步、多态**——也就是最常被连环追问的一组基础话题。

## 版本特性速览

| 版本 | 核心新增 |
|---|---|
| [C++11](versions/cpp11.md) | move 语义、lambda、`auto`、range-for、智能指针、线程库 |
| [C++14](versions/cpp14.md) | 泛型 lambda、变量模板、放宽的 `constexpr` |
| [C++17](versions/cpp17.md) | structured bindings、`if constexpr`、`optional` / `variant` / `any` |
| [C++20](versions/cpp20.md) | concepts、ranges、coroutines、modules、三路比较 `<=>` |
| [C++23](versions/cpp23.md) | `std::expected`、ranges 增强、deducing `this` |

## 读 C++ 这套材料时建议一直带着的问题

为了避免把 C++ 学成“关键字背诵比赛”，建议每篇都追问：

1. 这个机制在解决什么工程问题？
2. 它改善的是性能、安全性、可维护性，还是表达能力？
3. 它带来的代价是什么：复杂度、编译时间、运行时开销，还是调试难度？
4. 有没有更简单的替代方案，比如值语义、标准库容器、RAII 或更清晰的接口设计？

## 附录

| 文档 | 用途 |
|---|---|
| [术语表](appendix/glossary.md) | 关键术语速查 |
| [坑位清单](appendix/gotchas.md) | 常见陷阱与反直觉行为 |
| [参考资料](appendix/references.md) | 书籍、文章、标准草案链接 |
