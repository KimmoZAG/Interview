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

## 版本特性速览

| 版本 | 核心新增 |
|---|---|
| [C++11](versions/cpp11.md) | move 语义、lambda、`auto`、range-for、智能指针、线程库 |
| [C++14](versions/cpp14.md) | 泛型 lambda、变量模板、放宽的 `constexpr` |
| [C++17](versions/cpp17.md) | structured bindings、`if constexpr`、`optional` / `variant` / `any` |
| [C++20](versions/cpp20.md) | concepts、ranges、coroutines、modules、三路比较 `<=>` |
| [C++23](versions/cpp23.md) | `std::expected`、ranges 增强、deducing `this` |

## 附录

| 文档 | 用途 |
|---|---|
| [术语表](appendix/glossary.md) | 关键术语速查 |
| [坑位清单](appendix/gotchas.md) | 常见陷阱与反直觉行为 |
| [参考资料](appendix/references.md) | 书籍、文章、标准草案链接 |
