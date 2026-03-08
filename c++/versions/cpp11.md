# C++11 新特性速记

## 你应该优先掌握的（高频）

- 语言：`auto`、`nullptr`、范围 for、`enum class`、`override/final`、lambda、右值引用与移动语义
- 库：智能指针（`unique_ptr/shared_ptr`）、`std::thread`/mutex/cv、`std::chrono`

## 语言层面

- **移动语义**：减少不必要拷贝；理解 move 后对象的“有效但未指定”状态
- **lambda**：捕获列表（值/引用/`this`）、可变 lambda（`mutable`）
- **统一初始化**：`{}` 初始化；注意与 `initializer_list` 的重载竞争

## 标准库层面

- 智能指针、并发库、正则（`<regex>`）、tuple 等

## 易错点

- 误用 `std::move` / 在仍需使用对象时 move
- `{}` 初始化导致调用了 `initializer_list` 构造

## 最小例子（建议你后续补上能编译的代码）

- 移动构造/移动赋值的一个小类
- lambda 捕获与生命周期
