# C++23 新特性速记

## 高频（建议优先跟进）

- `std::expected`（错误处理：值或错误）
- `std::print` / `std::println`（格式化输出，依赖实现）
- `std::mdspan`（多维视图）
- `std::stacktrace`（调试辅助，依赖实现）

## 易错点

- 新库特性在不同标准库实现上的可用性差异（libstdc++/libc++/MSVC STL）
- 头文件/链接开关与编译器版本要求

## 你可以补的最小例子

- `expected<T, E>` 的成功/失败分支
- `mdspan` 访问与布局
