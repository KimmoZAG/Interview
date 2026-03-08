# 模板与类型系统速查

## 要点

- 模板是“编译期代码生成”机制；错误往往在实例化时爆炸
- 理解：类型推导、特化/偏特化、SFINAE/约束（C++20 Concepts）

## 你需要能讲清楚的概念

- `typename` vs `class`（模板参数位置等价；依赖名需要 `typename`）
- `decltype` / `auto` / `decltype(auto)`
- 转发引用（`T&&` 在模板推导场景）与 `std::forward`

## 常见模板坑

- 两阶段查找（two-phase lookup）
- ADL（参数依赖查找）
- `std::initializer_list` 参与重载决议导致的“意外匹配”

## 面试追问

- 解释“完美转发”与引用折叠规则
- `auto` 推导与 `decltype(auto)` 的差异
