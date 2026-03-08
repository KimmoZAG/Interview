# C++17 新特性速记

## 高频

- 结构化绑定（`auto [a,b] = ...`）
- `if constexpr`
- `std::optional` / `std::variant` / `std::any`
- `std::filesystem`（实现/编译器支持度通常较好）

## 语言

- **结构化绑定**：理解绑定的是拷贝还是引用（`auto& [x,y]`）
- **`if constexpr`**：模板分支在编译期裁剪

## 库

- `optional`：表达“可能没有值”
- `variant`：类型安全联合体；配合 `std::visit`

## 易错点

- 结构化绑定默认是拷贝，导致性能/语义意外
- `variant` 的异常与 `valueless_by_exception`
