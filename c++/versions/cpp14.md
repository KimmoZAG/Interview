# C++14 新特性速记

## 高频

- 泛型 lambda（lambda 参数用 `auto`）
- 返回类型推导增强（`auto` 返回）
- `std::make_unique`

## 语言

- **泛型 lambda**：写法更简洁，但注意捕获与模板实例化错误定位

## 库

- `std::make_unique`：避免手写 `new`，异常安全更好

## 易错点

- `auto` 返回推导有时会丢掉引用/const（需要时用 `decltype(auto)`）
