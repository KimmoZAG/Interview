# RAII 与智能指针

## 要点

- RAII：资源的获取与释放绑定对象生命周期（构造获取、析构释放）
- 能不用裸 `new/delete` 就不用；优先值类型与标准库容器

## `std::unique_ptr`

- 独占所有权；不可拷贝、可移动
- 支持自定义 deleter（管理 FILE*/socket 等）

## `std::shared_ptr` / `std::weak_ptr`

- 共享所有权（引用计数）
- 循环引用用 `weak_ptr` 打破

## 何时用哪个

- 默认：`unique_ptr`
- 确实需要共享生命周期：`shared_ptr`
- 只观察不延长生命周期：`weak_ptr`

## 面试追问

- `make_unique`/`make_shared` 为什么更推荐？
- `shared_ptr` 的控制块是什么？线程安全边界在哪里？
