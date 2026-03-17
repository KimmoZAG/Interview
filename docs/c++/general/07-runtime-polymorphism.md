---
title: 07 运行时多态（Runtime Polymorphism）
---

# 运行时多态（Runtime Polymorphism）

本文记录 C++ 中运行时多态的要点与示例，侧重于虚函数、纯虚函数、抽象类、虚析构、对象切片等常见主题与陷阱。所有示例面向现代 C++（含 `override` / `final`）。

## 概要
- 运行时多态是指通过基类的指针或引用在运行期间调用派生类的重写方法。
- 依赖关键字：`virtual`、`override`、`final`；机制基于 vtable（虚表）与虚指针（vptr）。

## 核心概念
- 虚函数（virtual function）：在基类声明为 `virtual`，允许派生类重写并通过基类指针/引用发生动态绑定。
- 纯虚函数（pure virtual）：`virtual void f() = 0;`，使类成为抽象类，无法实例化。
- 抽象类（abstract class）：包含至少一个纯虚函数的类。
- 虚析构函数：若基类通过指针删除派生类对象，基类应有虚析构以保证派生析构被调用。

## 语法与典型示例

简单示例：基类指针调用派生重写方法

```cpp
#include <iostream>

class Base {
public:
    virtual void speak() { std::cout << "Base\n"; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void speak() override { std::cout << "Derived\n"; }
};

int main() {
    Base* b = new Derived();
    b->speak(); // 输出 "Derived"（动态绑定）
    delete b;   // 调用 Derived 的析构（因为基类析构是 virtual）
}
```

说明：如果 `Base` 的析构非虚，则 `delete b;` 会导致未定义行为（通常是资源泄漏或未调用派生析构）。

## 纯虚函数与抽象类

```cpp
class Shape {
public:
    //纯虚函数的定义方式：在函数声明后面加上 "= 0"， const 表示该函数不会修改对象状态
    virtual double area() const = 0; // 纯虚函数
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    double r;
    Circle(double r): r(r) {}
    double area() const override { return 3.14159 * r * r; }
};

// Shape s; // 错误：不可实例化抽象类
```

备注：纯虚函数可以有实现（少见），但仍使类抽象：

```cpp
class A {
public:
    virtual void f() = 0;
};
void A::f() { /* 可被派生类显式调用 */ }
```

## `override` 与 `final`
- `override`：显式指出该方法意图重写基类虚方法，编译器会检查签名是否匹配，推荐总是使用。
- `final`：阻止进一步重写：`void foo() final;` 或 `class D final { ... };`

示例：签名不匹配会因 `override` 报错

```cpp
class B {
public:
    virtual void f(int) {}
};

class C : public B {
public:
    void f(float) override {} // 编译错误：没有正确重写基类 f(int)
};
```

## 对象切片（Object Slicing）
- 当把派生类对象赋给基类对象时，切掉派生部分，之后通过基类对象无法实现派生的多态行为。

```cpp
class Base {
public:
    virtual void speak() { std::cout << "Base\n"; }
};

class Derived : public Base {
public:
    void speak() override { std::cout << "Derived\n"; }
};

Derived d;
Base b = d; // 对象切片，b 仅为 Base 部分
b.speak();  // 调用 Base::speak
```

解决办法：使用指针或引用代替按值传递。

## 虚析构的重要性
- 若类被用作基类（通过指针删除派生对象），必须声明虚析构。
- 否则派生类的析构不会被调用，造成资源泄漏或未定义行为。

示例（错误）：

```cpp
class NoVirtualDtor {
public:
    ~NoVirtualDtor() { /* ... */ }
};

class Child : public NoVirtualDtor {
public:
    ~Child() { /* 释放资源 */ }
};

NoVirtualDtor* p = new Child;
delete p; // 未定义行为：Child::~Child 可能不被调用
```

## 协变返回类型（Covariant Return Types）
- 派生类可以在重写虚函数时返回与基类相同或派生于基类返回类型的指针/引用。

```cpp
class Base {
public:
    virtual Base* clone() const { return new Base(*this); }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    Derived* clone() const override { return new Derived(*this); }
};
```

## RTTI 与 `dynamic_cast`
- `dynamic_cast` 可用于在运行时安全地向下转换基类指针/引用（需至少含一个虚函数，以启用 RTTI）。

```cpp
Base* b = new Derived;
if (Derived* d = dynamic_cast<Derived*>(b)) {
    // 成功转换
}
```

## 性能与实现观测
- 虚函数调用比直接调用有一点开销（通过 vptr 查找 vtable，再调用），但现代优化器与 CPU 缓存使得实际开销通常较小。
- 可通过剖析/基准测量评估是否为性能瓶颈。对于频繁调用的短函数，考虑 `final` 或静态多态（CRTP）替代。

## 常见陷阱与注意事项
- 忘记 `virtual` 析构导致资源泄漏。
- 使用 `override` 可以提前捕获签名错误。
- 对象切片：避免按值传递基类而传入派生对象。
- 在析构函数中调用虚函数：在对象正在析构时，调用会绑定到当前层级的类（即派生部分已被析构），通常不是预期行为。

## 例子合集（更多情况）

1) 在构造/析构期间的虚调用

```cpp
class A {
public:
    A() { foo(); }
    virtual void foo() { std::cout << "A::foo\n"; }
    virtual ~A() = default;
};

class B : public A {
public:
    void foo() override { std::cout << "B::foo\n"; }
};

// B b; // 构造时只会调用 A::foo（B 子对象尚未构造）
```

2) 接口类 + 非成员工厂函数

```cpp
\#include <memory>

class IWorker {
public:
    virtual void work() = 0;
    virtual ~IWorker() = default;
};

class WorkerA : public IWorker {
public:
    void work() override { /* ... */ }
};

std::unique_ptr<IWorker> makeWorkerA() { return std::make_unique<WorkerA>(); }
```

3) CRTP 与静态多态（对比）

```cpp
template <typename Derived>
class CRTPBase {
public:
    void interface() { static_cast<Derived*>(this)->implementation(); }
};

class Impl : public CRTPBase<Impl> {
public:
    void implementation() { /* ... */ }
};
```

## 小结
- 运行时多态是面向对象设计的重要工具，便于通过统一接口处理不同实现。使用 `virtual`/`override`/`final` 能提高可维护性并减少错误。
- 常见要点：虚析构、避免对象切片、慎用构造/析构期间的虚调用、使用 `override` 保护签名一致性。

---

## 内存布局与 `sizeof` 分析

- vptr（虚指针）通常是每个含虚函数对象的一部分（编译器实现细节），大小等于指针大小（例如在 x86_64 上通常为 8 字节）。
- `sizeof` 的结果依赖于成员、对齐与填充（padding），以及是否有虚表指针。

示例：检查对象大小

```cpp
#include <iostream>

class B {
public:
    virtual void f() {}
    virtual ~B() = default;
};

class D : public B {
public:
    int x;
    double y;
};

int main() {
    std::cout << "sizeof(B) = " << sizeof(B) << '\n';
    std::cout << "sizeof(D) = " << sizeof(D) << '\n';
}
```

说明：
- 常见输出（x86_64）：`sizeof(B) = 8`（仅 vptr），`sizeof(D) = 24`（vptr + `int` + `double` + 填充），但具体值取决于编译器及 ABI。
- 空类（empty class）通常至少占 1 字节，但若含虚函数则还会包含 vptr，因此大小会更大。
- 虚继承会增加对象大小（用于存储基子对象位置的额外指针/偏移）。

如何分析：
- 使用 `sizeof` 查看在目标平台上的实际大小。
- 使用 `alignof` 与 `offsetof`（在 POD/标准布局类型上）理解对齐与成员偏移。
- 注意：C++ 标准并不规定 vptr 的存在或位置，只有行为语义；各实现细节由编译器/平台决定。

内存影响与实践建议：
- 如果大量小对象需要多态行为，留意每对象的额外 vptr 开销；可考虑对象池、共享指针或将状态外置以减少内存占用。
- 对性能敏感的热路径，测量虚调用与内存占用影响，必要时用 CRTP 或显式分派替代运行时多态。

## `struct` 与 `class` 的区别（一定要搞懂）

先给结论：在 C++ 里，`struct` 和 `class` **几乎是同一种东西**（都属于“类类型”），它们的区别只有两点“默认值”。

### 区别 1：成员默认访问权限
- `struct`：默认 `public`
- `class`：默认 `private`

对比示例：

```cpp
struct S {
    int x; // 默认 public
};

class C {
    int x; // 默认 private
};

int main() {
    S s;
    s.x = 1; // OK

    C c;
    // c.x = 1; // 编译错误：x 是 private
}
```

记忆法：`struct` 更像“数据结构”，字段默认公开；`class` 更像“封装的类”，字段默认私有。

### 区别 2：继承默认访问控制
- `struct Derived : Base` 等价于 `struct Derived : public Base`
- `class Derived : Base` 等价于 `class Derived : private Base`

这会直接影响“能不能当作基类指针/引用使用”：

```cpp
class Base { public: virtual ~Base() = default; };

class Derived1 : Base { /* 注意：这里是 private 继承 */ };
// Base* p1 = new Derived1; // 编译错误：private 继承下，Derived1 不能隐式转换为 Base*

class Derived2 : public Base { /* public 继承才是面向对象里的 is-a */ };
Base* p2 = new Derived2; // OK
delete p2;
```

### 所以，什么时候用哪个？
- 你这篇“运行时多态”笔记，主线更建议用 `class`：因为多态通常伴随“接口 + 封装 + 继承层次”，`class` 的语境更自然。
- `struct` 也完全可以写多态代码（语法不受限），但更常用在“以数据为主”的类型：例如坐标、参数包、聚合配置。

一句话总结：
- **语义/机制**：`struct` 和 `class` 没差。
- **默认值**：`struct` 默认 `public` / `public` 继承；`class` 默认 `private` / `private` 继承。
- **风格**：多态层次用 `class` 更常见；数据载体用 `struct` 更常见。

---

如果你希望，我可以：
- 把上面的 `sizeof` 示例做成一个可编译并运行的最小项目（含 `CMakeLists.txt` 或单文件指令），并在 Windows 上用 MSVC/MinGW 测试一次。
- 在 `versions/` 目录中说明不同标准（如 C++11/14/17/20）对多态相关特性的历史变更（若需）。

