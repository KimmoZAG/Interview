# 并发基础

## 要点

- 区分：并发（concurrency） vs 并行（parallelism）
- 共享数据必须有同步：互斥量/原子/条件变量

## 线程与同步原语

- `std::thread`：线程启动与 join/detach
- `std::mutex` / `std::lock_guard` / `std::unique_lock`
- `std::condition_variable`
- `std::atomic`：只在需要时用；理解内存序（先掌握默认顺序一致）

## 常见问题

- 数据竞争（data race）
- 死锁：锁顺序、锁粒度
- 虚假唤醒：`cv.wait(lock, pred)`

## 面试追问

- 原子与互斥分别适用于什么场景？
- `lock_guard` vs `unique_lock`
