# STL：容器、迭代器、算法

## 要点

- 算法更重要：优先用 `std::algorithm` 表达意图
- 复杂度意识：常见容器操作的 big-O 要熟

## 容器选型速记

- `vector`：首选；连续内存；尾插摊还 O(1)
- `deque`：两端插入更友好；不连续
- `list/forward_list`：节点式；迭代器稳定但缓存不友好
- `unordered_map`：哈希；平均 O(1)；注意 rehash
- `map`：红黑树；有序；O(log n)

## 迭代器失效

- `vector` 扩容会使指针/引用/迭代器失效
- `unordered_*` rehash 会使迭代器失效

## 面试追问

- 为什么 `vector` 常常比 `list` 更快？
- `emplace` 一定比 `push` 更快吗？
