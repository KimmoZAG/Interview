# 最小可运行 C++ 示例

该示例包含一个简单的 `Hello, world!` 程序，位于 `main.cpp`。

快速运行方式：

- 使用 `g++`（MinGW 或 WSL）：

```bash
g++ -std=c++17 main.cpp -O2 -o minimal_test
./minimal_test    # 在 WSL 或类 Unix 环境
```

在 Windows PowerShell（使用 MinGW-w64 的 `g++.exe`）：

```powershell
g++ -std=c++17 main.cpp -O2 -o minimal_test.exe
.\\minimal_test.exe
```

- 使用 CMake（推荐用于更复杂项目）：

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
./minimal_test    # Windows 下为 .\\minimal_test.exe
```

文件列表：

- `main.cpp` — 程序源码
- `CMakeLists.txt` — CMake 构建文件
