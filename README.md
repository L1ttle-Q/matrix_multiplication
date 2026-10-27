# matrix_multiplication

C++ / CUDA 方阵乘法实验：比较 CPU 循环重排、展开、分块、OpenMP，以及 GPU 基础核、共享内存优化和 cuBLAS。程序内置随机数据生成、结果比对和计时。

## 环境

- Git、CMake 3.18+、支持 C++17 的编译器（源码使用 `if constexpr`）。
- CPU：可用的 OpenMP C++ 运行库。
- GPU：CUDA Toolkit（含 cuBLAS）、兼容的宿主编译器和 NVIDIA GPU。默认目标架构为 `89`，可通过 CMake 参数覆盖。

## 只构建 CPU（不需要 CUDA）

Windows + MinGW-w64：

```powershell
git clone https://github.com/L1ttle-Q/matrix_multiplication.git
cd matrix_multiplication
cmake -S . -B build-cpu -G "MinGW Makefiles" -DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
$env:OMP_NUM_THREADS = "4"
.\build-cpu\cpu_matrix_multiplication.exe
```

Linux 可省略 `-G "MinGW Makefiles"`，运行 `OMP_NUM_THREADS=4 ./build-cpu/cpu_matrix_multiplication`。macOS 编译器需自行配置 OpenMP；本轮未实测该平台。

## 构建 CPU 和 GPU

Windows 的已验证组合为 Visual Studio 2022 x64 + CUDA 12.8。不要复用 MinGW 的构建目录：

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release --parallel
.\build-cuda\Release\cpu_matrix_multiplication.exe
.\build-cuda\Release\gpu_matrix_multiplication.exe
```

将 `89` 换为目标 GPU 支持的架构编号；多个架构可用引号包裹、分号分隔。Linux + CUDA 可使用：

```sh
cmake -S . -B build-cuda -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel
./build-cuda/gpu_matrix_multiplication
```

没有 CUDA 时必须显式设 `BUILD_CUDA=OFF`。`execute.bat` 是历史 Debug 启动脚本，性能比较请使用 Release。

## 程序行为和输出

无需输入矩阵或传入参数，直接启动即可。尺寸和类型在 `include/global.h` 中定义，当前为 **1024×1024、float、block size 32**。随机输入使用时间作为种子。

每个方法运行 7 次，逐次比对参考结果并打印耗时；虽然汇总标签写作 `average`，实际输出是排序后的**中位数**。CPU 单位为毫秒，GPU 为微秒。GPU 计时围绕核执行，不能直接理解为包含全部分配、主机到设备和设备到主机复制的端到端耗时。结果不匹配会打印 `Not correct result!` 并以异常终止。

## 校验与已知限制

- CPU 有 10 个方法，GPU 有 6 个自定义核；`USE_FLOAT` 时还运行 cuBLAS。现有程序的比对属于基准程序内置检查，并非独立、确定性的单元测试。
- 部分方法采用同一参考实现及正数随机数据，未覆盖负数、NaN、无穷、零参考值等浮点边界。
- 优化实现依赖方阵和可整除尺寸：修改尺寸时需同时核对展开步长、四等分分块、GPU 网格及 block size，不能假定支持任意矩形或尾块。
- MSVC 的传统 OpenMP 模式会忽略部分 `collapse` 子句和展开提示；构建可通过，但不能据此假定与 GCC 的线程分配一致。
- `No CUDA toolset found` 应检查 CUDA 与 Visual Studio 集成；OpenMP 查找失败应检查编译器及对应运行库；GPU 架构不匹配应重新配置 `CMAKE_CUDA_ARCHITECTURES`。
- 本次通过 CMake 的 C++17 特性和 OpenMP 导入目标配置编译器，添加 `BUILD_CUDA` 开关，并使用 `CUDA::cublas` 链接依赖。

## 目录与验证

`src/cpu_main.cpp` / `src/gpu_main.cu` 是两个入口，`src/matrix_cpu.cpp` / `src/matrix_gpu.cu` 是实现，`src/matrix.cpp` 提供数据生成、参考乘法和比对，`include/` 包含尺寸、计时器和接口。

2026-10-26：Windows 下 MinGW-w64 GCC 8.1 的 CPU Release 构建通过；MSVC 19.42 + CUDA 12.8、架构 89 的 CPU/GPU Release 构建通过。本机 RTX 4070 Ti SUPER 上，CPU 的 10 个方法和 GPU 的 7 个方法（含 cuBLAS）均各执行 7 次并通过内置比对；CPU 使用 `OMP_NUM_THREADS=4`。Linux/macOS 未实测。不同硬件、编译选项及线程数下的耗时不可直接比较。

使用 [Apache License 2.0](LICENSE)。
